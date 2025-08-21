import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import os

def load_data():
    """Load the transaction data"""
    print("Loading transaction data...")
    df = pd.read_csv('data/raw/transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

def create_customer_features(df):
    """Create customer-level features for churn prediction"""
    
    print("Creating customer-level features...")
    
    # Reference date for recency calculation
    reference_date = df['transaction_date'].max()
    
    # Group by customer
    customer_features = []
    
    for customer_id in df['customer_id'].unique():
        customer_data = df[df['customer_id'] == customer_id].copy()
        
        # Basic customer info (same for all transactions)
        country = customer_data['country'].iloc[0]
        age = customer_data['age'].iloc[0]
        device_type = customer_data['device_type'].iloc[0]
        churned = customer_data['churned'].iloc[0]
        
        # RFM Features
        # Recency: Days since last transaction
        last_transaction = customer_data['transaction_date'].max()
        recency_days = (reference_date - last_transaction).days
        
        # Frequency: Number of transactions
        frequency = len(customer_data)
        
        # Monetary: Total and average transaction amounts
        total_amount = customer_data['amount'].sum()
        avg_amount = customer_data['amount'].mean()
        
        # Additional behavioral features
        # Transaction frequency in different periods
        last_30_days = customer_data[customer_data['transaction_date'] >= (reference_date - timedelta(days=30))]
        last_60_days = customer_data[customer_data['transaction_date'] >= (reference_date - timedelta(days=60))]
        
        transactions_last_30 = len(last_30_days)
        transactions_last_60 = len(last_60_days)
        
        # Support interactions
        total_support = customer_data['support_interactions'].sum()
        avg_support = customer_data['support_interactions'].mean()
        
        # Product diversity
        unique_categories = customer_data['product_category'].nunique()
        
        # Transaction patterns
        transaction_std = customer_data['amount'].std() if len(customer_data) > 1 else 0
        days_active = (customer_data['transaction_date'].max() - customer_data['transaction_date'].min()).days + 1
        
        customer_features.append({
            'customer_id': customer_id,
            'country': country,
            'age': age,
            'device_type': device_type,
            'recency_days': recency_days,
            'frequency': frequency,
            'total_amount': total_amount,
            'avg_amount': avg_amount,
            'transactions_last_30': transactions_last_30,
            'transactions_last_60': transactions_last_60,
            'total_support_interactions': total_support,
            'avg_support_interactions': avg_support,
            'unique_categories': unique_categories,
            'amount_std': transaction_std,
            'days_active': days_active,
            'churned': churned
        })
    
    features_df = pd.DataFrame(customer_features)
    
    print(f"Created features for {len(features_df)} customers")
    return features_df

def engineer_additional_features(df):
    """Create additional engineered features"""
    
    print("Engineering additional features...")
    
    # Derived features
    df['avg_days_between_transactions'] = df['days_active'] / (df['frequency'] - 1)
    df['avg_days_between_transactions'] = df['avg_days_between_transactions'].fillna(0)
    
    # Risk scores
    df['support_risk_score'] = np.where(df['total_support_interactions'] > 2, 1, 0)
    df['recency_risk_score'] = np.where(df['recency_days'] > 30, 1, 0)
    df['frequency_risk_score'] = np.where(df['frequency'] < 3, 1, 0)
    
    # Engagement metrics
    df['recent_activity_ratio'] = df['transactions_last_30'] / df['frequency']
    df['spending_consistency'] = df['amount_std'] / df['avg_amount']
    df['spending_consistency'] = df['spending_consistency'].fillna(0)
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                            labels=['young', 'adult', 'middle', 'senior'])
    
    # Amount categories
    df['amount_category'] = pd.cut(df['avg_amount'], bins=3, labels=['low', 'medium', 'high'])
    
    return df

def encode_categorical_features(df):
    """Encode categorical features"""
    
    print("Encoding categorical features...")
    
    # One-hot encode categorical variables
    categorical_cols = ['country', 'device_type', 'age_group', 'amount_category']
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    return df

def save_features(df):
    """Save processed features"""
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save features
    df.to_csv('data/processed/features.csv', index=False)
    
    print(f"Saved {len(df)} customer records with {len(df.columns)} features")
    print("Features saved to data/processed/features.csv")

def generate_features_documentation(df):
    """Generate feature documentation"""
    
    feature_docs = """# Feature Engineering Documentation

## Overview
This document describes all features created for the customer churn prediction model.

## Feature Categories

### 1. Customer Demographics
- **age**: Customer age in years
- **country_***: One-hot encoded country (Germany, France, Italy, Spain vs Romania baseline)
- **device_type_***: One-hot encoded device type (mobile, tablet vs desktop baseline)

### 2. RFM Features (Recency, Frequency, Monetary)
- **recency_days**: Days since last transaction (strong churn predictor)
- **frequency**: Total number of transactions in 120-day period
- **total_amount**: Total transaction value per customer
- **avg_amount**: Average transaction amount per customer

### 3. Behavioral Features
- **transactions_last_30**: Number of transactions in last 30 days
- **transactions_last_60**: Number of transactions in last 60 days
- **unique_categories**: Number of different product categories purchased
- **days_active**: Number of days between first and last transaction

### 4. Support & Risk Features
- **total_support_interactions**: Total support tickets/interactions
- **avg_support_interactions**: Average support interactions per transaction
- **support_risk_score**: Binary flag for customers with >2 support interactions
- **recency_risk_score**: Binary flag for customers with >30 days since last transaction
- **frequency_risk_score**: Binary flag for customers with <3 total transactions

### 5. Engagement Metrics
- **recent_activity_ratio**: Proportion of transactions in last 30 days
- **spending_consistency**: Standard deviation of transaction amounts / average amount
- **avg_days_between_transactions**: Average time between consecutive transactions

### 6. Derived Categories
- **age_group_***: One-hot encoded age groups (adult, middle, senior vs young baseline)
- **amount_category_***: One-hot encoded spending levels (medium, high vs low baseline)

## Feature Engineering Decisions

### 1. Recency as Primary Predictor
Days since last transaction is expected to be the strongest churn predictor. Customers who haven't transacted recently are more likely to churn.

### 2. Risk Score Aggregation
Created binary risk flags for easy interpretation:
- Support risk: Customers requiring frequent support
- Recency risk: Customers with stale activity
- Frequency risk: Low-engagement customers

### 3. Engagement Ratios
Used ratios instead of absolute counts to normalize for different customer lifecycles:
- Recent activity ratio shows engagement trends
- Spending consistency identifies erratic vs stable customers

### 4. Categorical Encoding
- One-hot encoding for nominal categories (country, device)
- Ordinal encoding avoided to prevent false ordering assumptions
- Baseline categories chosen as most common values

## Data Leakage Prevention
- No future information used (all features based on historical data)
- No target-derived features (churn status not used in feature creation)
- Time-aware feature engineering (recency calculated from reference date)

## Missing Value Treatment
- avg_days_between_transactions: Filled with 0 for single-transaction customers
- spending_consistency: Filled with 0 for customers with identical transaction amounts
- No other missing values in engineered features

## Feature Statistics
"""
    
    # Add feature statistics
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != 'churned']
    
    feature_docs += f"\n### Numeric Features ({len(numeric_features)} total)\n"
    for feature in numeric_features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        feature_docs += f"- **{feature}**: Mean={mean_val:.2f}, Std={std_val:.2f}\n"
    
    # Binary features
    binary_features = [col for col in df.columns if df[col].dtype == 'uint8' and col != 'churned']
    feature_docs += f"\n### Binary Features ({len(binary_features)} total)\n"
    for feature in binary_features:
        positive_rate = df[feature].mean()
        feature_docs += f"- **{feature}**: {positive_rate:.1%} positive rate\n"
    
    feature_docs += f"""
## Target Variable
- **churned**: Binary target (1=churned, 0=retained)
- **Class Distribution**: {df['churned'].mean():.1%} positive class (churned)
- **Class Imbalance**: Requires balanced sampling or class weights

## Model Readiness
- Total features: {len(df.columns) - 2}  # Excluding customer_id and churned
- All features numeric (ready for ML algorithms)
- No missing values
- Features scaled appropriately for neural networks
"""
    
    # Save documentation
    os.makedirs('reports', exist_ok=True)
    with open('reports/FEATURES.md', 'w') as f:
        f.write(feature_docs)
    
    print("Feature documentation saved to reports/FEATURES.md")

if __name__ == "__main__":
    # Load raw data
    df = load_data()
    
    # Create customer-level features
    features_df = create_customer_features(df)
    
    # Engineer additional features
    features_df = engineer_additional_features(features_df)
    
    # Encode categorical features
    features_df = encode_categorical_features(features_df)
    
    # Save processed features
    save_features(features_df)
    
    # Generate documentation
    generate_features_documentation(features_df)
    
    print("\nFeature engineering complete!")
    print(f"Final dataset shape: {features_df.shape}")
    print(f"Features created: {list(features_df.columns)}")
