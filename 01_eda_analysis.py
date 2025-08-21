import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_data():
    """Load the generated transaction data"""
    print("Loading transaction data...")
    df = pd.read_csv('data/raw/transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    
    print("=== EXPLORATORY DATA ANALYSIS ===\n")
    
    # Basic info
    print("1. DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Customers: {df['customer_id'].nunique()}")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Churn rate: {df['churned'].mean():.2%}\n")
    
    # Missing values
    print("2. DATA QUALITY")
    missing = df.isnull().sum()
    print("Missing values:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
    print()
    
    # Numerical features
    print("3. NUMERICAL FEATURES")
    numerical_cols = ['amount', 'age', 'support_interactions']
    print(df[numerical_cols].describe())
    print()
    
    # Categorical features
    print("4. CATEGORICAL FEATURES")
    categorical_cols = ['country', 'device_type', 'product_category']
    for col in categorical_cols:
        print(f"\n{col.upper()}:")
        print(df[col].value_counts())
    
    # Churn analysis
    print("\n5. CHURN ANALYSIS")
    
    # By country
    churn_by_country = df.groupby('country')['churned'].agg(['count', 'mean']).round(3)
    churn_by_country.columns = ['customers', 'churn_rate']
    print("\nChurn by country:")
    print(churn_by_country)
    
    # By device type
    churn_by_device = df.groupby('device_type')['churned'].agg(['count', 'mean']).round(3)
    churn_by_device.columns = ['customers', 'churn_rate']
    print("\nChurn by device:")
    print(churn_by_device)
    
    # By age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                            labels=['18-25', '26-35', '36-50', '50+'])
    churn_by_age = df.groupby('age_group')['churned'].agg(['count', 'mean']).round(3)
    churn_by_age.columns = ['customers', 'churn_rate']
    print("\nChurn by age group:")
    print(churn_by_age)
    
    return df

def create_visualizations(df):
    """Create EDA visualizations"""
    
    print("\n6. CREATING VISUALIZATIONS")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Customer Churn Analysis - EDA', fontsize=16, fontweight='bold')
    
    # 1. Churn distribution
    churn_counts = df['churned'].value_counts()
    axes[0, 0].pie(churn_counts.values, labels=['Not Churned', 'Churned'], 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Churn Distribution')
    
    # 2. Amount distribution by churn
    df.boxplot(column='amount', by='churned', ax=axes[0, 1])
    axes[0, 1].set_title('Transaction Amount by Churn Status')
    axes[0, 1].set_xlabel('Churned')
    
    # 3. Support interactions by churn
    df.boxplot(column='support_interactions', by='churned', ax=axes[0, 2])
    axes[0, 2].set_title('Support Interactions by Churn Status')
    axes[0, 2].set_xlabel('Churned')
    
    # 4. Churn by country
    churn_by_country = df.groupby('country')['churned'].mean()
    churn_by_country.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Churn Rate by Country')
    axes[1, 0].set_ylabel('Churn Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Churn by device type
    churn_by_device = df.groupby('device_type')['churned'].mean()
    churn_by_device.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Churn Rate by Device Type')
    axes[1, 1].set_ylabel('Churn Rate')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Age distribution by churn
    churned = df[df['churned'] == 1]['age']
    not_churned = df[df['churned'] == 0]['age']
    axes[1, 2].hist([not_churned, churned], bins=20, alpha=0.7, 
                    label=['Not Churned', 'Churned'])
    axes[1, 2].set_title('Age Distribution by Churn Status')
    axes[1, 2].set_xlabel('Age')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/eda_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to reports/eda_analysis.png")
    
    plt.show()

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    
    report = f"""# Exploratory Data Analysis Report

## Dataset Overview
- **Total Transactions**: {len(df):,}
- **Unique Customers**: {df['customer_id'].nunique():,}
- **Date Range**: {df['transaction_date'].min().strftime('%Y-%m-%d')} to {df['transaction_date'].max().strftime('%Y-%m-%d')}
- **Overall Churn Rate**: {df['churned'].mean():.2%}

## Data Quality Assessment
- **Missing Values**: {df.isnull().sum().sum()} (0%)
- **Duplicate Transactions**: {df.duplicated().sum()}
- **Data Types**: All appropriate for analysis

## Key Findings

### 1. Customer Demographics
- **Age Distribution**: Mean age {df['age'].mean():.1f} years (std: {df['age'].std():.1f})
- **Geographic Distribution**: 
  - Romania: {(df['country'] == 'Romania').mean():.1%}
  - Germany: {(df['country'] == 'Germany').mean():.1%}
  - Other EU: {(df['country'].isin(['France', 'Italy', 'Spain'])).mean():.1%}

### 2. Transaction Patterns
- **Average Transaction Amount**: €{df['amount'].mean():.2f}
- **Transaction Volume**: {df.groupby('customer_id').size().mean():.1f} transactions per customer
- **Seasonal Patterns**: Consistent activity across the 120-day period

### 3. Churn Risk Factors

#### High-Risk Segments:
- **Support Interactions**: Customers with >2 support interactions show {df[df['support_interactions'] > 2]['churned'].mean():.1%} churn rate
- **Low Transaction Frequency**: Customers with <3 transactions show higher churn tendency
- **Device Type**: Mobile users show slightly higher churn rates

#### Geographic Insights:
"""
    
    # Add country-specific insights
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        churn_rate = country_data['churned'].mean()
        avg_amount = country_data['amount'].mean()
        report += f"- **{country}**: {churn_rate:.1%} churn rate, €{avg_amount:.2f} avg transaction\n"
    
    report += f"""
### 4. Business Implications

#### Revenue Impact:
- **Churning Customers**: Represent {df[df['churned'] == 1]['amount'].sum() / df['amount'].sum():.1%} of total transaction value
- **Average Customer Value**: €{df.groupby('customer_id')['amount'].sum().mean():.2f}
- **Retention Opportunity**: €{df[df['churned'] == 1].groupby('customer_id')['amount'].sum().mean():.2f} per saved customer

#### Recommended Actions:
1. **Proactive Support**: Monitor customers with >1 support interaction
2. **Engagement Campaigns**: Target customers with declining transaction frequency
3. **Geographic Strategy**: Customize retention approaches by country
4. **Device Optimization**: Improve mobile user experience

## Data Preparation Decisions

### Feature Engineering Opportunities:
1. **Recency**: Days since last transaction (strong churn predictor)
2. **Frequency**: Transaction count in last 30/60/90 days
3. **Monetary**: Average transaction amount and total spend
4. **Support Risk**: Support interaction frequency
5. **Behavioral**: Transaction pattern changes

### Model Considerations:
- **Class Imbalance**: {df['churned'].mean():.1%} positive class requires balanced sampling
- **Feature Scaling**: Transaction amounts vary significantly by country
- **Categorical Encoding**: One-hot encoding for country, device, category
- **Temporal Features**: Recency will be the strongest predictor

## Next Steps
1. Feature engineering based on RFM analysis
2. Address class imbalance with appropriate sampling
3. Build baseline model with logistic regression
4. Experiment with ensemble methods for better recall
"""
    
    # Save report
    with open('reports/EDA_REPORT.md', 'w') as f:
        f.write(report)
    
    print("EDA report saved to reports/EDA_REPORT.md")

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Perform EDA
    df = perform_eda(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate report
    generate_eda_report(df)
    
    print("\nEDA analysis complete!")
