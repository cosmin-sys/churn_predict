import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Check if required packages are available
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install pandas numpy")
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_customers=5000):
    """Generate realistic customer transaction data"""
    
    print("Generating sample customer data...")
    
    try:
        # Customer demographics
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        countries = np.random.choice(['Romania', 'Germany', 'France', 'Italy', 'Spain'], 
                                    n_customers, p=[0.4, 0.2, 0.15, 0.15, 0.1])
        ages = np.random.normal(35, 12, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)
        
        device_types = np.random.choice(['mobile', 'desktop', 'tablet'], 
                                       n_customers, p=[0.6, 0.3, 0.1])
        
        # Generate transaction patterns
        data = []
        base_date = datetime(2024, 1, 1)
        
        for i, customer_id in enumerate(customer_ids):
            # Customer characteristics
            country = countries[i]
            age = ages[i]
            device_type = device_types[i]
            
            # Determine if customer will churn (22% churn rate)
            will_churn = np.random.random() < 0.22
            
            # Generate transaction history (last 120 days)
            n_transactions = np.random.poisson(8 if not will_churn else 4)
            n_transactions = max(1, n_transactions)  # At least 1 transaction
            
            for j in range(n_transactions):
                # Transaction timing
                days_ago = np.random.randint(1, 121)
                transaction_date = base_date - timedelta(days=days_ago)
                
                # If churning, make recent transactions less likely
                if will_churn and days_ago < 30:
                    if np.random.random() < 0.7:  # 70% chance to skip recent transactions
                        continue
                
                # Transaction amount (influenced by country and age)
                base_amount = 50
                if country == 'Germany':
                    base_amount *= 1.3
                elif country == 'Romania':
                    base_amount *= 0.8
                
                if age > 50:
                    base_amount *= 1.2
                
                amount = np.random.lognormal(np.log(base_amount), 0.5)
                amount = round(amount, 2)
                
                # Support interactions (churning customers have more)
                support_interactions = np.random.poisson(2 if will_churn else 0.5)
                
                # Product categories
                categories = ['electronics', 'clothing', 'books', 'home', 'sports']
                category = np.random.choice(categories)
                
                data.append({
                    'customer_id': customer_id,
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'country': country,
                    'age': age,
                    'device_type': device_type,
                    'product_category': category,
                    'support_interactions': support_interactions,
                    'churned': 1 if will_churn else 0
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Save to CSV
        df.to_csv('data/raw/transactions.csv', index=False)
        
        print(f"✅ Generated {len(df)} transactions for {n_customers} customers")
        print(f"✅ Churn rate: {df['churned'].mean():.2%}")
        print("✅ Data saved to data/raw/transactions.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return None

if __name__ == "__main__":
    print("=== GENERATING SAMPLE DATA ===\n")
    
    df = generate_sample_data()
    
    if df is not None:
        print("\n✅ Sample data generation complete!")
        print("\nSample data preview:")
        print(df.head())
    else:
        print("\n❌ Data generation failed!")
