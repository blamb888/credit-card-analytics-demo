"""
Synthetic Credit Card Data Generator

This module generates realistic synthetic credit card transaction data
for analytics and modeling purposes. Includes customer demographics,
transaction patterns, and geographic information.

Author: Brandon (Data Scientist)
Purpose: Job Application Portfolio Project
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict
import random

from config import data_config, BUSINESS_CATEGORIES, GEOGRAPHIC_REGIONS, path_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreditCardDataGenerator:
    """
    Generates synthetic credit card transaction data with realistic patterns.
    
    Features:
    - Realistic customer demographics
    - Category-based spending patterns
    - Geographic distribution
    - Time-based transaction patterns
    - Weekend/weekday variations
    """
    
    def __init__(self, config=None):
        """Initialize the data generator with configuration."""
        self.config = config or data_config
        self.categories = BUSINESS_CATEGORIES
        self.regions = GEOGRAPHIC_REGIONS
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        
        logger.info(f"Initialized CreditCardDataGenerator with {self.config.n_customers:,} customers")
    
    def generate_customers(self) -> pd.DataFrame:
        """
        Generate customer demographic data.
        
        Returns:
            pd.DataFrame: Customer profiles with demographics and regional info
        """
        logger.info("Generating customer profiles...")
        
        customers = []
        
        for customer_id in range(1, self.config.n_customers + 1):
            # Assign region based on population distribution
            region = np.random.choice(
                list(self.regions.keys()), 
                p=[self.regions[r]['population_pct'] for r in self.regions.keys()]
            )
            
            # Generate age with realistic distribution
            age = np.random.normal(42, 15)
            age = max(18, min(80, age))  # Clamp between 18-80
            
            # Income correlated with age and region
            base_income = 35000 + (age - 18) * 1000 + np.random.normal(0, 15000)
            income = base_income * self.regions[region]['spending_multiplier']
            income = max(20000, income)  # Minimum income floor
            
            customers.append({
                'customer_id': customer_id,
                'age': int(age),
                'income': int(income),
                'region': region,
                'credit_limit': int(income * np.random.uniform(0.3, 0.8)),
                'signup_date': datetime(2022, 1, 1) + timedelta(
                    days=np.random.randint(0, 730)
                )
            })
        
        customer_df = pd.DataFrame(customers)
        logger.info(f"Generated {len(customer_df):,} customer profiles")
        
        return customer_df
    
    def generate_transactions(self, customers: pd.DataFrame) -> pd.DataFrame:
        """
        Generate transaction data based on customer profiles.
        
        Args:
            customers: Customer demographic data
            
        Returns:
            pd.DataFrame: Transaction records with realistic patterns
        """
        logger.info(f"Generating {self.config.n_transactions:,} transactions...")
        
        transactions = []
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        for i in range(self.config.n_transactions):
            if i % 50000 == 0:
                logger.info(f"  Generated {i:,} transactions...")
            
            # Select random customer
            customer = customers.sample(1).iloc[0]
            
            # Generate transaction date
            transaction_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            
            # Determine if weekend (affects spending patterns)
            is_weekend = transaction_date.weekday() >= 5
            
            # Select category based on frequency weights
            category = self._select_category()
            
            # Calculate transaction amount
            amount = self._calculate_amount(category, customer, is_weekend)
            
            # Generate merchant and location
            merchant_id = f"MERCH_{category[:3].upper()}_{np.random.randint(1000, 9999)}"
            latitude, longitude = self._generate_location(customer['region'])
            
            # Generate transaction hour with realistic distribution
            hours = list(range(6, 23))  # 6 AM to 10 PM (17 hours)
            hour_probs = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]
            # Normalize probabilities to ensure they sum to 1
            hour_probs = np.array(hour_probs) / np.sum(hour_probs)
            hour = np.random.choice(hours, p=hour_probs)
            
            transactions.append({
                'transaction_id': f"TXN_{i+1:07d}",
                'customer_id': customer['customer_id'],
                'transaction_date': transaction_date,
                'amount': round(amount, 2),
                'category': category,
                'merchant_id': merchant_id,
                'latitude': round(latitude, 6),
                'longitude': round(longitude, 6),
                'is_weekend': is_weekend,
                'hour': hour
            })
        
        transaction_df = pd.DataFrame(transactions)
        logger.info(f"Generated {len(transaction_df):,} transactions")
        logger.info(f"Total transaction value: ${transaction_df['amount'].sum():,.2f}")
        
        return transaction_df
    
    def _select_category(self) -> str:
        """Select transaction category based on frequency weights."""
        category_weights = [self.categories[cat]['frequency'] for cat in self.categories.keys()]
        return np.random.choice(
            list(self.categories.keys()), 
            p=np.array(category_weights) / sum(category_weights)
        )
    
    def _calculate_amount(self, category: str, customer: Dict, is_weekend: bool) -> float:
        """Calculate transaction amount based on category, customer, and timing."""
        base_amount = self.categories[category]['avg_spend']
        weekend_multiplier = self.categories[category]['weekend_boost'] if is_weekend else 1.0
        income_multiplier = (customer['income'] / 50000) ** 0.3  # Diminishing returns
        
        amount = base_amount * weekend_multiplier * income_multiplier * np.random.lognormal(0, 0.4)
        
        # Apply reasonable bounds
        return max(5, min(amount, customer['credit_limit'] * 0.2))
    
    def _generate_location(self, region: str) -> Tuple[float, float]:
        """Generate latitude/longitude based on customer region."""
        lat_base, lon_base = self.regions[region]['coords']
        
        # Add some randomness to location
        latitude = lat_base + np.random.normal(0, 0.1)
        longitude = lon_base + np.random.normal(0, 0.1)
        
        return latitude, longitude
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with customers and transactions.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (customers, transactions)
        """
        logger.info("Starting complete dataset generation...")
        
        # Generate customers
        customers = self.generate_customers()
        
        # Generate transactions
        transactions = self.generate_transactions(customers)
        
        # Save to files
        customers_path = f"{path_config.raw_data_dir}/synthetic_customers.csv"
        transactions_path = f"{path_config.raw_data_dir}/synthetic_transactions.csv"
        
        customers.to_csv(customers_path, index=False)
        transactions.to_csv(transactions_path, index=False)
        
        logger.info(f"Saved customers to: {customers_path}")
        logger.info(f"Saved transactions to: {transactions_path}")
        
        # Log summary statistics
        self._log_summary_stats(customers, transactions)
        
        return customers, transactions
    
    def _log_summary_stats(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> None:
        """Log summary statistics about the generated dataset."""
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Customers: {len(customers):,}")
        logger.info(f"Transactions: {len(transactions):,}")
        logger.info(f"Date Range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
        logger.info(f"Categories: {transactions['category'].nunique()}")
        logger.info(f"Total Volume: ${transactions['amount'].sum():,.2f}")
        logger.info(f"Avg Transaction: ${transactions['amount'].mean():.2f}")
        logger.info(f"Unique Merchants: {transactions['merchant_id'].nunique():,}")

def main():
    """Main function to generate synthetic dataset."""
    generator = CreditCardDataGenerator()
    customers, transactions = generator.generate_dataset()
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Files saved to {path_config.raw_data_dir}/")
    print(f"📊 {len(customers):,} customers, {len(transactions):,} transactions")

if __name__ == "__main__":
    main()