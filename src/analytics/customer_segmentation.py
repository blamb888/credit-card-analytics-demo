"""
Customer Segmentation Analytics Module

Implements RFM (Recency, Frequency, Monetary) analysis and machine learning
clustering to identify distinct customer segments for targeted marketing
and business strategy.

Author: Brandon (Data Scientist)
Purpose: Job Application Portfolio Project
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import model_config, viz_config, path_config

logger = logging.getLogger(__name__)

class CustomerSegmentationAnalyzer:
    """
    Advanced customer segmentation using RFM analysis and K-means clustering.
    
    Features:
    - RFM (Recency, Frequency, Monetary) metrics calculation
    - Category spending behavior analysis
    - K-means clustering with optimal cluster selection
    - Business-friendly segment labeling
    - Segment performance analysis
    """
    
    def __init__(self, config=None):
        """Initialize the segmentation analyzer."""
        self.config = config or model_config
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.segment_labels = {}
        
        logger.info("Initialized CustomerSegmentationAnalyzer")
    
    def calculate_rfm_metrics(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        
        Args:
            transactions: Transaction data
            customers: Customer demographic data
            
        Returns:
            pd.DataFrame: RFM metrics with customer demographics
        """
        logger.info("Calculating RFM metrics...")
        
        # Get current date as reference point
        current_date = transactions['transaction_date'].max()
        
        # Calculate RFM metrics
        rfm_data = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm_data.columns = ['customer_id', 'recency', 'frequency', 'monetary_value']
        
        # Add customer demographics
        rfm_data = rfm_data.merge(
            customers[['customer_id', 'age', 'income', 'region']], 
            on='customer_id'
        )
        
        # Calculate additional business metrics
        rfm_data['avg_transaction_value'] = rfm_data['monetary_value'] / rfm_data['frequency']
        rfm_data['transaction_frequency_per_month'] = rfm_data['frequency'] / 24  # 24 months of data
        
        logger.info(f"Calculated RFM metrics for {len(rfm_data):,} customers")
        return rfm_data
    
    def add_category_features(self, rfm_data: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Add category spending behavior features to RFM data.
        
        Args:
            rfm_data: RFM metrics data
            transactions: Transaction data
            
        Returns:
            pd.DataFrame: Enhanced RFM data with category features
        """
        logger.info("Adding category spending features...")
        
        # Category spending amounts
        category_spending = transactions.groupby(['customer_id', 'category'])['amount'].sum().unstack(fill_value=0)
        category_spending.columns = [f'{col}_spending' for col in category_spending.columns]
        
        # Category spending percentages
        total_spending = category_spending.sum(axis=1)
        category_pcts = category_spending.div(total_spending, axis=0) * 100
        category_pcts.columns = [f'{col.replace("_spending", "")}_pct' for col in category_pcts.columns]
        
        # Merge with RFM data
        rfm_enhanced = rfm_data.merge(category_spending, left_on='customer_id', right_index=True, how='left')
        rfm_enhanced = rfm_enhanced.merge(category_pcts, left_on='customer_id', right_index=True, how='left')
        
        # Fill any missing values
        rfm_enhanced = rfm_enhanced.fillna(0)
        
        logger.info(f"Added {len([col for col in rfm_enhanced.columns if '_spending' in col])} category features")
        return rfm_enhanced
    
    def find_optimal_clusters(self, X: np.ndarray, k_range: range = None) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X: Scaled feature matrix
            k_range: Range of k values to test
            
        Returns:
            int: Optimal number of clusters
        """
        if k_range is None:
            k_range = range(2, 11)
        
        logger.info(f"Finding optimal clusters in range {k_range}")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # Use configured number of clusters (5 for business interpretation)
        optimal_k = self.config.n_clusters
        logger.info(f"Using {optimal_k} clusters for business segmentation")
        
        return optimal_k
    
    def perform_clustering(self, rfm_enhanced: pd.DataFrame) -> pd.DataFrame:
        """
        Perform K-means clustering on customer features.
        
        Args:
            rfm_enhanced: RFM data with category features
            
        Returns:
            pd.DataFrame: RFM data with cluster assignments
        """
        logger.info("Performing customer clustering...")
        
        # Prepare features for clustering
        clustering_features = self.config.clustering_features.copy()
        
        # Add category spending features
        spending_cols = [col for col in rfm_enhanced.columns if col.endswith('_spending')]
        clustering_features.extend(spending_cols)
        
        # Ensure all features exist
        available_features = [feat for feat in clustering_features if feat in rfm_enhanced.columns]
        X = rfm_enhanced[available_features].fillna(0)
        
        logger.info(f"Using {len(available_features)} features for clustering")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters
        optimal_k = self.find_optimal_clusters(X_scaled)
        
        # Perform final clustering
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels to data
        rfm_clustered = rfm_enhanced.copy()
        rfm_clustered['cluster'] = cluster_labels
        
        logger.info(f"Clustering complete with {optimal_k} segments")
        return rfm_clustered
    
    def label_segments(self, rfm_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Create business-friendly labels for customer segments.
        
        Args:
            rfm_clustered: RFM data with cluster assignments
            
        Returns:
            pd.DataFrame: Data with business segment labels
        """
        logger.info("Creating business-friendly segment labels...")
        
        self.segment_labels = {}
        
        for cluster_id in range(self.config.n_clusters):
            cluster_data = rfm_clustered[rfm_clustered['cluster'] == cluster_id]
            
            avg_monetary = cluster_data['monetary_value'].mean()
            avg_frequency = cluster_data['frequency'].mean()
            avg_recency = cluster_data['recency'].mean()
            
            # Business logic for segment labeling
            if avg_monetary > rfm_clustered['monetary_value'].quantile(0.8):
                if avg_frequency > rfm_clustered['frequency'].quantile(0.8):
                    label = "VIP_Champions"
                else:
                    label = "High_Value_Occasional"
            elif avg_monetary > rfm_clustered['monetary_value'].quantile(0.6):
                if avg_frequency > rfm_clustered['frequency'].quantile(0.6):
                    label = "Loyal_Customers"
                else:
                    label = "Potential_Loyalists"
            elif avg_recency > rfm_clustered['recency'].quantile(0.8):
                label = "At_Risk_Customers"
            else:
                label = "New_Customers"
            
            self.segment_labels[cluster_id] = label
        
        # Apply labels
        rfm_clustered['segment'] = rfm_clustered['cluster'].map(self.segment_labels)
        
        # Log segment distribution
        for cluster_id, label in self.segment_labels.items():
            count = (rfm_clustered['cluster'] == cluster_id).sum()
            pct = count / len(rfm_clustered) * 100
            logger.info(f"  Cluster {cluster_id}: {label} ({count:,} customers, {pct:.1f}%)")
        
        return rfm_clustered
    
    def analyze_segments(self, rfm_segmented: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze segment characteristics and performance.
        
        Args:
            rfm_segmented: Segmented customer data
            
        Returns:
            pd.DataFrame: Segment analysis summary
        """
        logger.info("Analyzing segment characteristics...")
        
        # Create segment summary
        segment_summary = rfm_segmented.groupby('segment').agg({
            'customer_id': 'count',
            'monetary_value': ['mean', 'median', 'std'],
            'frequency': ['mean', 'median'],
            'recency': ['mean', 'median'],
            'avg_transaction_value': ['mean', 'median'],
            'age': 'mean',
            'income': 'mean'
        }).round(2)
        
        # Flatten column names
        segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns]
        segment_summary = segment_summary.rename(columns={'customer_id_count': 'customer_count'})
        
        # Calculate segment value and percentage
        segment_summary['total_value'] = (
            segment_summary['customer_count'] * segment_summary['monetary_value_mean']
        ).round(2)
        
        segment_summary['percentage_of_customers'] = (
            segment_summary['customer_count'] / segment_summary['customer_count'].sum() * 100
        ).round(1)
        
        # Sort by total value descending
        segment_summary = segment_summary.sort_values('total_value', ascending=False)
        
        logger.info("Segment analysis complete")
        return segment_summary
    
    def run_complete_analysis(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete customer segmentation analysis.
        
        Args:
            transactions: Transaction data
            customers: Customer data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (segmented_customers, segment_summary)
        """
        logger.info("Starting complete customer segmentation analysis...")
        
        # Calculate RFM metrics
        rfm_data = self.calculate_rfm_metrics(transactions, customers)
        
        # Add category features
        rfm_enhanced = self.add_category_features(rfm_data, transactions)
        
        # Perform clustering
        rfm_clustered = self.perform_clustering(rfm_enhanced)
        
        # Label segments
        rfm_segmented = self.label_segments(rfm_clustered)
        
        # Analyze segments
        segment_summary = self.analyze_segments(rfm_segmented)
        
        # Save results
        output_path = f"{path_config.processed_data_dir}/customer_segments.csv"
        rfm_segmented.to_csv(output_path, index=False)
        
        summary_path = f"{path_config.processed_data_dir}/segment_summary.csv"
        segment_summary.to_csv(summary_path)
        
        logger.info(f"Segmentation results saved to {output_path}")
        logger.info(f"Segment summary saved to {summary_path}")
        
        return rfm_segmented, segment_summary
    
    def get_segment_insights(self, rfm_segmented: pd.DataFrame) -> Dict:
        """
        Generate actionable business insights from segmentation.
        
        Args:
            rfm_segmented: Segmented customer data
            
        Returns:
            Dict: Business insights and recommendations
        """
        insights = {}
        
        # Overall metrics
        total_customers = len(rfm_segmented)
        total_revenue = rfm_segmented['monetary_value'].sum()
        
        # Top performing segment
        segment_revenue = rfm_segmented.groupby('segment')['monetary_value'].sum().sort_values(ascending=False)
        top_segment = segment_revenue.index[0]
        top_segment_revenue = segment_revenue.iloc[0]
        top_segment_customers = len(rfm_segmented[rfm_segmented['segment'] == top_segment])
        
        # At-risk analysis
        at_risk_customers = rfm_segmented[rfm_segmented['segment'] == 'At_Risk_Customers']
        at_risk_count = len(at_risk_customers)
        at_risk_value = at_risk_customers['monetary_value'].sum()
        
        # Recovery potential (assuming 30% recovery rate)
        recovery_potential = at_risk_value * 0.3
        
        insights = {
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'avg_customer_value': total_revenue / total_customers,
            'top_segment': {
                'name': top_segment,
                'revenue': top_segment_revenue,
                'customers': top_segment_customers,
                'avg_value': top_segment_revenue / top_segment_customers
            },
            'at_risk': {
                'customers': at_risk_count,
                'revenue_at_risk': at_risk_value,
                'recovery_potential': recovery_potential
            },
            'segments': {
                segment: {
                    'customers': len(rfm_segmented[rfm_segmented['segment'] == segment]),
                    'revenue': rfm_segmented[rfm_segmented['segment'] == segment]['monetary_value'].sum(),
                    'avg_value': rfm_segmented[rfm_segmented['segment'] == segment]['monetary_value'].mean()
                }
                for segment in rfm_segmented['segment'].unique()
            }
        }
        
        return insights

def main():
    """Main function to run customer segmentation analysis."""
    # Load data
    customers = pd.read_csv(f"{path_config.raw_data_dir}/synthetic_customers.csv")
    transactions = pd.read_csv(f"{path_config.raw_data_dir}/synthetic_transactions.csv")
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    # Run analysis
    analyzer = CustomerSegmentationAnalyzer()
    segmented_customers, segment_summary = analyzer.run_complete_analysis(transactions, customers)
    
    # Generate insights
    insights = analyzer.get_segment_insights(segmented_customers)
    
    # Print summary
    print(f"\n✅ Customer Segmentation Analysis Complete!")
    print(f"📊 {insights['total_customers']:,} customers segmented into {len(insights['segments'])} groups")
    print(f"💰 Total revenue analyzed: ${insights['total_revenue']:,.2f}")
    print(f"🎯 Top segment: {insights['top_segment']['name']} ({insights['top_segment']['customers']:,} customers)")
    print(f"⚠️  At-risk customers: {insights['at_risk']['customers']:,}")
    print(f"💡 Recovery potential: ${insights['at_risk']['recovery_potential']:,.2f}")

if __name__ == "__main__":
    main()