"""
Tableau-Style Business Intelligence Visualizations
Creates professional BI-style dashboards using the generated data

Author: Brandon (Data Scientist)  
Purpose: Job Application Portfolio - BI Dashboard Demo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class TableauStyleDashboard:
    """Creates professional BI-style dashboards mimicking Tableau aesthetics."""
    
    def __init__(self):
        """Initialize dashboard creator."""
        print("🎨 Creating Tableau-Style Business Intelligence Dashboards")
        print("=" * 60)
        
    def load_processed_data(self):
        """Load the processed data files."""
        try:
            # Load customer segments
            self.segments_df = pd.read_csv('data/processed/customer_segments.csv')
            
            # Load regional analysis  
            self.regional_df = pd.read_csv('data/processed/regional_analysis.csv')
            
            # Load original transaction data for trends
            self.transactions_df = pd.read_csv('data/raw/synthetic_transactions.csv')
            self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
            
            print(f"✅ Loaded {len(self.segments_df):,} customer segments")
            print(f"✅ Loaded {len(self.regional_df)} regional analyses") 
            print(f"✅ Loaded {len(self.transactions_df):,} transactions")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Data files not found: {e}")
            print("   Please run 'python main.py' first to generate the data")
            return False
    
    def create_executive_dashboard(self):
        """Create executive-level dashboard with key KPIs."""
        print("\n📊 Creating Executive Dashboard...")
        
        # Create subplot layout (2x2)
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Credit Card Analytics - Executive Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Revenue by Customer Segment (Top Left)
        plt.subplot(2, 2, 1)
        segment_revenue = self.segments_df.groupby('segment')['monetary_value'].sum().sort_values(ascending=False)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        bars = plt.bar(range(len(segment_revenue)), segment_revenue.values, color=colors[:len(segment_revenue)])
        plt.title('Revenue by Customer Segment', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Total Revenue ($M)')
        plt.xticks(range(len(segment_revenue)), segment_revenue.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, segment_revenue.values)):
            plt.text(i, value + max(segment_revenue.values)*0.01, f'${value/1000000:.1f}M', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Regional Performance (Top Right)
        plt.subplot(2, 2, 2)
        regional_spending = self.regional_df['Spending_Per_Customer'].sort_values(ascending=True)
        bars = plt.barh(range(len(regional_spending)), regional_spending.values, color='#2ca02c')
        plt.title('Average Spending per Customer by Region', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Average Spending per Customer ($)')
        plt.yticks(range(len(regional_spending)), regional_spending.index)
        
        # Add value labels
        for i, value in enumerate(regional_spending.values):
            plt.text(value + max(regional_spending.values)*0.01, i, f'${value:,.0f}', 
                    va='center', fontweight='bold')
        
        # 3. Monthly Revenue Trends (Bottom Left)
        plt.subplot(2, 2, 3)
        monthly_data = self.transactions_df.copy()
        monthly_data['month'] = monthly_data['transaction_date'].dt.to_period('M')
        monthly_revenue = monthly_data.groupby('month')['amount'].sum()
        
        plt.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=3, markersize=8, color='#1f77b4')
        plt.title('Monthly Revenue Trends', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Monthly Revenue ($M)')
        plt.xlabel('Month')
        plt.xticks(range(0, len(monthly_revenue), 3), 
                   [str(monthly_revenue.index[i]) for i in range(0, len(monthly_revenue), 3)], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to millions
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))
        
        # 4. Customer Segment Distribution (Bottom Right) 
        plt.subplot(2, 2, 4)
        segment_counts = self.segments_df['segment'].value_counts()
        wedges, texts, autotexts = plt.pie(segment_counts.values, labels=segment_counts.index, 
                                          autopct='%1.1f%%', colors=colors[:len(segment_counts)])
        plt.title('Customer Segment Distribution', fontsize=14, fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save dashboard
        plt.savefig('outputs/visualizations/executive_dashboard.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("✅ Executive dashboard saved: outputs/visualizations/executive_dashboard.png")
        
        return fig
    
    def create_customer_analytics_dashboard(self):
        """Create customer analytics dashboard with RFM analysis."""
        print("\n🔍 Creating Customer Analytics Dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Customer Analytics - RFM & Segmentation Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. RFM Scatter Plot (Top Left)
        plt.subplot(2, 2, 1)
        segments = self.segments_df['segment'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, segment in enumerate(segments):
            segment_data = self.segments_df[self.segments_df['segment'] == segment]
            plt.scatter(segment_data['recency'], segment_data['monetary_value'], 
                       s=segment_data['frequency']*2, alpha=0.6, 
                       color=colors[i % len(colors)], label=segment)
        
        plt.xlabel('Recency (Days Since Last Purchase)')
        plt.ylabel('Monetary Value ($)')
        plt.title('RFM Analysis: Customer Segments', fontsize=14, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Average Transaction Value by Segment (Top Right)
        plt.subplot(2, 2, 2)
        avg_transaction = self.segments_df.groupby('segment')['avg_transaction_value'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(avg_transaction)), avg_transaction.values, color=colors[:len(avg_transaction)])
        plt.title('Average Transaction Value by Segment', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Average Transaction Value ($)')
        plt.xticks(range(len(avg_transaction)), avg_transaction.index, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, avg_transaction.values)):
            plt.text(i, value + max(avg_transaction.values)*0.01, f'${value:.0f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Customer Frequency Distribution (Bottom Left)
        plt.subplot(2, 2, 3)
        frequency_ranges = pd.cut(self.segments_df['frequency'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        freq_counts = frequency_ranges.value_counts()
        plt.bar(range(len(freq_counts)), freq_counts.values, color='#2ca02c')
        plt.title('Customer Purchase Frequency Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Number of Customers')
        plt.xlabel('Purchase Frequency Category')
        plt.xticks(range(len(freq_counts)), freq_counts.index)
        
        # 4. Revenue Distribution by Segment (Bottom Right)
        plt.subplot(2, 2, 4)
        segment_revenue_pct = self.segments_df.groupby('segment')['monetary_value'].sum()
        segment_revenue_pct = segment_revenue_pct / segment_revenue_pct.sum() * 100
        
        bars = plt.barh(range(len(segment_revenue_pct)), segment_revenue_pct.values, color=colors[:len(segment_revenue_pct)])
        plt.title('Revenue Distribution by Segment (%)', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Percentage of Total Revenue')
        plt.yticks(range(len(segment_revenue_pct)), segment_revenue_pct.index)
        
        # Add percentage labels
        for i, value in enumerate(segment_revenue_pct.values):
            plt.text(value + max(segment_revenue_pct.values)*0.01, i, f'{value:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save dashboard
        plt.savefig('outputs/visualizations/customer_analytics_dashboard.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("✅ Customer analytics dashboard saved: outputs/visualizations/customer_analytics_dashboard.png")
        
        return fig
    
    def create_interactive_plotly_dashboard(self):
        """Create interactive Plotly dashboard for web sharing."""
        print("\n🌐 Creating Interactive Plotly Dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue by Segment', 'Regional Performance', 
                           'Monthly Trends', 'RFM Analysis'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Revenue by Segment
        segment_revenue = self.segments_df.groupby('segment')['monetary_value'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=segment_revenue.index, y=segment_revenue.values,
                   name='Revenue by Segment', marker_color='#1f77b4',
                   text=[f'${v/1000000:.1f}M' for v in segment_revenue.values],
                   textposition='auto'),
            row=1, col=1
        )
        
        # 2. Regional Performance
        fig.add_trace(
            go.Bar(x=self.regional_df['Spending_Per_Customer'], y=self.regional_df.index,
                   orientation='h', name='Regional Performance', marker_color='#2ca02c',
                   text=[f'${v:,.0f}' for v in self.regional_df['Spending_Per_Customer']],
                   textposition='auto'),
            row=1, col=2
        )
        
        # 3. Monthly Trends
        monthly_data = self.transactions_df.copy()
        monthly_data['month'] = monthly_data['transaction_date'].dt.to_period('M')
        monthly_revenue = monthly_data.groupby('month')['amount'].sum()
        
        fig.add_trace(
            go.Scatter(x=[str(m) for m in monthly_revenue.index], y=monthly_revenue.values,
                      mode='lines+markers', name='Monthly Revenue', line=dict(color='#ff7f0e', width=3)),
            row=2, col=1
        )
        
        # 4. RFM Scatter
        fig.add_trace(
            go.Scatter(x=self.segments_df['recency'], y=self.segments_df['monetary_value'],
                      mode='markers', name='Customer RFM',
                      marker=dict(size=self.segments_df['frequency']/5, 
                                color=self.segments_df['segment'].astype('category').cat.codes,
                                colorscale='viridis', opacity=0.7),
                      text=self.segments_df['segment'],
                      hovertemplate='<b>%{text}</b><br>Recency: %{x} days<br>Value: $%{y}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Credit Card Analytics - Interactive Business Intelligence Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False
        )
        
        # Save interactive dashboard
        fig.write_html("outputs/dashboards/interactive_bi_dashboard.html")
        print("✅ Interactive dashboard saved: outputs/dashboards/interactive_bi_dashboard.html")
        
        return fig
    
    def generate_tableau_data_extracts(self):
        """Generate Tableau-optimized data extracts."""
        print("\n📋 Creating Tableau-Ready Data Extracts...")
        
        # Create Tableau-optimized customer summary
        tableau_customers = self.segments_df[['customer_id', 'segment', 'monetary_value', 'frequency', 
                                             'recency', 'avg_transaction_value', 'age', 'income', 'region']].copy()
        
        # Add calculated fields for Tableau
        tableau_customers['customer_lifetime_value'] = tableau_customers['monetary_value']
        tableau_customers['purchase_frequency_category'] = pd.cut(tableau_customers['frequency'], 
                                                                 bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        tableau_customers['value_tier'] = pd.cut(tableau_customers['monetary_value'], 
                                                bins=5, labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond'])
        
        # Save Tableau extract
        tableau_customers.to_csv('outputs/tableau_data/customer_analysis_extract.csv', index=False)
        
        # Create transaction summary for Tableau
        transaction_summary = self.transactions_df.groupby(['customer_id', 'category', 'transaction_date']).agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        transaction_summary.columns = ['customer_id', 'category', 'transaction_date', 'daily_spending', 'daily_transactions']
        
        # Add time dimensions
        transaction_summary['year'] = pd.to_datetime(transaction_summary['transaction_date']).dt.year
        transaction_summary['month'] = pd.to_datetime(transaction_summary['transaction_date']).dt.month
        transaction_summary['quarter'] = pd.to_datetime(transaction_summary['transaction_date']).dt.quarter
        transaction_summary['day_of_week'] = pd.to_datetime(transaction_summary['transaction_date']).dt.day_name()
        
        transaction_summary.to_csv('outputs/tableau_data/transaction_analysis_extract.csv', index=False)
        
        print("✅ Tableau extracts saved:")
        print("   • outputs/tableau_data/customer_analysis_extract.csv")
        print("   • outputs/tableau_data/transaction_analysis_extract.csv")
        
        print("\n🎯 Ready for Tableau Public Import!")
        print("   1. Open Tableau Public Desktop")
        print("   2. Connect to Text File -> Select the CSV extracts")
        print("   3. Create visualizations using the pre-calculated fields")
        print("   4. Publish to Tableau Public for sharing")

def main():
    """Create comprehensive BI dashboards."""
    # Create output directories
    import os
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/dashboards', exist_ok=True)
    os.makedirs('outputs/tableau_data', exist_ok=True)
    
    # Initialize dashboard creator
    dashboard = TableauStyleDashboard()
    
    # Load data
    if not dashboard.load_processed_data():
        return
    
    # Create dashboards
    dashboard.create_executive_dashboard()
    dashboard.create_customer_analytics_dashboard()
    dashboard.create_interactive_plotly_dashboard()
    dashboard.generate_tableau_data_extracts()
    
    print(f"\n🎉 BI Dashboard Creation Complete!")
    print(f"📊 Professional dashboards created in Tableau style")
    print(f"🌐 Interactive web dashboard ready for sharing")
    print(f"📋 Tableau Public data extracts ready for import")
    print(f"📁 All outputs available in: outputs/")

if __name__ == "__main__":
    main()