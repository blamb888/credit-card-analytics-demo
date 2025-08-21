"""
Interactive Credit Card Analytics Dashboard
Professional web-based BI dashboard using Streamlit

Author: Brandon (Data Scientist)
Purpose: Job Application Portfolio - Interactive BI Demo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Card Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #64b5f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #64b5f6;
        margin: 1rem 0;
    }
    .stApp {
        background-color: #1a202c;
        color: #e2e8f0;
    }
    .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        # Load customer segments
        segments_df = pd.read_csv('data/processed/customer_segments.csv')
        
        # Load regional analysis  
        regional_df = pd.read_csv('data/processed/regional_analysis.csv')
        
        # Load transactions
        transactions_df = pd.read_csv('data/raw/synthetic_transactions.csv')
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        
        return segments_df, regional_df, transactions_df
    except FileNotFoundError:
        st.error("Data files not found. Please run 'python main.py' first to generate the data.")
        return None, None, None

def create_kpi_metrics(segments_df, transactions_df):
    """Create KPI metrics display."""
    total_customers = len(segments_df)
    total_revenue = segments_df['monetary_value'].sum()
    avg_customer_value = total_revenue / total_customers
    total_transactions = len(transactions_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            delta=f"{len(segments_df[segments_df['segment'] == 'New_Customers']):,} new"
        )
    
    with col2:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue/1000000:.1f}M",
            delta=f"${total_revenue/total_customers:.0f} per customer"
        )
    
    with col3:
        st.metric(
            label="Avg Customer Value",
            value=f"${avg_customer_value:,.0f}",
            delta=f"{(avg_customer_value/5000)*100:.0f}% vs benchmark"
        )
    
    with col4:
        st.metric(
            label="Total Transactions",
            value=f"{total_transactions:,}",
            delta=f"${transactions_df['amount'].mean():.0f} avg"
        )

def create_revenue_analysis(segments_df):
    """Create revenue analysis charts."""
    st.subheader("💰 Revenue Analysis by Customer Segment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue by segment bar chart
        segment_revenue = segments_df.groupby('segment')['monetary_value'].sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=segment_revenue.index,
            y=segment_revenue.values,
            title="Total Revenue by Customer Segment",
            labels={'x': 'Customer Segment', 'y': 'Total Revenue ($)'},
            color=segment_revenue.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment distribution pie chart
        segment_counts = segments_df['segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Segment Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_rfm_analysis(segments_df):
    """Create RFM analysis visualization."""
    st.subheader("🎯 RFM Analysis: Customer Segmentation")
    
    # Interactive RFM scatter plot
    fig = px.scatter(
        segments_df,
        x='recency',
        y='monetary_value',
        size='frequency',
        color='segment',
        hover_data=['customer_id', 'avg_transaction_value'],
        title="RFM Analysis: Recency vs Monetary Value (sized by Frequency)",
        labels={
            'recency': 'Recency (Days Since Last Purchase)',
            'monetary_value': 'Monetary Value ($)',
            'frequency': 'Purchase Frequency'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics table
    st.subheader("📊 Segment Characteristics Summary")
    segment_summary = segments_df.groupby('segment').agg({
        'customer_id': 'count',
        'monetary_value': ['mean', 'sum'],
        'frequency': 'mean',
        'recency': 'mean',
        'avg_transaction_value': 'mean'
    }).round(2)
    
    # Flatten column names
    segment_summary.columns = ['Customer_Count', 'Avg_Value', 'Total_Revenue', 'Avg_Frequency', 'Avg_Recency', 'Avg_Transaction']
    segment_summary['Revenue_Percentage'] = (segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100).round(1)
    
    st.dataframe(segment_summary, use_container_width=True)

def create_regional_analysis(regional_df, segments_df):
    """Create regional performance analysis."""
    st.subheader("🌍 Regional Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional spending bar chart
        fig = px.bar(
            regional_df,
            x='Spending_Per_Customer',
            y=regional_df.index,
            orientation='h',
            title="Average Spending per Customer by Region",
            labels={'Spending_Per_Customer': 'Average Spending per Customer ($)', 'y': 'Region'},
            color='Spending_Per_Customer',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer distribution by region
        customer_region = segments_df['region'].value_counts()
        
        fig = px.pie(
            values=customer_region.values,
            names=customer_region.index,
            title="Customer Distribution by Region"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_trend_analysis(transactions_df):
    """Create time-based trend analysis."""
    st.subheader("📈 Transaction Trends Analysis")
    
    # Monthly revenue trends
    monthly_data = transactions_df.copy()
    monthly_data['month'] = monthly_data['transaction_date'].dt.to_period('M')
    monthly_revenue = monthly_data.groupby('month')['amount'].sum()
    monthly_transactions = monthly_data.groupby('month').size()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Revenue Trends', 'Monthly Transaction Volume'),
        vertical_spacing=0.1
    )
    
    # Revenue trend
    fig.add_trace(
        go.Scatter(
            x=[str(m) for m in monthly_revenue.index],
            y=monthly_revenue.values,
            mode='lines+markers',
            name='Monthly Revenue',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # Transaction volume trend
    fig.add_trace(
        go.Scatter(
            x=[str(m) for m in monthly_transactions.index],
            y=monthly_transactions.values,
            mode='lines+markers',
            name='Monthly Transactions',
            line=dict(color='#ff7f0e', width=3)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(title_text="Transaction Count", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def create_business_insights(segments_df, regional_df):
    """Create business insights and recommendations."""
    st.subheader("💡 Key Business Insights & Recommendations")
    
    # Calculate key insights
    total_revenue = segments_df['monetary_value'].sum()
    best_region = regional_df['Spending_Per_Customer'].idxmax()
    worst_region = regional_df['Spending_Per_Customer'].idxmin()
    performance_gap = regional_df.loc[best_region, 'Spending_Per_Customer'] - regional_df.loc[worst_region, 'Spending_Per_Customer']
    
    vip_customers = len(segments_df[segments_df['segment'] == 'VIP_Champions'])
    vip_revenue = segments_df[segments_df['segment'] == 'VIP_Champions']['monetary_value'].sum()
    
    # Display insights in styled boxes
    insights = [
        f"🏆 **Top Performing Region**: {best_region} with ${regional_df.loc[best_region, 'Spending_Per_Customer']:,.0f} per customer",
        f"📊 **Regional Opportunity**: ${performance_gap:,.0f} spending gap between best and worst performing regions",
        f"💎 **VIP Champions**: {vip_customers:,} customers generating ${vip_revenue/1000000:.1f}M in revenue ({vip_revenue/total_revenue*100:.1f}% of total)",
        f"🎯 **Growth Potential**: Focus on converting {len(segments_df[segments_df['segment'] == 'Potential_Loyalists']):,} Potential Loyalists to higher tiers"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 🎯 Strategic Recommendations")
    recommendations = [
        f"**Expand in High-Performing Regions**: Replicate successful strategies from {best_region} to other regions",
        f"**VIP Customer Retention**: Implement premium services for {vip_customers:,} VIP Champions worth ${vip_revenue/1000000:.1f}M",
        f"**Regional Development**: Bridge the ${performance_gap:,.0f} spending gap through targeted marketing",
        f"**Segment Conversion**: Develop programs to upgrade Potential Loyalists and New Customers to higher value tiers"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    segments_df, regional_df, transactions_df = load_data()
    
    if segments_df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔧 Dashboard Filters")
    
    # Segment filter
    selected_segments = st.sidebar.multiselect(
        "Select Customer Segments",
        options=segments_df['segment'].unique(),
        default=segments_df['segment'].unique()
    )
    
    # Region filter
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=segments_df['region'].unique(),
        default=segments_df['region'].unique()
    )
    
    # Filter data based on selections
    filtered_segments = segments_df[
        (segments_df['segment'].isin(selected_segments)) &
        (segments_df['region'].isin(selected_regions))
    ]
    
    # KPI Metrics
    create_kpi_metrics(filtered_segments, transactions_df)
    st.markdown("---")
    
    # Revenue Analysis
    create_revenue_analysis(filtered_segments)
    st.markdown("---")
    
    # RFM Analysis
    create_rfm_analysis(filtered_segments)
    st.markdown("---")
    
    # Regional Analysis
    create_regional_analysis(regional_df, filtered_segments)
    st.markdown("---")
    
    # Trend Analysis
    create_trend_analysis(transactions_df)
    st.markdown("---")
    
    # Business Insights
    create_business_insights(filtered_segments, regional_df)
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Dashboard built with Python & Streamlit | Data Science Portfolio Project**")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
