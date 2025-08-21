"""
Configuration settings for Credit Card Analytics project.
Centralizes all configurable parameters for easy maintenance.
"""

from dataclasses import dataclass
from typing import Dict, List
import os

@dataclass
class DataConfig:
    """Data generation and processing configuration."""
    n_customers: int = 10000
    n_transactions: int = 500000
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    random_seed: int = 42

@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    n_clusters: int = 5
    clustering_features: List[str] = None
    test_size: float = 0.2
    n_estimators: int = 100
    
    def __post_init__(self):
        if self.clustering_features is None:
            self.clustering_features = [
                'recency', 'frequency', 'monetary_value', 
                'avg_transaction_value', 'transaction_frequency_per_month',
                'age', 'income'
            ]

@dataclass
class VisualizationConfig:
    """Visualization and dashboard configuration."""
    figure_size: tuple = (20, 16)
    dpi: int = 300
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

@dataclass
class PathConfig:
    """File paths and directory structure."""
    base_dir: str = "."
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs"
    viz_dir: str = "outputs/visualizations"
    dashboard_dir: str = "outputs/dashboards"
    reports_dir: str = "outputs/reports"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        directories = [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.output_dir, self.viz_dir, self.dashboard_dir, self.reports_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Business logic configuration
BUSINESS_CATEGORIES = {
    'Grocery': {'avg_spend': 85, 'frequency': 0.25, 'weekend_boost': 1.3},
    'Gas Station': {'avg_spend': 45, 'frequency': 0.15, 'weekend_boost': 1.1},
    'Restaurant': {'avg_spend': 65, 'frequency': 0.20, 'weekend_boost': 1.8},
    'Retail': {'avg_spend': 120, 'frequency': 0.12, 'weekend_boost': 1.6},
    'Entertainment': {'avg_spend': 95, 'frequency': 0.08, 'weekend_boost': 2.2},
    'Online': {'avg_spend': 75, 'frequency': 0.18, 'weekend_boost': 1.0},
    'Healthcare': {'avg_spend': 150, 'frequency': 0.02, 'weekend_boost': 0.8}
}

GEOGRAPHIC_REGIONS = {
    'Urban_High_Income': {'spending_multiplier': 1.4, 'population_pct': 0.15, 'coords': (40.7589, -73.9851)},
    'Urban_Mid_Income': {'spending_multiplier': 1.1, 'population_pct': 0.25, 'coords': (40.6892, -74.0445)},
    'Suburban': {'spending_multiplier': 1.2, 'population_pct': 0.35, 'coords': (40.7282, -73.7949)},
    'Rural': {'spending_multiplier': 0.8, 'population_pct': 0.25, 'coords': (41.2033, -77.1945)}
}

# Initialize configuration objects
data_config = DataConfig()
model_config = ModelConfig()
viz_config = VisualizationConfig()
path_config = PathConfig()

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file']
}