"""
Configuration file for Customer Intelligence System
Contains all project settings, paths, and constants
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
ECOMMERCE_CUSTOMERS_FILE = RAW_DATA_DIR / "ecommerce_customers.csv"
ONLINE_RETAIL_FILE = RAW_DATA_DIR / "Online+Retail.xlsx"
CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.csv"
FINAL_RESULTS_FILE = PROCESSED_DATA_DIR / "customer_intelligence_results.csv"

# ============================================================================
# RFM ANALYSIS SETTINGS
# ============================================================================
RFM_CONFIG = {
    'quantiles': 5,  # Number of quantiles for RFM scoring
    'weights': {
        'recency': 0.3,
        'frequency': 0.3,
        'monetary': 0.4
    }
}

# RFM Segment Mapping
SEGMENT_MAPPING = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

# ============================================================================
# HVC (HIGH-VALUE CUSTOMER) SETTINGS
# ============================================================================
HVC_CONFIG = {
    'percentile_threshold': 80,  # Top 20% are HVCs
    'min_monetary_value': 0,
    'min_frequency': 1
}

# ============================================================================
# MACHINE LEARNING SETTINGS
# ============================================================================
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'target_column': 'Monetary',  # What we're predicting
    'feature_columns': [
        'Recency',
        'Frequency', 
        'R_Score',
        'F_Score',
        'M_Score',
        'HVC_Score'
    ]
}

# ============================================================================
# RISK ASSESSMENT THRESHOLDS
# ============================================================================
RISK_THRESHOLDS = {
    'high_recency_days': 90,      # More than 90 days = high risk
    'low_frequency_count': 2,      # Less than 2 purchases = low engagement
    'low_monetary_value': 100      # Less than $100 = low value
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VIZ_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'whitegrid',
    'color_palette': 'Set2',
    'save_format': 'png'
}

# ============================================================================
# BUSINESS INSIGHTS SETTINGS
# ============================================================================
INSIGHT_CONFIG = {
    'top_segments_count': 3,
    'top_customers_count': 10,
    'min_at_risk_value': 500  # Minimum value to flag as important at-risk customer
}

# ============================================================================
# PRINT FORMATTING
# ============================================================================
HEADER_LINE = "=" * 80
SUBHEADER_LINE = "-" * 80