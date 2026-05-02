"""
Main Execution File
Run this file to execute the complete Customer Intelligence System
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config.config import ONLINE_RETAIL_FILE
from src.pipeline.intelligence_pipeline import CustomerIntelligencePipeline


def main():
    """
    Main execution function
    Runs the complete Customer Intelligence & Revenue Prediction System
    """

    # Configuration
    DATA_PATH = ONLINE_RETAIL_FILE
    ENABLE_VISUALIZATIONS = True

    # Initialize and run pipeline
    pipeline = CustomerIntelligencePipeline(
        data_path=DATA_PATH,
        visualize=ENABLE_VISUALIZATIONS
    )

    # Execute complete pipeline
    pipeline.run_complete_pipeline()

    # Print final message
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
    📁 Check the following directories for outputs:

    1. data/processed/
       - customer_features.csv               (RFM features)
       - customer_intelligence_results.csv   (Final results — now includes
                                              Recommended_Products column)

    2. outputs/figures/
       - rfm_analysis_dashboard.png
       - ml_performance_dashboard.png
       - customer_insights_dashboard.png
       - executive_summary.png

    3. outputs/models/
       - revenue_predictor_model.pkl         (Saved ML model)

    💡 Use the results to:
       - Target high-value customers with VIP programs
       - Launch retention campaigns for at-risk customers
       - Optimize marketing spend by segment
       - Predict and plan for future revenue
       - Send personalized product recommendations per customer  ← NEW

    🛒 Product Recommendations (Stage 5):
       - Based on Market Basket Analysis (Association Rules / Apriori)
       - Each customer gets segment-aware recommendations
         (Champions → 5 premium picks, Hibernating → 2 reactivation picks, etc.)
       - Recommendations are stored in the Recommended_Products column
         of customer_intelligence_results.csv
    """)


if __name__ == "__main__":
    main()
