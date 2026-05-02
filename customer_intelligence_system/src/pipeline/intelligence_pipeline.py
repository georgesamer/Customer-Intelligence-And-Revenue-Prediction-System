"""
Customer Intelligence Pipeline
Orchestrates the entire analytics and ML workflow
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    ONLINE_RETAIL_FILE, CUSTOMER_FEATURES_FILE, FINAL_RESULTS_FILE,
    RISK_THRESHOLDS, HEADER_LINE, MODELS_DIR
)
from src.analytics.customer_analytics import CustomerAnalytics
from src.ml.revenue_predictor import RevenuePredictor
from src.recommender.product_recommender import ProductRecommender   # ← NEW
from src.utils.visualizer import Visualizer


class CustomerIntelligencePipeline:
    """
    End-to-end pipeline for customer intelligence and revenue prediction.

    Workflow:
    1. Customer Analytics  (RFM, Segmentation, HVC)
    2. Feature Generation
    3. ML Model Training & Prediction
    4. Customer Scoring & Risk Assessment
    5. Product Recommendations          ← NEW
    6. Insights & Visualization
    """

    def __init__(self, data_path, visualize=True):
        """Initialize pipeline"""
        self.data_path = data_path
        self.visualize = visualize
        self.analytics = None
        self.predictor = None
        self.recommender = None          # ← NEW
        self.visualizer = Visualizer(save_figures=True) if visualize else None
        self.results = None

    def run_customer_analytics(self):
        """Execute customer analytics workflow"""
        print("\n" + "="*80)
        print("STAGE 1: CUSTOMER ANALYTICS")
        print("="*80)

        self.analytics = CustomerAnalytics(self.data_path)

        # Run analytics pipeline
        (self.analytics
            .load_data()
            .preprocess_data()
            .calculate_rfm()
            .create_rfm_scores()
            .segment_customers()
            .identify_hvc()
            .generate_features()
            .save_features(CUSTOMER_FEATURES_FILE))

        # Get insights
        self.analytics.get_business_insights()

        # Visualize if enabled
        if self.visualize:
            self.visualizer.plot_rfm_analysis(self.analytics.rfm_data)

        return self

    def run_ml_prediction(self):
        """Execute ML prediction workflow"""
        print("\n" + "="*80)
        print("STAGE 2: MACHINE LEARNING PREDICTION")
        print("="*80)

        self.predictor = RevenuePredictor()

        # Run ML pipeline
        (self.predictor
            .load_features(CUSTOMER_FEATURES_FILE)
            .prepare_data()
            .train_model()
            .evaluate_model()
            .analyze_feature_importance()
            .predict_all_customers())

        # Save model
        model_path = MODELS_DIR / "revenue_predictor_model.pkl"
        self.predictor.save_model(model_path)

        # Visualize if enabled
        if self.visualize:
            self.visualizer.plot_ml_performance(
                self.predictor.y_test,
                self.predictor.y_pred,
                self.predictor.feature_importance
            )

        return self

    def create_customer_scores(self):
        """Generate unified customer scores and risk assessment"""
        print("\n" + "="*80)
        print("STAGE 3: CUSTOMER SCORING & RISK ASSESSMENT")
        print("="*80)

        # Get prediction results
        self.results = self.predictor.get_prediction_results()

        # Normalize HVC_Score to 0-50
        hvc_score_norm = (self.results['HVC_Score'] - self.results['HVC_Score'].min()) / \
                         (self.results['HVC_Score'].max() - self.results['HVC_Score'].min()) * 50

        # Normalize Predicted Revenue to 0-50
        pred_revenue_norm = (self.results['Predicted_Revenue'] - self.results['Predicted_Revenue'].min()) / \
                           (self.results['Predicted_Revenue'].max() - self.results['Predicted_Revenue'].min()) * 50

        self.results['Customer_Score'] = hvc_score_norm + pred_revenue_norm

        # Risk Assessment
        def assess_risk(row):
            if row['Recency'] > RISK_THRESHOLDS['high_recency_days'] and \
               row['Frequency'] < RISK_THRESHOLDS['low_frequency_count']:
                return 'High Risk'
            elif row['Recency'] > RISK_THRESHOLDS['high_recency_days'] or \
                 row['Frequency'] < RISK_THRESHOLDS['low_frequency_count']:
                return 'Medium Risk'
            else:
                return 'Low Risk'

        self.results['Risk_Status'] = self.results.apply(assess_risk, axis=1)

        # Priority Flag
        self.results['Priority_Flag'] = self.results.apply(
            lambda x: 'CRITICAL' if x['Customer_Type'] == 'High-Value' and x['Risk_Status'] == 'High Risk'
                     else 'HIGH' if x['Customer_Type'] == 'High-Value'
                     else 'MEDIUM' if x['Risk_Status'] == 'High Risk'
                     else 'NORMAL',
            axis=1
        )

        print("\n✓ Customer Scoring Complete")
        print(f"\nCustomer Score Statistics:")
        print(f"  Mean:   {self.results['Customer_Score'].mean():.2f}")
        print(f"  Median: {self.results['Customer_Score'].median():.2f}")
        print(f"  Min:    {self.results['Customer_Score'].min():.2f}")
        print(f"  Max:    {self.results['Customer_Score'].max():.2f}")

        print(f"\n✓ Risk Assessment Complete")
        risk_dist = self.results['Risk_Status'].value_counts()
        for risk, count in risk_dist.items():
            pct = (count / len(self.results)) * 100
            print(f"  {risk:15s}: {count:5,} ({pct:5.1f}%)")

        print(f"\n✓ Priority Flags Assigned")
        priority_dist = self.results['Priority_Flag'].value_counts()
        for priority, count in priority_dist.items():
            pct = (count / len(self.results)) * 100
            print(f"  {priority:10s}: {count:5,} ({pct:5.1f}%)")

        return self

    # =========================================================================
    # STAGE 5 — PRODUCT RECOMMENDATIONS  ← NEW
    # =========================================================================

    def run_product_recommendations(self):
        """
        Execute the product recommendation workflow using Market Basket Analysis.

        Uses the cleaned transaction data already loaded by CustomerAnalytics
        so we don't re-read the Excel file. The resulting recommendations are
        merged directly into self.results as a new 'Recommended_Products' column.
        """
        print("\n" + "="*80)
        print("STAGE 5: PRODUCT RECOMMENDATIONS (MARKET BASKET ANALYSIS)")
        print("="*80)

        self.recommender = ProductRecommender()

        # Load transactions from the already-cleaned DataFrame (no re-read needed)
        self.recommender.load_transactions(self.analytics.df_clean)

        # Load segment info from current results
        self.recommender.load_customer_segments(self.results)

        # Run the recommendation pipeline
        (self.recommender
            .build_basket_matrix()
            .mine_association_rules()
            .generate_recommendations())

        # Print diagnostics
        self.recommender.print_sample_recommendations(n=5)
        self.recommender.get_top_recommended_products(top_n=10)

        # Merge recommendations into the main results table
        self.results = self.recommender.merge_with_results(self.results)

        return self

    # =========================================================================
    # STAGES 4 & 6 — unchanged
    # =========================================================================

    def generate_insights(self):
        """Generate comprehensive business insights"""
        print("\n" + "="*80)
        print("STAGE 4: BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*80)

        print("\n" + "🎯 EXECUTIVE SUMMARY".center(80))
        print("="*80)

        total_customers = len(self.results)
        total_revenue = self.results['Actual_Revenue'].sum()
        predicted_revenue = self.results['Predicted_Revenue'].sum()
        avg_customer_value = total_revenue / total_customers

        print(f"\n📊 BUSINESS METRICS")
        print(f"  • Total Active Customers: {total_customers:,}")
        print(f"  • Total Revenue: ${total_revenue:,.2f}")
        print(f"  • Predicted Total Revenue: ${predicted_revenue:,.2f}")
        print(f"  • Average Customer Value: ${avg_customer_value:,.2f}")
        print(f"  • Revenue Prediction Accuracy: {(1 - abs(predicted_revenue - total_revenue)/total_revenue)*100:.1f}%")

        hvc_data = self.results[self.results['Customer_Type'] == 'High-Value']
        hvc_count = len(hvc_data)
        hvc_revenue = hvc_data['Actual_Revenue'].sum()
        hvc_revenue_pct = (hvc_revenue / total_revenue) * 100

        print(f"\n💎 HIGH-VALUE CUSTOMERS (HVC)")
        print(f"  • Count: {hvc_count:,} ({(hvc_count/total_customers)*100:.1f}% of base)")
        print(f"  • Revenue Contribution: ${hvc_revenue:,.2f} ({hvc_revenue_pct:.1f}% of total)")
        print(f"  • Average HVC Value: ${hvc_revenue/hvc_count:,.2f}")

        critical_customers = self.results[self.results['Priority_Flag'] == 'CRITICAL']
        at_risk_revenue = critical_customers['Actual_Revenue'].sum()

        print(f"\n⚠️  CRITICAL AT-RISK CUSTOMERS")
        print(f"  • Count: {len(critical_customers):,}")
        print(f"  • Revenue at Risk: ${at_risk_revenue:,.2f}")
        print(f"  • Percentage of Total Revenue: {(at_risk_revenue/total_revenue)*100:.1f}%")

        print(f"\n🏆 TOP 3 REVENUE SEGMENTS")
        top_segments = self.results.groupby('Segment').agg({
            'Actual_Revenue': 'sum',
            'CustomerID': 'count'
        }).sort_values('Actual_Revenue', ascending=False).head(3)

        for i, (segment, row) in enumerate(top_segments.iterrows(), 1):
            revenue_pct = (row['Actual_Revenue'] / total_revenue) * 100
            print(f"  {i}. {segment:25s}: ${row['Actual_Revenue']:12,.2f} "
                  f"({int(row['CustomerID']):4,} customers, {revenue_pct:.1f}%)")

        print(f"\n💡 STRATEGIC RECOMMENDATIONS BY SEGMENT")
        print("="*80)

        segment_strategies = {
            'Champions': {
                'icon': '👑', 'action': 'VIP Program',
                'tactics': 'Exclusive access, personal account manager, loyalty rewards'
            },
            'Loyal_Customers': {
                'icon': '🎁', 'action': 'Retention & Growth',
                'tactics': 'Referral incentives, cross-sell opportunities, premium features'
            },
            'Potential_Loyalists': {
                'icon': '📈', 'action': 'Acceleration Program',
                'tactics': 'Engagement campaigns, product education, upsell offers'
            },
            'At_Risk': {
                'icon': '🚨', 'action': 'Win-Back Campaign',
                'tactics': 'Reactivation offers, feedback survey, personalized outreach'
            },
            'Cant_Loose': {
                'icon': '💰', 'action': 'Emergency Retention',
                'tactics': 'Executive contact, special pricing, service recovery'
            },
            'Hibernating': {
                'icon': '😴', 'action': 'Reactivation',
                'tactics': 'Major discounts, "we miss you" campaigns, new product showcase'
            }
        }

        for segment, strategy in segment_strategies.items():
            count = (self.results['Segment'] == segment).sum()
            if count > 0:
                revenue = self.results[self.results['Segment'] == segment]['Actual_Revenue'].sum()
                print(f"\n{strategy['icon']} {segment.upper()} ({count:,} customers, ${revenue:,.0f})")
                print(f"  Strategy: {strategy['action']}")
                print(f"  Tactics:  {strategy['tactics']}")

        print(f"\n⭐ TOP 10 CUSTOMERS BY SCORE")
        print("="*80)

        # Show Recommended_Products column if it exists
        top_cols = ['CustomerID', 'Customer_Score', 'Actual_Revenue',
                    'Segment', 'Customer_Type', 'Risk_Status', 'Priority_Flag']
        if 'Recommended_Products' in self.results.columns:
            top_cols.append('Recommended_Products')

        top_10 = self.results.nlargest(10, 'Customer_Score')[top_cols]
        print(top_10.to_string(index=False))

        return self

    def save_results(self):
        """Save final enriched customer table"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        output_columns = [
            'CustomerID', 'Customer_Score', 'Customer_Type', 'Segment',
            'Risk_Status', 'Priority_Flag',
            'Actual_Revenue', 'Predicted_Revenue', 'Prediction_Error',
            'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'HVC_Score',
            'RFM_Segment'
        ]

        # Append Recommended_Products if it exists
        if 'Recommended_Products' in self.results.columns:
            output_columns.append('Recommended_Products')

        final_output = self.results[output_columns]
        final_output.to_csv(FINAL_RESULTS_FILE, index=False)

        print(f"\n✓ Results saved to: {FINAL_RESULTS_FILE}")
        print(f"  Rows    : {len(final_output):,}")
        print(f"  Columns : {len(final_output.columns)}")

        return self

    def visualize_results(self):
        """Create final visualizations"""
        if self.visualize:
            print("\n" + "="*80)
            print("GENERATING VISUALIZATIONS")
            print("="*80)

            self.visualizer.plot_customer_insights(self.results)
            self.visualizer.create_executive_summary_chart(self.results)

        return self

    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        print("\n")
        print("█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "CUSTOMER INTELLIGENCE & REVENUE PREDICTION SYSTEM".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        # Execute all stages in order
        (self
            .run_customer_analytics()       # Stage 1
            .run_ml_prediction()            # Stage 2
            .create_customer_scores()       # Stage 3
            .generate_insights()            # Stage 4
            .run_product_recommendations()  # Stage 5 ← NEW
            .save_results()
            .visualize_results())

        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "PIPELINE EXECUTION COMPLETE ✓".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        return self
