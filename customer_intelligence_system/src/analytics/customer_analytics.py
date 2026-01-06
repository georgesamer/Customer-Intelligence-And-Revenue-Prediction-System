"""
Customer Analytics Module
Performs RFM Analysis, Customer Segmentation, and HVC Identification
"""

import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import (
    RFM_CONFIG, SEGMENT_MAPPING, HVC_CONFIG, 
    HEADER_LINE, SUBHEADER_LINE
)


class CustomerAnalytics:
    """
    Comprehensive Customer Analytics Engine
    Performs RFM analysis, customer segmentation, and HVC identification
    """
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.df_raw = None
        self.df_clean = None
        self.rfm_data = None
        self.feature_data = None
        
    def load_data(self):
        """Load data from Excel or CSV file"""
        print(HEADER_LINE)
        print("LOADING DATA")
        print(HEADER_LINE)
        
        try:
            if str(self.data_path).endswith('.xlsx'):
                self.df_raw = pd.read_excel(self.data_path)
            else:
                self.df_raw = pd.read_csv(self.data_path)
            
            print(f"✓ Data loaded successfully")
            print(f"  Shape: {self.df_raw.shape[0]:,} rows × {self.df_raw.shape[1]} columns")
            print(f"  Columns: {list(self.df_raw.columns)}")
            return self
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print(f"\n{HEADER_LINE}")
        print("DATA PREPROCESSING")
        print(HEADER_LINE)
        
        self.df_clean = self.df_raw.copy()
        
        # Initial stats
        initial_rows = len(self.df_clean)
        print(f"\n1. Initial dataset: {initial_rows:,} rows")
        
        # Remove missing values
        missing_before = self.df_clean.isna().sum().sum()
        self.df_clean.dropna(inplace=True)
        print(f"2. Removed {missing_before:,} missing values → {len(self.df_clean):,} rows remaining")
        
        # Remove cancelled orders (invoices starting with 'C')
        if 'InvoiceNo' in self.df_clean.columns:
            cancelled_count = self.df_clean['InvoiceNo'].astype(str).str.contains('C', na=False).sum()
            self.df_clean = self.df_clean[~self.df_clean['InvoiceNo'].astype(str).str.contains('C', na=False)]
            print(f"3. Removed {cancelled_count:,} cancelled orders → {len(self.df_clean):,} rows remaining")
        
        # Calculate TotalPrice
        if 'Quantity' in self.df_clean.columns and 'UnitPrice' in self.df_clean.columns:
            self.df_clean['TotalPrice'] = self.df_clean['Quantity'] * self.df_clean['UnitPrice']
        
        # Remove negative quantities and prices
        negative_count = len(self.df_clean[(self.df_clean['Quantity'] <= 0) | (self.df_clean['TotalPrice'] <= 0)])
        self.df_clean = self.df_clean[(self.df_clean['Quantity'] > 0) & (self.df_clean['TotalPrice'] > 0)]
        print(f"4. Removed {negative_count:,} negative/zero values → {len(self.df_clean):,} rows remaining")
        
        # Final summary
        data_loss_pct = ((initial_rows - len(self.df_clean)) / initial_rows) * 100
        print(f"\n✓ Preprocessing complete")
        print(f"  Final dataset: {len(self.df_clean):,} rows ({data_loss_pct:.1f}% data loss)")
        print(f"  Date range: {self.df_clean['InvoiceDate'].min()} to {self.df_clean['InvoiceDate'].max()}")
        
        return self
    
    def calculate_rfm(self):
        """Calculate RFM metrics for each customer"""
        print(f"\n{HEADER_LINE}")
        print("RFM ANALYSIS")
        print(HEADER_LINE)
        
        # Reference date (1 day after last transaction)
        reference_date = self.df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
        print(f"\nReference Date: {reference_date.date()}")
        
        # Calculate RFM
        rfm_agg = self.df_clean.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': lambda x: x.nunique(),                         # Frequency
            'TotalPrice': lambda x: x.sum()                             # Monetary
        })
        
        self.rfm_data = rfm_agg.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        })
        
        # Filter positive monetary values
        self.rfm_data = self.rfm_data[self.rfm_data['Monetary'] > 0]
        
        print(f"\n✓ RFM calculated for {len(self.rfm_data):,} customers")
        print(f"\nRFM Statistics:")
        print(self.rfm_data.describe().round(2).to_string())
        
        return self
    
    def create_rfm_scores(self):
        """Create RFM scores using quantile-based binning"""
        print(f"\n{SUBHEADER_LINE}")
        print("Creating RFM Scores...")
        
        # Calculate scores (R is inverse - lower recency is better)
        self.rfm_data['R_Score'] = pd.qcut(
            self.rfm_data['Recency'], 
            RFM_CONFIG['quantiles'], 
            labels=[5, 4, 3, 2, 1]
        )
        
        self.rfm_data['F_Score'] = pd.qcut(
            self.rfm_data['Frequency'].rank(method="first"), 
            RFM_CONFIG['quantiles'], 
            labels=[1, 2, 3, 4, 5]
        )
        
        self.rfm_data['M_Score'] = pd.qcut(
            self.rfm_data['Monetary'], 
            RFM_CONFIG['quantiles'], 
            labels=[1, 2, 3, 4, 5]
        )
        
        # Convert to integers for calculations
        self.rfm_data['R_Score'] = self.rfm_data['R_Score'].astype(int)
        self.rfm_data['F_Score'] = self.rfm_data['F_Score'].astype(int)
        self.rfm_data['M_Score'] = self.rfm_data['M_Score'].astype(int)
        
        print("✓ RFM scores created (1-5 scale)")
        
        return self
    
    def segment_customers(self):
        """Segment customers based on RFM scores"""
        print(f"\n{HEADER_LINE}")
        print("CUSTOMER SEGMENTATION")
        print(HEADER_LINE)
        
        # Create RFM segment string
        self.rfm_data['RFM_Segment'] = (
            self.rfm_data['R_Score'].astype(str) + 
            self.rfm_data['F_Score'].astype(str)
        )
        
        # Map to business segments
        self.rfm_data['Segment'] = self.rfm_data['RFM_Segment'].replace(
            SEGMENT_MAPPING, 
            regex=True
        )
        
        # Display segment distribution
        print("\nSegment Distribution:")
        segment_dist = self.rfm_data['Segment'].value_counts().sort_values(ascending=False)
        for segment, count in segment_dist.items():
            percentage = (count / len(self.rfm_data)) * 100
            print(f"  {segment:25s}: {count:5,} customers ({percentage:5.1f}%)")
        
        # Segment characteristics
        print(f"\n{SUBHEADER_LINE}")
        print("Segment Characteristics:")
        print(SUBHEADER_LINE)
        
        segment_summary = self.rfm_data.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum', 'count']
        }).round(2)
        
        segment_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Count']
        segment_summary = segment_summary.sort_values('Total_Revenue', ascending=False)
        
        print(segment_summary.to_string())
        
        return self
    
    def identify_hvc(self):
        """Identify High-Value Customers"""
        print(f"\n{HEADER_LINE}")
        print("HIGH-VALUE CUSTOMER (HVC) IDENTIFICATION")
        print(HEADER_LINE)
        
        # Define HVC threshold
        percentile = HVC_CONFIG['percentile_threshold']
        hvc_threshold = self.rfm_data['Monetary'].quantile(percentile / 100)
        
        print(f"\nHVC Definition: Top {100-percentile}% by Monetary Value")
        print(f"Monetary Threshold: ${hvc_threshold:,.2f}")
        
        # Classify customers
        self.rfm_data['Customer_Type'] = self.rfm_data['Monetary'].apply(
            lambda x: 'High-Value' if x >= hvc_threshold else 'Regular'
        )
        
        # Calculate HVC Score (weighted RFM)
        weights = RFM_CONFIG['weights']
        self.rfm_data['HVC_Score'] = (
            self.rfm_data['R_Score'] * weights['recency'] +
            self.rfm_data['F_Score'] * weights['frequency'] +
            self.rfm_data['M_Score'] * weights['monetary']
        )
        
        # Statistics
        hvc_count = (self.rfm_data['Customer_Type'] == 'High-Value').sum()
        hvc_pct = (hvc_count / len(self.rfm_data)) * 100
        
        hvc_revenue = self.rfm_data[self.rfm_data['Customer_Type'] == 'High-Value']['Monetary'].sum()
        total_revenue = self.rfm_data['Monetary'].sum()
        hvc_revenue_pct = (hvc_revenue / total_revenue) * 100
        
        print(f"\n✓ HVC Identification Complete")
        print(f"  Total HVCs: {hvc_count:,} ({hvc_pct:.1f}% of customer base)")
        print(f"  HVC Revenue: ${hvc_revenue:,.2f} ({hvc_revenue_pct:.1f}% of total revenue)")
        print(f"  Average HVC Value: ${hvc_revenue/hvc_count:,.2f}")
        
        # Comparison table
        print(f"\n{SUBHEADER_LINE}")
        print("HVC vs Regular Customer Comparison:")
        comparison = self.rfm_data.groupby('Customer_Type').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum', 'count'],
            'HVC_Score': 'mean'
        }).round(2)
        
        comparison.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Count', 'Avg_HVC_Score']
        print(comparison.to_string())
        
        # Top HVCs
        print(f"\n{SUBHEADER_LINE}")
        print("Top 10 High-Value Customers:")
        top_hvc = self.rfm_data.nlargest(10, 'Monetary')[
            ['Recency', 'Frequency', 'Monetary', 'Segment', 'HVC_Score', 'Customer_Type']
        ]
        print(top_hvc.to_string())
        
        return self
    
    def generate_features(self):
        """Generate final feature table for ML"""
        print(f"\n{HEADER_LINE}")
        print("FEATURE GENERATION")
        print(HEADER_LINE)
        
        self.feature_data = self.rfm_data.copy()
        
        # Add customer ID as column
        self.feature_data['CustomerID'] = self.feature_data.index
        
        # Reorder columns
        feature_cols = [
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'HVC_Score',
            'RFM_Segment', 'Segment', 'Customer_Type'
        ]
        
        self.feature_data = self.feature_data[feature_cols]
        
        print(f"\n✓ Feature table generated")
        print(f"  Shape: {self.feature_data.shape}")
        print(f"  Columns: {list(self.feature_data.columns)}")
        print(f"\nSample Features:")
        print(self.feature_data.head().to_string())
        
        return self
    
    def save_features(self, output_path):
        """Save feature data to CSV"""
        self.feature_data.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to: {output_path}")
        return self
    
    def get_business_insights(self):
        """Generate business insights and recommendations"""
        print(f"\n{HEADER_LINE}")
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print(HEADER_LINE)
        
        total_customers = len(self.rfm_data)
        total_revenue = self.rfm_data['Monetary'].sum()
        hvc_count = (self.rfm_data['Customer_Type'] == 'High-Value').sum()
        
        print(f"\n📊 CUSTOMER BASE OVERVIEW")
        print(f"  • Total Active Customers: {total_customers:,}")
        print(f"  • Total Revenue: ${total_revenue:,.2f}")
        print(f"  • Average Customer Value: ${total_revenue/total_customers:,.2f}")
        print(f"  • High-Value Customers: {hvc_count:,} ({(hvc_count/total_customers)*100:.1f}%)")
        
        print(f"\n🏆 TOP REVENUE SEGMENTS")
        top_segments = self.rfm_data.groupby('Segment')['Monetary'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(3)
        for i, (segment, row) in enumerate(top_segments.iterrows(), 1):
            print(f"  {i}. {segment:25s}: ${row['sum']:12,.2f} ({int(row['count']):,} customers)")
        
        print(f"\n⚠️  AT-RISK CUSTOMERS")
        at_risk_segments = ['At_Risk', 'Cant_Loose', 'Hibernating', 'About_to_Sleep']
        at_risk = self.rfm_data[self.rfm_data['Segment'].isin(at_risk_segments)]
        at_risk_revenue = at_risk['Monetary'].sum()
        print(f"  • Count: {len(at_risk):,} customers")
        print(f"  • Potential Revenue at Risk: ${at_risk_revenue:,.2f}")
        print(f"  • Percentage of Total Revenue: {(at_risk_revenue/total_revenue)*100:.1f}%")
        
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        
        recommendations = {
            'Champions': '🎯 VIP treatment, exclusive offers, early access to new products',
            'Loyal_Customers': '🎁 Loyalty rewards, referral programs, personalized experiences',
            'Potential_Loyalists': '📈 Cross-sell/upsell campaigns, engagement programs',
            'New_Customers': '👋 Onboarding series, welcome discounts, education content',
            'Promising': '🌟 Nurture campaigns, engagement incentives',
            'Need_Attention': '📧 Re-engagement emails, special offers, feedback surveys',
            'About_to_Sleep': '⏰ Win-back campaigns, limited-time offers',
            'At_Risk': '🚨 Urgent retention campaigns, personalized outreach',
            'Cant_Loose': '💰 High-priority retention, account manager contact',
            'Hibernating': '🔔 Reactivation campaigns, major incentives'
        }
        
        for segment, recommendation in recommendations.items():
            count = (self.rfm_data['Segment'] == segment).sum()
            if count > 0:
                print(f"  {segment:25s} ({count:4,}): {recommendation}")
        
        return self