"""
Visualization Module
Creates all charts and plots for the Customer Intelligence System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import VIZ_CONFIG, FIGURES_DIR

# Set style
sns.set_style(VIZ_CONFIG['style'])
plt.rcParams['figure.dpi'] = VIZ_CONFIG['dpi']


class Visualizer:
    """Handles all visualizations for the project"""
    
    def __init__(self, save_figures=True):
        """Initialize visualizer"""
        self.save_figures = save_figures
        self.figures_dir = FIGURES_DIR
        
    def _save_figure(self, filename):
        """Helper to save figures"""
        if self.save_figures:
            filepath = self.figures_dir / filename
            plt.savefig(filepath, bbox_inches='tight', dpi=VIZ_CONFIG['dpi'])
            print(f"  ✓ Saved: {filename}")
    
    def plot_rfm_analysis(self, rfm_data):
        """Create comprehensive RFM analysis dashboard"""
        print("\n📊 Generating RFM Analysis Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Segment Distribution
        segment_counts = rfm_data['Segment'].value_counts().sort_values(ascending=True)
        axes[0, 0].barh(range(len(segment_counts)), segment_counts.values, color='steelblue')
        axes[0, 0].set_yticks(range(len(segment_counts)))
        axes[0, 0].set_yticklabels(segment_counts.index)
        axes[0, 0].set_xlabel('Number of Customers')
        axes[0, 0].set_title('Customer Segment Distribution')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Revenue by Segment
        segment_revenue = rfm_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(segment_revenue)))
        axes[0, 1].barh(range(len(segment_revenue)), segment_revenue.values, color=colors)
        axes[0, 1].set_yticks(range(len(segment_revenue)))
        axes[0, 1].set_yticklabels(segment_revenue.index)
        axes[0, 1].set_xlabel('Total Revenue ($)')
        axes[0, 1].set_title('Revenue by Customer Segment')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Customer Type Distribution (HVC vs Regular)
        customer_type_counts = rfm_data['Customer_Type'].value_counts()
        colors_pie = ['#ff6b6b', '#4ecdc4']
        wedges, texts, autotexts = axes[0, 2].pie(
            customer_type_counts.values, 
            labels=customer_type_counts.index,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )
        axes[0, 2].set_title('Customer Type Distribution')
        
        # 4. RFM Scatter: Recency vs Monetary (colored by Frequency)
        scatter = axes[1, 0].scatter(
            rfm_data['Recency'], 
            rfm_data['Monetary'],
            c=rfm_data['Frequency'],
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        axes[1, 0].set_xlabel('Recency (days)')
        axes[1, 0].set_ylabel('Monetary Value ($)')
        axes[1, 0].set_title('Recency vs Monetary (sized by Frequency)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Frequency')
        axes[1, 0].grid(alpha=0.3)
        
        # 5. HVC Score Distribution
        axes[1, 1].hist(rfm_data['HVC_Score'], bins=30, color='#95a5a6', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(rfm_data['HVC_Score'].mean(), color='red', linestyle='--', 
                           linewidth=2, label=f"Mean: {rfm_data['HVC_Score'].mean():.2f}")
        axes[1, 1].axvline(rfm_data['HVC_Score'].median(), color='blue', linestyle='--',
                           linewidth=2, label=f"Median: {rfm_data['HVC_Score'].median():.2f}")
        axes[1, 1].set_xlabel('HVC Score')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_title('HVC Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 6. Customer Type by Segment (Stacked Bar)
        ct_segment = pd.crosstab(rfm_data['Segment'], rfm_data['Customer_Type'])
        ct_segment.plot(kind='bar', stacked=True, ax=axes[1, 2], 
                       color=['#ffbe76', '#74b9ff'], width=0.7)
        axes[1, 2].set_title('Customer Type Distribution by Segment')
        axes[1, 2].set_xlabel('Segment')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend(title='Customer Type', loc='upper right')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('rfm_analysis_dashboard.png')
        plt.show()
        
        return self
    
    def plot_ml_performance(self, y_test, y_pred, feature_importance_df):
        """Create ML model performance visualizations"""
        print("\n📊 Generating ML Performance Visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('ML Model Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual
        axes[0].scatter(y_test, y_pred, alpha=0.6, s=50, color='#3498db')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Revenue ($)', fontsize=11)
        axes[0].set_ylabel('Predicted Revenue ($)', fontsize=11)
        axes[0].set_title('Predicted vs Actual Revenue')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. Residual Plot
        residuals = y_pred - y_test
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='#e74c3c')
        axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Revenue ($)', fontsize=11)
        axes[1].set_ylabel('Residuals ($)', fontsize=11)
        axes[1].set_title('Residual Plot')
        axes[1].grid(alpha=0.3)
        
        # 3. Feature Importance
        feature_importance_df = feature_importance_df.sort_values('Coefficient', ascending=True)
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in feature_importance_df['Coefficient']]
        
        axes[2].barh(range(len(feature_importance_df)), feature_importance_df['Coefficient'], color=colors)
        axes[2].set_yticks(range(len(feature_importance_df)))
        axes[2].set_yticklabels(feature_importance_df['Feature'])
        axes[2].set_xlabel('Coefficient Value', fontsize=11)
        axes[2].set_title('Feature Importance (Coefficients)')
        axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('ml_performance_dashboard.png')
        plt.show()
        
        return self
    
    def plot_customer_insights(self, results_df):
        """Create business insights visualizations"""
        print("\n📊 Generating Customer Insights Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Intelligence Insights', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Customer Score Distribution by Type
        hvc = results_df[results_df['Customer_Type'] == 'High-Value']['Customer_Score']
        regular = results_df[results_df['Customer_Type'] == 'Regular']['Customer_Score']
        
        axes[0, 0].hist(regular, bins=30, alpha=0.6, label='Regular', color='#95a5a6')
        axes[0, 0].hist(hvc, bins=30, alpha=0.6, label='High-Value', color='#f39c12')
        axes[0, 0].set_xlabel('Customer Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Customer Score Distribution by Type')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Risk Status Distribution
        if 'Risk_Status' in results_df.columns:
            risk_counts = results_df['Risk_Status'].value_counts()
            colors_risk = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
            bar_colors = [colors_risk.get(x, '#95a5a6') for x in risk_counts.index]
            
            axes[0, 1].bar(range(len(risk_counts)), risk_counts.values, color=bar_colors, width=0.6)
            axes[0, 1].set_xticks(range(len(risk_counts)))
            axes[0, 1].set_xticklabels(risk_counts.index, rotation=15)
            axes[0, 1].set_ylabel('Number of Customers')
            axes[0, 1].set_title('Customer Risk Status Distribution')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Top 10 Customers by Score
        top_10 = results_df.nlargest(10, 'Customer_Score')
        axes[1, 0].barh(range(10), top_10['Customer_Score'].values, color='#9b59b6')
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([f"ID: {int(x)}" for x in top_10['CustomerID'].values])
        axes[1, 0].set_xlabel('Customer Score')
        axes[1, 0].set_title('Top 10 Customers by Overall Score')
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].invert_yaxis()
        
        # 4. Predicted vs Actual Revenue by Segment
        segment_comparison = results_df.groupby('Segment').agg({
            'Actual_Revenue': 'sum',
            'Predicted_Revenue': 'sum'
        }).sort_values('Actual_Revenue', ascending=False).head(8)
        
        x = np.arange(len(segment_comparison))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, segment_comparison['Actual_Revenue'], width, 
                      label='Actual', color='#3498db', alpha=0.8)
        axes[1, 1].bar(x + width/2, segment_comparison['Predicted_Revenue'], width,
                      label='Predicted', color='#e74c3c', alpha=0.8)
        
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('Revenue ($)')
        axes[1, 1].set_title('Actual vs Predicted Revenue by Segment')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(segment_comparison.index, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure('customer_insights_dashboard.png')
        plt.show()
        
        return self
    
    def create_executive_summary_chart(self, results_df):
        """Create a single executive summary visualization"""
        print("\n📊 Generating Executive Summary Chart...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('🎯 EXECUTIVE SUMMARY - Customer Intelligence System', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Top row - KPIs
        # 1. Total Customers
        ax1 = fig.add_subplot(gs[0, 0])
        total_customers = len(results_df)
        ax1.text(0.5, 0.5, f"{total_customers:,}", ha='center', va='center', 
                fontsize=40, fontweight='bold', color='#2c3e50')
        ax1.text(0.5, 0.15, "Total Customers", ha='center', va='center', 
                fontsize=14, color='#7f8c8d')
        ax1.axis('off')
        ax1.set_facecolor('#ecf0f1')
        
        # 2. Total Revenue
        ax2 = fig.add_subplot(gs[0, 1])
        total_revenue = results_df['Actual_Revenue'].sum()
        ax2.text(0.5, 0.5, f"${total_revenue/1e6:.1f}M", ha='center', va='center',
                fontsize=40, fontweight='bold', color='#27ae60')
        ax2.text(0.5, 0.15, "Total Revenue", ha='center', va='center',
                fontsize=14, color='#7f8c8d')
        ax2.axis('off')
        ax2.set_facecolor('#d5f4e6')
        
        # 3. HVC Percentage
        ax3 = fig.add_subplot(gs[0, 2])
        hvc_pct = (results_df['Customer_Type'] == 'High-Value').sum() / len(results_df) * 100
        ax3.text(0.5, 0.5, f"{hvc_pct:.1f}%", ha='center', va='center',
                fontsize=40, fontweight='bold', color='#e74c3c')
        ax3.text(0.5, 0.15, "High-Value Customers", ha='center', va='center',
                fontsize=14, color='#7f8c8d')
        ax3.axis('off')
        ax3.set_facecolor('#fadbd8')
        
        # Middle row
        # 4. Segment Distribution
        ax4 = fig.add_subplot(gs[1, :2])
        segment_revenue = results_df.groupby('Segment')['Actual_Revenue'].sum().sort_values(ascending=False).head(6)
        colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.9, len(segment_revenue)))
        ax4.bar(range(len(segment_revenue)), segment_revenue.values, color=colors_gradient)
        ax4.set_xticks(range(len(segment_revenue)))
        ax4.set_xticklabels(segment_revenue.index, rotation=30, ha='right')
        ax4.set_ylabel('Revenue ($)', fontsize=11)
        ax4.set_title('Top Revenue Segments', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Customer Type Pie
        ax5 = fig.add_subplot(gs[1, 2])
        type_counts = results_df['Customer_Type'].value_counts()
        ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=['#ff6b6b', '#4ecdc4'], startangle=90, textprops={'fontsize': 10})
        ax5.set_title('Customer Mix', fontsize=13, fontweight='bold')
        
        # Bottom row
        # 6. Model Performance Metrics
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate metrics
        mae = np.abs(results_df['Prediction_Error']).mean()
        r2 = 1 - (np.sum((results_df['Actual_Revenue'] - results_df['Predicted_Revenue'])**2) / 
                 np.sum((results_df['Actual_Revenue'] - results_df['Actual_Revenue'].mean())**2))
        
        metrics_text = f"""
        ┌─────────────────────────────────────────────────────────────────────────────────┐
        │  MODEL PERFORMANCE: R² = {r2:.3f}  |  MAE = ${mae:,.0f}  |  Prediction Accuracy: {(1-mae/results_df['Actual_Revenue'].mean())*100:.1f}%  │
        └─────────────────────────────────────────────────────────────────────────────────┘
        """
        ax6.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
        
        self._save_figure('executive_summary.png')
        plt.show()
        
        return self