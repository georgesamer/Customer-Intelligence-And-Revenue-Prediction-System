# 🎯 Customer Intelligence & Revenue Prediction System

A complete end-to-end data science project that combines **Customer Analytics** (RFM Analysis, Segmentation), **Machine Learning** (Revenue Prediction), and **Product Recommendations** (Market Basket Analysis) to generate actionable business insights.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Output & Results](#output--results)
- [Business Impact](#business-impact)

---

## 🎯 Overview

This system helps businesses:

- **Understand customer behavior** through RFM analysis
- **Segment customers** into actionable groups (Champions, Loyal, At-Risk, etc.)
- **Identify High-Value Customers (HVC)** for targeted campaigns
- **Predict future revenue** using machine learning
- **Assess customer risk** and prioritize retention efforts
- **Recommend personalized products** using Market Basket Analysis
- **Generate executive-level insights** for strategic decision-making

---

## ✨ Features

### 1. Customer Analytics
- **RFM Analysis**: Recency, Frequency, Monetary metrics
- **Customer Segmentation**: 10 distinct business segments
- **HVC Identification**: Automated detection of high-value customers
- **Risk Assessment**: Flag at-risk customers automatically
- **Business Insights**: Segment-specific recommendations

### 2. Machine Learning
- **Revenue Prediction**: Predict customer lifetime value
- **Feature Engineering**: RFM scores, HVC scores
- **Model Evaluation**: R², MAE, RMSE, residual analysis
- **Feature Importance**: Understand revenue drivers

### 3. Customer Scoring
- **Unified Score**: Combines HVC score + predicted revenue
- **Priority Flags**: CRITICAL, HIGH, MEDIUM, NORMAL
- **Risk Status**: High, Medium, Low risk classification

### 4. Product Recommendations ✦ New
- **Market Basket Analysis**: Apriori algorithm — pure Python, no external ML library
- **Association Rules**: Mines A → B rules with support, confidence, and lift
- **Segment-Aware**: Recommendation count and strategy vary per RFM segment
- **Integrated Output**: Recommendations merged directly into the final results CSV

### 5. Visualizations
- RFM Analysis Dashboard (6 charts)
- ML Performance Dashboard (3 charts)
- Customer Insights Dashboard (4 charts)
- Executive Summary (KPIs + charts)

---

## 📁 Project Structure

```
customer_intelligence_system/
│
├── config/
│   └── config.py                        # Project configuration + recommender settings
│
├── data/
│   ├── raw/
│   │   └── online_retail.xlsx           # Original transaction data
│   └── processed/
│       ├── customer_features.csv        # RFM features per customer
│       └── customer_intelligence_results.csv  # Final enriched results
│
├── src/
│   ├── analytics/
│   │   └── customer_analytics.py        # RFM & segmentation
│   │
│   ├── ml/
│   │   └── revenue_predictor.py         # ML prediction model
│   │
│   ├── recommender/                     # ✦ New
│   │   └── product_recommender.py       # Market Basket Analysis engine
│   │
│   ├── pipeline/
│   │   └── intelligence_pipeline.py     # Main orchestration (5 stages)
│   │
│   └── utils/
│       └── visualizer.py                # All visualizations
│
├── outputs/
│   ├── figures/
│   │   ├── rfm_analysis_dashboard.png
│   │   ├── ml_performance_dashboard.png
│   │   ├── customer_insights_dashboard.png
│   │   └── executive_summary.png
│   │
│   ├── reports/
│   └── models/
│       └── revenue_predictor_model.pkl
│
├── main.py                              # Main execution file
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl joblib
```

### Required Libraries
```python
pandas          # Data manipulation
numpy           # Numerical operations
matplotlib      # Plotting
seaborn         # Statistical visualizations
scikit-learn    # Machine learning
openpyxl        # Excel file support
joblib          # Model serialization
```

> The product recommender uses no additional libraries — Apriori is implemented in pure Python + NumPy.

---

## 💻 Usage

### Quick Start

1. **Prepare your data**
   - Place your retail transaction data in `data/raw/`
   - Ensure it has: `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, `UnitPrice`

2. **Update configuration** (optional)
   - Edit `config/config.py` to customize paths, thresholds, and recommender settings

3. **Run the system**
   ```bash
   python main.py
   ```

### Step-by-Step Execution

The system automatically runs through 5 stages:

#### Stage 1: Customer Analytics
```python
# Loads data, performs RFM analysis, segments customers
# Output: customer_features.csv
```

#### Stage 2: ML Prediction
```python
# Trains Linear Regression model, makes predictions
# Output: revenue_predictor_model.pkl
```

#### Stage 3: Customer Scoring
```python
# Generates unified scores and risk flags
# Output: enriched results dataframe
```

#### Stage 4: Insights & Visualization
```python
# Creates dashboards and business reports
# Output: 4 PNG visualizations
```

#### Stage 5: Product Recommendations ✦ New
```python
# Mines association rules, generates per-customer product recommendations
# Output: Recommended_Products column added to customer_intelligence_results.csv
```

### Recommender Configuration

All recommender settings live in `config/config.py` under `RECOMMENDER_CONFIG`:

```python
RECOMMENDER_CONFIG = {
    'min_support':    0.02,   # Lower → more rules, more noise
    'min_confidence': 0.30,   # P(B|A) threshold
    'min_lift':       1.2,    # Minimum strength above random chance
    'min_support_count': 10,  # Filter out rare/obscure products
}
```

---

## 🔬 Methodology

### RFM Analysis

**Recency (R)**: Days since last purchase — lower is better, score 1–5 (5 = best)

**Frequency (F)**: Number of purchases — higher is better, score 1–5 (5 = best)

**Monetary (M)**: Total spend — higher is better, score 1–5 (5 = best)

### Customer Segments

| Segment | RF Pattern | Description |
|---------|-----------|-------------|
| **Champions** | 5-5 | Best customers — recent, frequent, high spend |
| **Loyal Customers** | 3-4 / 4-5 | Regular buyers with good value |
| **Potential Loyalists** | 4-5 / 2-3 | Recent customers, growth potential |
| **New Customers** | 5-1 | Brand new, need nurturing |
| **Promising** | 4-1 | Recent buyers, encourage repeat |
| **Need Attention** | 3-3 | Average customers, re-engage |
| **About to Sleep** | 3 / 1-2 | Decreasing engagement |
| **At Risk** | 1-2 / 3-4 | Used to be good, now declining |
| **Can't Loose** | 1-2 / 5 | High spenders who haven't returned |
| **Hibernating** | 1-2 / 1-2 | Long gone, minimal value |

### HVC Identification

```python
HVC_Score = 0.3 × R_Score + 0.3 × F_Score + 0.4 × M_Score
```

Top 20% by monetary value = High-Value Customers. Weighted scoring emphasizes monetary contribution.

### ML Model

**Algorithm**: Linear Regression

**Features**: Recency, Frequency, R_Score, F_Score, M_Score, HVC_Score

**Target**: Monetary (customer revenue)

**Evaluation Metrics**: R², MAE, RMSE

### Customer Scoring

```python
Customer_Score = Normalized(HVC_Score) + Normalized(Predicted_Revenue)
```

**Risk Assessment**:
- High Risk: Recency > 90 days AND Frequency < 2
- Medium Risk: Recency > 90 days OR Frequency < 2
- Low Risk: Recent and frequent

**Priority Flags**:
- CRITICAL: High-Value + High Risk
- HIGH: High-Value customer
- MEDIUM: High Risk customer
- NORMAL: Everyone else

### Product Recommendations (Market Basket Analysis) ✦ New

The recommender runs **Apriori** directly on raw transaction data — no re-reading the Excel file since the cleaned DataFrame is already in memory from Stage 1.

**Step 1 — Build basket matrix**

Each row = one invoice. Each column = one product (StockCode). Value = 1 if the product appeared in that invoice, 0 otherwise. Rare products (appearing in fewer than `min_support_count` invoices) are removed first.

**Step 2 — Mine frequent itemsets**

Frequent 1-itemsets (single products) and 2-itemsets (pairs) are mined using the support threshold:

```
Support(A, B) = invoices containing both A and B / total invoices
```

**Step 3 — Generate association rules**

For every frequent pair {A, B}, two rules are derived:

```
A → B   confidence = Support(A,B) / Support(A)
B → A   confidence = Support(A,B) / Support(B)
```

Only rules passing both `min_confidence` and `min_lift` thresholds are kept.

**Step 4 — Recommend per customer**

For each customer:
1. Look up all products they have already purchased
2. Find all rules where the antecedent is in their purchase history
3. Filter out products they already own
4. Rank candidates by `lift × confidence`
5. Keep top-N based on their RFM segment

**Segment-aware recommendation counts:**

| Segment | Max Recommendations | Strategy |
|---------|-------------------|---------|
| Champions | 5 | Premium Cross-Sell |
| Loyal Customers | 5 | Loyalty Upsell |
| Potential Loyalists | 4 | Growth Engagement |
| At Risk | 4 | Retention Recommendations |
| Can't Loose | 5 | High-Priority Recovery |
| New Customers | 3 | Onboarding Recommendations |
| Promising | 3 | Repeat Purchase Nudge |
| Need Attention | 3 | Re-engagement Offers |
| About to Sleep | 3 | Win-Back Suggestions |
| Hibernating | 2 | Reactivation Picks |

---

## 📊 Output & Results

### 1. Customer Features (`customer_features.csv`)
Contains RFM metrics, scores, segments, and customer types.

### 2. Final Results (`customer_intelligence_results.csv`)

Enriched customer table with 18 columns:

| Column | Description |
|--------|-------------|
| CustomerID | Unique customer identifier |
| Customer_Score | Unified score (0–100) |
| Customer_Type | High-Value / Regular |
| Segment | RFM segment label |
| Risk_Status | High / Medium / Low Risk |
| Priority_Flag | CRITICAL / HIGH / MEDIUM / NORMAL |
| Actual_Revenue | Historical revenue |
| Predicted_Revenue | ML-predicted revenue |
| Prediction_Error | Predicted − Actual |
| Recency | Days since last purchase |
| Frequency | Number of invoices |
| Monetary | Total spend |
| R_Score / F_Score / M_Score | RFM scores (1–5) |
| HVC_Score | Weighted RFM score |
| RFM_Segment | Raw RF score string |
| **Recommended_Products** | **Pipe-separated product recommendations ✦ New** |

### 3. Visualizations

#### RFM Analysis Dashboard
Segment distribution, revenue by segment, customer type breakdown, RFM scatter plots, HVC score distribution.

#### ML Performance Dashboard
Predicted vs Actual plot, residual analysis, feature importance.

#### Customer Insights Dashboard
Score distribution by type, risk status breakdown, top 10 customers, revenue comparison by segment.

#### Executive Summary
Total customers KPI, total revenue KPI, HVC percentage, top segments chart, model performance metrics.

### 4. Saved Model
Trained ML model saved to `outputs/models/revenue_predictor_model.pkl` for future predictions without retraining.

---

## 💼 Business Impact

### Strategic Use Cases

1. **Retention Campaigns**
   - Target "At Risk" and "Can't Loose" segments
   - Focus on CRITICAL priority customers
   - Estimated impact: Reduce churn by 15–25%

2. **Revenue Optimization**
   - Prioritize HVC customers for upsell
   - Allocate marketing budget by segment
   - Estimated impact: Increase revenue per customer by 10–20%

3. **Customer Lifetime Value (CLV)**
   - Predict future value for acquisition decisions
   - Calculate break-even for retention costs
   - Estimated impact: Improve CAC/LTV ratio

4. **Personalized Product Recommendations ✦ New**
   - Each customer receives recommendations tailored to their purchase history
   - Recommendation volume scales with segment value (Champions get 5, Hibernating get 2)
   - Rules are ranked by lift × confidence — highest-quality associations surface first
   - Estimated impact: Increase average order value by 10–15% through relevant cross-sell

### Example Insights

**Champions (633 customers, $4.3M revenue)**
- Strategy: VIP program with exclusive benefits
- Tactics: Early access, personal account manager, loyalty rewards, premium cross-sell recommendations
- Expected ROI: 5x on retention investment

**Can't Loose (63 customers, $176K at risk)**
- Strategy: Emergency retention with executive involvement
- Tactics: Personal outreach, special pricing, service recovery, high-priority product recommendations
- Expected ROI: Save 60–70% of at-risk revenue

---

## 🎓 Technical Highlights

- **Modular Architecture**: Clean separation of concerns across 5 independent stages
- **Production-Ready**: Python modules, not notebooks
- **No Extra Dependencies**: Apriori implemented in pure Python + NumPy — no mlxtend required
- **Memory-Efficient**: Recommender reuses the already-cleaned DataFrame from Stage 1, no redundant file reads
- **Scalable**: Can handle hundreds of thousands of transactions
- **Configurable**: All thresholds and parameters centralized in `config.py`
- **Documented**: Comprehensive docstrings throughout

---

## 🔄 Future Enhancements

1. **Advanced ML Models**
   - Random Forest, XGBoost for better revenue prediction accuracy
   - Time series forecasting for revenue trends

2. **Real-Time Scoring**
   - API endpoint for live customer scoring
   - Monitoring dashboard for score drift

3. **A/B Testing Framework**
   - Test retention strategies by segment
   - Measure campaign effectiveness against control groups

4. **Automated Reporting**
   - Scheduled daily/weekly reports
   - Email alerts for CRITICAL priority customers

---

## 👨‍💻 Author

**George Samer — AI/ML Engineer**

This project demonstrates:
- End-to-end data science pipeline
- Business analytics expertise
- Machine learning implementation
- Unsupervised learning (Market Basket Analysis / Apriori)
- Production-quality Python code
- Strategic business thinking

Perfect for:
- Data Science & ML Engineering interviews
- Portfolio demonstration
- Graduation projects
- Business case studies

---

## 📝 License

This project is created for educational and professional portfolio purposes.

---

## 🙏 Acknowledgments

Built with:
- Python ecosystem (pandas, numpy, scikit-learn, matplotlib, seaborn)
- RFM methodology from marketing analytics
- Apriori algorithm from association rule mining literature
- Best practices from production ML systems

---

**Ready to use for your next interview or client presentation! 🚀**
