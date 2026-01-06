"""
Revenue Prediction Module
Machine Learning model to predict customer revenue/value
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import ML_CONFIG, HEADER_LINE, SUBHEADER_LINE


class RevenuePredictor:
    """
    Machine Learning model for predicting customer revenue/lifetime value
    Uses Linear Regression on RFM features
    """
    
    def __init__(self):
        """Initialize predictor"""
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_columns = None
        self.target_column = None
        self.metrics = {}
        self.feature_importance = None
        
    def load_features(self, feature_path):
        """Load processed customer features"""
        print(HEADER_LINE)
        print("LOADING CUSTOMER FEATURES")
        print(HEADER_LINE)
        
        try:
            self.data = pd.read_csv(feature_path)
            print(f"\n✓ Features loaded successfully")
            print(f"  Shape: {self.data.shape}")
            print(f"  Columns: {list(self.data.columns)}")
            
            return self
            
        except Exception as e:
            print(f"✗ Error loading features: {e}")
            return None
    
    def prepare_data(self):
        """Prepare features and target for training"""
        print(f"\n{HEADER_LINE}")
        print("DATA PREPARATION")
        print(HEADER_LINE)
        
        # Define features and target
        self.feature_columns = ML_CONFIG['feature_columns']
        self.target_column = ML_CONFIG['target_column']
        
        print(f"\nTarget Variable: {self.target_column}")
        print(f"Feature Variables: {self.feature_columns}")
        
        # Extract X and y
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        
        print(f"\nFeature Matrix Shape: {X.shape}")
        print(f"Target Vector Shape: {y.shape}")
        
        # Train-test split
        test_size = ML_CONFIG['test_size']
        random_state = ML_CONFIG['random_state']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split complete")
        print(f"  Training set: {len(self.X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing set:  {len(self.X_test):,} samples ({test_size*100:.0f}%)")
        print(f"\nTarget Statistics:")
        print(f"  Train - Mean: ${self.y_train.mean():,.2f}, Std: ${self.y_train.std():,.2f}")
        print(f"  Test  - Mean: ${self.y_test.mean():,.2f}, Std: ${self.y_test.std():,.2f}")
        
        return self
    
    def train_model(self):
        """Train Linear Regression model"""
        print(f"\n{HEADER_LINE}")
        print("MODEL TRAINING")
        print(HEADER_LINE)
        
        print("\nTraining Linear Regression model...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        print("✓ Model trained successfully")
        print(f"\nModel Parameters:")
        print(f"  Intercept: ${self.model.intercept_:,.2f}")
        print(f"  Number of Features: {len(self.model.coef_)}")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print(f"\n{HEADER_LINE}")
        print("MODEL EVALUATION")
        print(HEADER_LINE)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        self.metrics['r2'] = r2_score(self.y_test, self.y_pred)
        self.metrics['mae'] = mean_absolute_error(self.y_test, self.y_pred)
        self.metrics['mse'] = mean_squared_error(self.y_test, self.y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        
        # Display metrics
        print("\n📊 PERFORMANCE METRICS")
        print(f"  R² Score:  {self.metrics['r2']:.4f} ({self.metrics['r2']*100:.2f}% variance explained)")
        print(f"  MAE:       ${self.metrics['mae']:,.2f}")
        print(f"  MSE:       ${self.metrics['mse']:,.2f}")
        print(f"  RMSE:      ${self.metrics['rmse']:,.2f}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION")
        if self.metrics['r2'] > 0.8:
            quality = "Excellent"
        elif self.metrics['r2'] > 0.6:
            quality = "Good"
        elif self.metrics['r2'] > 0.4:
            quality = "Moderate"
        else:
            quality = "Needs Improvement"
        
        print(f"  Model Quality: {quality}")
        print(f"  Average Prediction Error: ±${self.metrics['mae']:,.2f}")
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance from coefficients"""
        print(f"\n{SUBHEADER_LINE}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(SUBHEADER_LINE)
        
        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Coefficient': self.model.coef_,
            'Abs_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nFeature Coefficients (Impact on Revenue):")
        for _, row in self.feature_importance.iterrows():
            direction = "↑" if row['Coefficient'] > 0 else "↓"
            print(f"  {row['Feature']:20s}: ${row['Coefficient']:10,.2f} {direction}")
        
        print("\n💡 INSIGHTS")
        top_feature = self.feature_importance.iloc[0]
        print(f"  • Most Important Feature: {top_feature['Feature']}")
        print(f"  • Impact: ${abs(top_feature['Coefficient']):,.2f} per unit change")
        
        positive_features = self.feature_importance[self.feature_importance['Coefficient'] > 0]
        if len(positive_features) > 0:
            print(f"  • Positive Drivers: {', '.join(positive_features['Feature'].values)}")
        
        return self
    
    def predict_all_customers(self):
        """Generate predictions for all customers"""
        print(f"\n{HEADER_LINE}")
        print("GENERATING PREDICTIONS FOR ALL CUSTOMERS")
        print(HEADER_LINE)
        
        # Predict on all data
        X_all = self.data[self.feature_columns]
        predictions = self.model.predict(X_all)
        
        # Add predictions to dataframe
        self.data['Predicted_Revenue'] = predictions
        self.data['Actual_Revenue'] = self.data[self.target_column]
        
        # Calculate prediction error
        self.data['Prediction_Error'] = self.data['Predicted_Revenue'] - self.data['Actual_Revenue']
        self.data['Prediction_Error_Pct'] = (self.data['Prediction_Error'] / self.data['Actual_Revenue']) * 100
        
        print(f"\n✓ Predictions generated for {len(self.data):,} customers")
        print(f"\nPrediction Summary:")
        print(f"  Mean Predicted Revenue: ${self.data['Predicted_Revenue'].mean():,.2f}")
        print(f"  Mean Actual Revenue: ${self.data['Actual_Revenue'].mean():,.2f}")
        print(f"  Mean Absolute Error: ${np.abs(self.data['Prediction_Error']).mean():,.2f}")
        
        # Show examples
        print(f"\n{SUBHEADER_LINE}")
        print("Sample Predictions:")
        sample = self.data.sample(5)[['CustomerID', 'Actual_Revenue', 'Predicted_Revenue', 'Prediction_Error']]
        print(sample.to_string(index=False))
        
        return self
    
    def get_prediction_results(self):
        """Return dataframe with predictions"""
        return self.data
    
    def save_model(self, model_path):
        """Save trained model to disk"""
        joblib.dump(self.model, model_path)
        print(f"\n✓ Model saved to: {model_path}")
        return self
    
    def load_saved_model(self, model_path):
        """Load pre-trained model from disk"""
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        return self