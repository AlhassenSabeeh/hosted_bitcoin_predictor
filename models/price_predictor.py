import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import pickle
import os
from sklearn.metrics import precision_score, classification_report

class BitcoinPricePredictor:
    def __init__(self):
        self.model = None
        self.predictors = []
    
    def load_data(self):
        """Load and prepare Bitcoin price data"""
        print("Loading Bitcoin price data...")
        
        btc_ticker = yf.Ticker("BTC-USD")
        btc = btc_ticker.history(period='max')
        btc = btc.reset_index()
        btc['Date'] = btc['Date'].dt.tz_localize(None)
        
        # Remove unnecessary columns
        del btc["Dividends"]
        del btc["Stock Splits"]
        
        # Standardize column names
        btc.columns = [c.lower() for c in btc.columns]
        
        return btc
    
    def merge_with_sentiment(self, btc_data):
        """Merge price data with sentiment data"""
        print("Merging with sentiment data...")
        
        # Load sentiment data
        bit_sent = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
        
        # Prepare dates for merging
        btc_data['date'] = btc_data['date'].dt.normalize()
        
        # Merge datasets
        btc_data = btc_data.merge(bit_sent, left_on='date', right_index=True, how='left')
        btc_data[['sentiment', 'neg_sentiment', 'edit_count']] = btc_data[['sentiment', 'neg_sentiment', 'edit_count']].fillna(0)
        btc_data = btc_data.set_index('date')
        
        return btc_data
    
    def create_features(self, data):
        """Create technical features for prediction"""
        print("Creating features...")
        
        # Create target variable
        data["tomorrow"] = data["close"].shift(-1)
        data["target"] = (data["tomorrow"] > data["close"]).astype(int)
        data = data.dropna(subset=['target'])
        
        horizons = [2, 7, 60, 365]
        new_predictors = ["close", "sentiment", "neg_sentiment"]

        for horizon in horizons:
            rolling_averages = data.rolling(horizon, min_periods=1).mean()

            ratio_column = f"close_ratio_{horizon}"
            data[ratio_column] = data["close"] / rolling_averages["close"]

            edit_column = f"edit_{horizon}"
            data[edit_column] = rolling_averages["edit_count"]

            rolling = data.rolling(horizon, closed='left', min_periods=1).mean()
            trend_column = f"trend_{horizon}"
            data[trend_column] = rolling["target"]

            new_predictors.extend([ratio_column, trend_column, edit_column])

        self.predictors = new_predictors
        return data
    
    def train_model(self, data):
        """Train the prediction model"""
        print("Training model...")
        
        # Split data (use most recent data for testing)
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        
        # Train XGBoost model
        self.model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=100)
        self.model.fit(train_data[self.predictors], train_data["target"])
        
        # Save model
        os.makedirs('models/saved_models', exist_ok=True)
        with open('models/saved_models/bitcoin_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("Model training complete and saved.")
        return self.model
    
    def predict_next_day(self, data):
        """Predict next day price movement"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Get the latest data point
        latest_data = data.iloc[-1:].copy()
        
        # Ensure all predictors are present
        for pred in self.predictors:
            if pred not in latest_data.columns:
                latest_data[pred] = 0
        
        prediction = self.model.predict(latest_data[self.predictors])
        prediction_proba = self.model.predict_proba(latest_data[self.predictors])
        
        confidence = max(prediction_proba[0])
        
        result = {
            "prediction": "UP" if prediction[0] == 1 else "DOWN",
            "confidence": round(confidence * 100, 2),
            "current_price": round(latest_data['close'].iloc[0], 2),
            "prediction_proba": {
                "up_probability": round(prediction_proba[0][1] * 100, 2),
                "down_probability": round(prediction_proba[0][0] * 100, 2)
            }
        }
        
        return result
    
    def run_full_pipeline(self):
        """Run the complete prediction pipeline"""
        print("Starting Bitcoin prediction pipeline...")
        
        # Step 1: Load data
        btc_data = self.load_data()
        
        # Step 2: Merge with sentiment
        merged_data = self.merge_with_sentiment(btc_data)
        
        # Step 3: Create features
        enhanced_data = self.create_features(merged_data)
        
        # Step 4: Train model
        self.train_model(enhanced_data)
        
        # Step 5: Make prediction
        prediction = self.predict_next_day(enhanced_data)
        
        print("Prediction pipeline complete!")
        return prediction