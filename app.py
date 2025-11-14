from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

class BitcoinPredictor:
    def __init__(self):
        self.model = None
        self.btc_data = None
        self.last_update = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_paths = [
                'models/saved_models/bitcoin_model.pkl',
                'models/bitcoin_model.pkl'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Model loaded successfully from {model_path}")
                    return
            
            print("No pre-trained model found. Please run the update first.")
            self.model = None
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def get_current_data(self):
        """Get current Bitcoin data and prepare features"""
        try:
            if not os.path.exists("wikipedia_edits.csv"):
                print("Sentiment data not found. Please run update first.")
                return False
            
            sentiment_data = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            
            btc_ticker = yf.Ticker("BTC-USD")
            btc = btc_ticker.history(period="60d")
            btc = btc.reset_index()
            
            if 'Date' in btc.columns:
                btc['Date'] = btc['Date'].dt.tz_localize(None) if hasattr(btc['Date'].dt, 'tz_localize') else btc['Date']
            btc.columns = [c.lower() for c in btc.columns]
            
            btc['date'] = pd.to_datetime(btc['date']).dt.normalize()
            btc = btc.merge(sentiment_data, left_on='date', right_index=True, how='left')
            
            sentiment_cols = ['sentiment', 'neg_sentiment', 'edit_count']
            for col in sentiment_cols:
                if col in btc.columns:
                    btc[col] = btc[col].fillna(0)
                else:
                    btc[col] = 0
            
            btc = btc.set_index('date')
            btc = self.create_features(btc)
            
            self.btc_data = btc
            self.last_update = datetime.now()
            print("Current data loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error getting current data: {e}")
            return False
    
    def create_features(self, data):
        """Create technical features for prediction"""
        horizons = [2, 7, 60, 365]
        
        for horizon in horizons:
            rolling_close = data["close"].rolling(horizon, min_periods=1).mean()
            ratio_column = f"close_ratio_{horizon}"
            data[ratio_column] = data["close"] / rolling_close
            
            if 'edit_count' in data.columns:
                rolling_edits = data["edit_count"].rolling(horizon, min_periods=1).mean()
                edit_column = f"edit_{horizon}"
                data[edit_column] = rolling_edits
            else:
                data[f"edit_{horizon}"] = 0
            
            trend_column = f"trend_{horizon}"
            data[trend_column] = (data["close"] > data["close"].shift(1)).rolling(horizon, min_periods=1).mean().fillna(0)
        
        return data
    
    def predict_tomorrow(self):
        """Make prediction for tomorrow's price movement"""
        if self.model is None:
            return {"error": "Model not loaded. Please run update first."}
        
        if self.btc_data is None:
            success = self.get_current_data()
            if not success:
                return {"error": "Failed to load current data"}
        
        try:
            latest_data = self.btc_data.iloc[-1:].copy()
            
            predictors = [
                'close', 'sentiment', 'neg_sentiment', 'close_ratio_2', 
                'trend_2', 'edit_2', 'close_ratio_7', 'trend_7', 'edit_7', 
                'close_ratio_60', 'trend_60', 'edit_60', 'close_ratio_365', 
                'trend_365', 'edit_365'
            ]
            
            for pred in predictors:
                if pred not in latest_data.columns:
                    latest_data[pred] = 0
            
            if latest_data.empty:
                return {"error": "No data available for prediction"}
            
            prediction = self.model.predict(latest_data[predictors])
            prediction_proba = self.model.predict_proba(latest_data[predictors])
            
            confidence = float(max(prediction_proba[0]))
            current_price = float(latest_data['close'].iloc[0])
            
            result = {
                "prediction": "UP" if prediction[0] == 1 else "DOWN",
                "confidence": round(confidence * 100, 2),
                "current_price": round(current_price, 2),
                "last_updated": self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize predictor
predictor = BitcoinPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get prediction"""
    try:
        if predictor.last_update is None or (datetime.now() - predictor.last_update).seconds > 3600:
            print("Refreshing data...")
            predictor.get_current_data()
        
        result = predictor.predict_tomorrow()
        
        if 'error' not in result:
            result = {
                "prediction": str(result["prediction"]),
                "confidence": float(result["confidence"]),
                "current_price": float(result["current_price"]),
                "last_updated": str(result["last_updated"]),
                "prediction_date": str(result["prediction_date"])
            }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"})

@app.route('/update', methods=['POST'])
def update_model():
    """Force update of model and data"""
    try:
        print("Starting update...")
        
        # Simple refresh - just reload current data
        predictor.btc_data = None
        success = predictor.get_current_data()
        
        if success:
            return jsonify({"status": "success", "message": "Data refreshed successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to refresh data"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/status')
def status():
    """Get current system status"""
    status_info = {
        "model_loaded": bool(predictor.model is not None),
        "data_loaded": bool(predictor.btc_data is not None),
        "last_update": predictor.last_update.strftime("%Y-%m-%d %H:%M:%S") if predictor.last_update else "Never",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(status_info)

if __name__ == '__main__':
    try:
        predictor.get_current_data()
    except Exception as e:
        print(f"Warning: Could not load data on startup: {e}")
    
    print("Bitcoin Predictor starting...")
    print("Visit http://localhost:5000 to use the application")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))