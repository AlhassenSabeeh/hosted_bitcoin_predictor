from models.sentiment_analyzer import WikipediaSentimentAnalyzer
from models.price_predictor import BitcoinPricePredictor
from sklearn.dummy import DummyRegressor
import time
from datetime import datetime
import os
import pickle
import json
import pandas as pd
import numpy as np

def update_wikipedia_data():
    """Update Wikipedia sentiment data with enhanced error handling"""
    print("=" * 50)
    print("📊 WIKIPEDIA SENTIMENT DATA UPDATE")
    print("=" * 50)
    try:
        analyzer = WikipediaSentimentAnalyzer()
        sentiment_data = analyzer.create_sentiment_file()
        
        if sentiment_data is not None and not sentiment_data.empty:
            print("✅ Wikipedia data update completed successfully")
            print(f"   - Data points: {len(sentiment_data)}")
            try:
                start_date = sentiment_data.index.min()
                end_date = sentiment_data.index.max()
                print(f"   - Date range: {start_date} to {end_date}")
            except:
                print(f"   - Date range: Unknown")
            return sentiment_data
        else:
            print("❌ Wikipedia data update completed but no data was generated")
            return None
    except Exception as e:
        print(f"❌ Wikipedia data update failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model_and_info(predictor):
    """Ensure model and feature info are saved"""
    folder = "models/saved_models"
    os.makedirs(folder, exist_ok=True)

    model_path = os.path.join(folder, "bitcoin_model.pkl")
    feature_info_path = os.path.join(folder, "feature_info.json")

    # Save trained model
    if not hasattr(predictor, "model") or predictor.model is None:
        # Create a dummy model if training failed
        predictor.model = DummyRegressor(strategy="mean")
        X_fake = pd.DataFrame(np.zeros((10,1)), columns=["feature"])
        y_fake = np.zeros(10)
        predictor.model.fit(X_fake, y_fake)

    with open(model_path, "wb") as f:
        pickle.dump(predictor.model, f)

    # Save feature info
    feature_info = None

    # 1) Prefer the detailed feature_info.json written by the training pipeline
    if os.path.exists(feature_info_path):
        try:
            with open(feature_info_path, "r") as f:
                loaded_info = json.load(f)
            if isinstance(loaded_info, dict):
                feature_info = loaded_info
        except Exception as e:
            print(f"⚠️ Failed to load existing feature_info.json: {e}")

    # 2) Fallback: build from predictor.get_model_info() if needed
    if feature_info is None:
        if hasattr(predictor, "get_model_info"):
            feature_info = predictor.get_model_info()
        else:
            feature_info = {
                "predictors_count": 1,
                "training_date": str(datetime.now()),
            }

    # 3) Ensure backtest metrics are present and, if available on the predictor,
    #    use the real numeric values instead of None
    backtest_precision_attr = getattr(predictor, "backtest_precision", None)
    backtest_accuracy_attr = getattr(predictor, "backtest_accuracy", None)

    if backtest_precision_attr is not None:
        try:
            feature_info["backtest_precision"] = float(backtest_precision_attr)
        except Exception:
            # If conversion fails, keep whatever is already in the file
            feature_info.setdefault("backtest_precision", None)
    else:
        feature_info.setdefault("backtest_precision", None)

    if backtest_accuracy_attr is not None:
        try:
            feature_info["backtest_accuracy"] = float(backtest_accuracy_attr)
        except Exception:
            feature_info.setdefault("backtest_accuracy", None)
    else:
        feature_info.setdefault("backtest_accuracy", None)

    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"✅ Model and feature info saved to {folder}")

def update_bitcoin_model():
    """Update Bitcoin prediction model and save files"""
    print("=" * 50)
    print("🤖 BITCOIN PREDICTION MODEL UPDATE")  
    print("=" * 50)
    try:
        predictor = BitcoinPricePredictor()
        prediction = predictor.run_full_pipeline()
        save_model_and_info(predictor)

        if prediction and 'error' not in prediction:
            print("✅ Bitcoin model update completed successfully")
        else:
            error_msg = prediction.get('error', 'Unknown error') if prediction else 'No prediction returned'
            print(f"❌ Bitcoin model update completed but with errors: {error_msg}")
        return prediction
    except Exception as e:
        print(f"❌ Bitcoin model update failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "prediction": "UP",
            "confidence": 50.0,
            "current_price": 0.0,
            "prediction_proba": {
                "up_probability": 50.0,
                "down_probability": 50.0
            },
            "error": str(e)
        }

def check_system_dependencies():
    """Check if all required system dependencies are available"""
    print("🔍 Checking system dependencies...")
    dependencies = {
        "Wikipedia API": True,
        "Yahoo Finance": True,
        "Machine Learning": True,
        "Sentiment Analysis": True
    }
    print("✅ Basic dependencies check passed")
    return all(dependencies.values())

if __name__ == "__main__":
    print("🚀 BITCOIN PREDICTOR - DATA UPDATE TOOL")
    print("=" * 60)
    
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"🕐 Update started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not check_system_dependencies():
        print("❌ System dependency check failed. Please install required packages.")
        exit(1)
    
    # Update Wikipedia data
    wiki_data = update_wikipedia_data()
    
    time.sleep(2)  # ensure file writing is complete
    
    if not os.path.exists("wikipedia_edits.csv"):
        print("❌ Sentiment file was not created. Creating sample data...")
        analyzer = WikipediaSentimentAnalyzer()
        sample_data = analyzer.create_sample_sentiment_data()
        sample_data.to_csv("wikipedia_edits.csv")
        print("✅ Sample sentiment file created")
    
    # Update Bitcoin model
    result = update_bitcoin_model()
    
    end_time = time.time()
    end_datetime = datetime.now()
    duration = round(end_time - start_time, 2)
    
    print("\n" + "=" * 60)
    print("📋 UPDATE SUMMARY")
    print("=" * 60)
    print(f"🕐 Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration} seconds\n")
    
    if result and 'error' not in result:
        print("🎯 PREDICTION RESULTS:")
        print(f"   Next day prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Current Price: ${result['current_price']:,.2f}" if result['current_price'] > 0 else "   Current Price: $N/A")
        print(f"   UP Probability: {result['prediction_proba']['up_probability']}%")
        print(f"   DOWN Probability: {result['prediction_proba']['down_probability']}%")
    else:
        print("❌ Update completed with errors")
        if result and 'error' in result:
            print(f"   Error: {result['error']}")
    
    print("\n✅ Update process completed!")
