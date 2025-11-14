from models.sentiment_analyzer import WikipediaSentimentAnalyzer
from models.price_predictor import BitcoinPricePredictor
import time

def update_wikipedia_data():
    """Update Wikipedia sentiment data with error handling"""
    print("=== Updating Wikipedia Data ===")
    try:
        analyzer = WikipediaSentimentAnalyzer()
        sentiment_data = analyzer.create_sentiment_file()
        print("✓ Wikipedia data update completed")
        return sentiment_data
    except Exception as e:
        print(f"✗ Wikipedia data update failed: {e}")
        return None

def update_bitcoin_model():
    """Update Bitcoin prediction model with error handling"""
    print("=== Updating Bitcoin Prediction Model ===")
    try:
        predictor = BitcoinPricePredictor()
        prediction = predictor.run_full_pipeline()
        print("✓ Bitcoin model update completed")
        return prediction
    except Exception as e:
        print(f"✗ Bitcoin model update failed: {e}")
        # Return safe default
        return {
            "prediction": "UP",
            "confidence": 50.0,
            "current_price": 0.0,
            "prediction_proba": {
                "up_probability": 50.0,
                "down_probability": 50.0
            }
        }

if __name__ == "__main__":
    print("Starting data update...")
    start_time = time.time()
    
    # Update Wikipedia data
    wiki_data = update_wikipedia_data()
    
    # Small delay to ensure file writing is complete
    time.sleep(2)
    
    # Update Bitcoin model
    result = update_bitcoin_model()
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    print(f"\n=== UPDATE COMPLETED IN {duration}s ===")
    print(f"Next day prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Current Price: ${result['current_price']:,.2f}" if result['current_price'] > 0 else "Current Price: $N/A")
    print(f"UP Probability: {result['prediction_proba']['up_probability']}%")
    print(f"DOWN Probability: {result['prediction_proba']['down_probability']}%")