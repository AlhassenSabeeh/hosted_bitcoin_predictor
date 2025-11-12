from models.sentiment_analyzer import WikipediaSentimentAnalyzer
from models.price_predictor import BitcoinPricePredictor

def update_wikipedia_data():
    """Update Wikipedia sentiment data"""
    analyzer = WikipediaSentimentAnalyzer()
    sentiment_data = analyzer.create_sentiment_file()
    return sentiment_data

def update_bitcoin_model():
    """Update Bitcoin prediction model"""
    predictor = BitcoinPricePredictor()
    prediction = predictor.run_full_pipeline()
    return prediction

if __name__ == "__main__":
    print("Starting data update...")
    
    # Update Wikipedia data
    print("=== Updating Wikipedia Data ===")
    update_wikipedia_data()
    
    # Update Bitcoin model
    print("\n=== Updating Bitcoin Prediction Model ===")
    result = update_bitcoin_model()
    
    print(f"\n=== PREDICTION RESULT ===")
    print(f"Next day prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Current Price: ${result['current_price']}")
    print("Update completed!")