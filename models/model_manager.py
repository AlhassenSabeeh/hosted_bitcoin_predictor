import pickle
import pandas as pd
import os

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_path = 'models/saved_models/bitcoin_model.pkl'
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully")
                return True
            else:
                print("No trained model found")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def model_exists(self):
        """Check if model file exists"""
        return os.path.exists(self.model_path)
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            if not self.load_model():
                return {"status": "no_model"}
        
        info = {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "features_used": getattr(self.model, 'feature_names_in_', []).tolist() if hasattr(self.model, 'feature_names_in_') else []
        }
        
        return info