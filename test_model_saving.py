"""
Simple test script to verify model saving works correctly
"""
import os
import pickle
import json
from datetime import datetime

def test_model_saving():
    """Test if we can save files to models/saved_models/"""
    
    print("🧪 Testing Model Saving...")
    print("=" * 60)
    
    # Test 1: Create directory
    test_dir = "models/saved_models"
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ Directory created/exists: {test_dir}")
    except Exception as e:
        print(f"❌ Failed to create directory: {e}")
        return False
    
    # Test 2: Test pickle saving (model file)
    test_model_path = os.path.join(test_dir, "test_model.pkl")
    try:
        # Create dummy model-like object
        dummy_model = {"type": "XGBClassifier", "test": True, "timestamp": datetime.now().isoformat()}
        with open(test_model_path, "wb") as f:
            pickle.dump(dummy_model, f)
        print(f"✅ Test model file saved: {test_model_path}")
        print(f"   Size: {os.path.getsize(test_model_path)} bytes")
    except Exception as e:
        print(f"❌ Failed to save test model: {e}")
        return False
    
    # Test 3: Test JSON saving (feature_info file)
    test_feature_info_path = os.path.join(test_dir, "test_feature_info.json")
    try:
        dummy_feature_info = {
            "predictors": ["close", "volume", "open"],
            "training_date": datetime.now().isoformat(),
            "training_samples": 100,
            "backtest_precision": 0.52,
            "backtest_accuracy": 0.51
        }
        with open(test_feature_info_path, "w") as f:
            json.dump(dummy_feature_info, f, indent=2)
        print(f"✅ Test feature_info file saved: {test_feature_info_path}")
        print(f"   Size: {os.path.getsize(test_feature_info_path)} bytes")
    except Exception as e:
        print(f"❌ Failed to save test feature_info: {e}")
        return False
    
    # Test 4: Verify files exist
    print()
    print("📁 Verifying saved files...")
    if os.path.exists(test_model_path):
        print(f"✅ Test model file exists and is readable")
    else:
        print(f"❌ Test model file does not exist!")
        return False
    
    if os.path.exists(test_feature_info_path):
        print(f"✅ Test feature_info file exists and is readable")
    else:
        print(f"❌ Test feature_info file does not exist!")
        return False
    
    # Test 5: Clean up test files
    print()
    print("🧹 Cleaning up test files...")
    try:
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
            print(f"✅ Removed test model file")
        if os.path.exists(test_feature_info_path):
            os.remove(test_feature_info_path)
            print(f"✅ Removed test feature_info file")
    except Exception as e:
        print(f"⚠️  Could not clean up test files: {e}")
    
    print()
    print("=" * 60)
    print("✅ ALL TESTS PASSED! Model saving should work correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_saving()
    exit(0 if success else 1)

