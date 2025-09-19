#!/usr/bin/env python3
"""
Test script to verify that the "models not trained" error is fixed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_models_exist():
    """Check if all required model files exist"""
    print("🔍 Checking if all model files exist...")
    
    models_dir = Path("models")
    required_models = [
        'content_based_model.pkl',
        'collaborative_model.pkl', 
        'value_based_model.pkl',
        'luxury_model.pkl',
        'family_model.pkl',
        'hybrid_model.pkl'
    ]
    
    missing = []
    for model_name in required_models:
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✅ {model_name} exists")
        else:
            print(f"❌ {model_name} missing")
            missing.append(model_name)
    
    return len(missing) == 0

def test_recommender_initialization():
    """Test if the recommender can initialize without errors"""
    print("\n🧪 Testing recommender initialization...")
    
    try:
        from src.models.recommender import HotelRecommendationSystem
        print("✅ Import successful")
        
        recommender = HotelRecommendationSystem()
        print("✅ Recommender initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommender initialization failed: {e}")
        if "models are not trained" in str(e).lower():
            print("   This is the error we're trying to fix!")
        return False

def test_api_import():
    """Test if the main API can be imported"""
    print("\n🚀 Testing API import...")
    
    try:
        from src.main import app
        print("✅ API imported successfully!")
        return True
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False

def main():
    print("🧪 TravelMind Model Error Fix Verification")
    print("=" * 50)
    
    # Test 1: Check model files
    models_ok = test_models_exist()
    
    # Test 2: Test recommender
    recommender_ok = test_recommender_initialization()
    
    # Test 3: Test API
    api_ok = test_api_import()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"📁 Model Files: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"🤖 Recommender: {'✅ PASS' if recommender_ok else '❌ FAIL'}")
    print(f"🚀 API Import: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if models_ok and recommender_ok and api_ok:
        print("\n🎉 SUCCESS! The 'models not trained' error is FIXED!")
        print("\n📋 You can now run:")
        print("   python -m src.main          # Start API server")
        print("   streamlit run frontend/app.py  # Start web interface")
        return True
    else:
        print("\n❌ Some tests failed. The error may still exist.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
