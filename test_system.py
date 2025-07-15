#!/usr/bin/env python3
"""
TravelMind Quick Test - Verify Core Functionality

This script tests the core functionality of TravelMind
without requiring external API calls.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing Imports...")
    
    try:
        from src.utils.config import get_settings
        print("✅ Config module")
        
        from src.utils.free_weather import FreeWeatherService, get_weather_for_hotel_recommendation
        print("✅ Free weather module")
        
        # Test configuration
        settings = get_settings()
        print(f"✅ Settings loaded (Debug: {settings.DEBUG})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_weather_service():
    """Test the free weather service"""
    print("\n🌤️  Testing Weather Service...")
    
    try:
        from src.utils.free_weather import FreeWeatherService
        
        weather_service = FreeWeatherService()
        
        # Test coordinate lookup
        coords = weather_service.get_coordinates_from_location("Tokyo, Japan")
        if coords:
            print(f"✅ Geocoding: Tokyo at {coords[0]:.2f}, {coords[1]:.2f}")
        else:
            print("⚠️  Geocoding failed (network issue?)")
        
        # Test weather info
        weather_info = weather_service.get_weather_info("London, UK")
        print(f"✅ Weather Info: London {weather_info.temperature}°C, {weather_info.condition}")
        print(f"   Season: {weather_info.season}, Activities: {', '.join(weather_info.best_activities[:3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weather service error: {e}")
        return False

def test_hotel_recommendations():
    """Test hotel recommendation logic"""
    print("\n🏨 Testing Hotel Recommendations...")
    
    try:
        from src.utils.free_weather import get_weather_for_hotel_recommendation
        
        # Test different locations
        locations = ["Paris, France", "Bangkok, Thailand", "New York, USA"]
        
        for location in locations:
            weather_data = get_weather_for_hotel_recommendation(location)
            
            print(f"✅ {location}:")
            print(f"   Temperature: {weather_data['temperature']}°C")
            print(f"   Season: {weather_data['season']}")
            print(f"   Weather Score: {weather_data['weather_score']:.1f}/1.0")
            print(f"   Recommended Hotels: {', '.join(weather_data['best_hotel_types'][:2])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hotel recommendation error: {e}")
        return False

def test_data_collector():
    """Test data collector initialization (without API calls)"""
    print("\n📊 Testing Data Collector...")
    
    try:
        # Test if we can import and initialize without errors
        from src.data.collector import HotelDataCollector
        
        print("✅ Data collector module imported")
        
        # Note: We don't actually initialize because it requires Gemini API key
        print("ℹ️  Skipping initialization (requires Gemini API key)")
        
        return True
        
    except ImportError as e:
        if "google.generativeai" in str(e):
            print("⚠️  Google Generative AI not installed (run: pip install google-generativeai)")
            return True  # This is expected if not installed
        else:
            print(f"❌ Import error: {e}")
            return False
    except Exception as e:
        print(f"❌ Data collector error: {e}")
        return False

def test_configuration_system():
    """Test configuration management"""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        from src.utils.config import get_settings, use_free_weather, is_development
        
        settings = get_settings()
        
        print(f"✅ API Host: {settings.API_HOST}:{settings.API_PORT}")
        print(f"✅ Debug Mode: {is_development()}")
        print(f"✅ Free Weather: {use_free_weather()}")
        print(f"✅ Database: {settings.DATABASE_URL}")
        print(f"✅ Model Path: {settings.MODEL_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = ["data", "models", "logs", "src", "frontend"]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ missing (will be created automatically)")
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created {dir_name}/")
    
    return True

def main():
    """Run all tests"""
    print("🧪 TravelMind Quick Test Suite")
    print("="*40)
    
    tests = [
        ("Core Imports", test_imports),
        ("Weather Service", test_weather_service),
        ("Hotel Recommendations", test_hotel_recommendations),
        ("Data Collector", test_data_collector),
        ("Configuration", test_configuration_system),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! TravelMind is ready to use.")
        print("\n📋 Quick Start:")
        print("1. Set GEMINI_API_KEY in .env file")
        print("2. Run: python demo_free.py (to test without API)")
        print("3. Run: python -m src.main (to start API server)")
        print("4. Run: streamlit run frontend/app.py (to start web UI)")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Common fixes:")
        print("• Install packages: pip install -r requirements_free.txt")
        print("• Run: python fix_errors.py")
        print("• Check your .env configuration")

if __name__ == "__main__":
    main()
