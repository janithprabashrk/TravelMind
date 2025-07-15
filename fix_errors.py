#!/usr/bin/env python3
"""
TravelMind Error Checker and Fixer

This script checks for common errors in the TravelMind project
and provides fixes or suggestions.
"""

import sys
import os
import importlib
from pathlib import Path

def check_imports():
    """Check if all required packages are available"""
    print("🔍 Checking Python Packages...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'fastapi', 'uvicorn',
        'streamlit', 'requests', 'pydantic'
    ]
    
    optional_packages = [
        'google.generativeai'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} - REQUIRED")
    
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} - OPTIONAL")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Run: pip install -r requirements_free.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Run: pip install google-generativeai")
    
    return True

def check_file_structure():
    """Check if all required files exist"""
    print("\n🔍 Checking Project Structure...")
    
    required_files = [
        "src/__init__.py",
        "src/main.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/free_weather.py",
        "src/data/__init__.py",
        "src/data/collector.py",
        "src/models/__init__.py",
        "src/api/__init__.py",
        "frontend/app.py",
        ".env",
        "requirements_free.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_configuration():
    """Check configuration settings"""
    print("\n🔍 Checking Configuration...")
    
    try:
        from src.utils.config import get_settings
        settings = get_settings()
        
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            print("✅ Gemini API key configured")
        else:
            print("⚠️  Gemini API key not configured")
            print("   Edit .env file and set GEMINI_API_KEY")
        
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        print(f"✅ API Host: {settings.API_HOST}:{settings.API_PORT}")
        print(f"✅ Debug mode: {settings.DEBUG}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_weather_service():
    """Test the free weather service"""
    print("\n🔍 Testing Free Weather Service...")
    
    try:
        from src.utils.free_weather import get_weather_for_hotel_recommendation
        
        # Test with a simple location
        weather_data = get_weather_for_hotel_recommendation("Paris, France")
        
        if weather_data and 'temperature' in weather_data:
            print("✅ Free weather service working")
            print(f"   Sample: Paris temperature: {weather_data['temperature']}°C")
            return True
        else:
            print("⚠️  Weather service returned incomplete data")
            return False
            
    except Exception as e:
        print(f"❌ Weather service error: {e}")
        return False

def create_missing_directories():
    """Create missing directories"""
    print("\n🔍 Creating Missing Directories...")
    
    required_dirs = [
        "data", "models", "logs", "src", "src/utils", 
        "src/data", "src/models", "src/api", "frontend", "tests"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✅ {dir_path}/")

def fix_common_issues():
    """Fix common issues automatically"""
    print("\n🔧 Fixing Common Issues...")
    
    # Create __init__.py files if missing
    init_files = [
        "src/__init__.py",
        "src/utils/__init__.py", 
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/api/__init__.py"
    ]
    
    for init_file in init_files:
        if not Path(init_file).exists():
            with open(init_file, 'w') as f:
                f.write('"""TravelMind package initialization"""\n')
            print(f"✅ Created {init_file}")
    
    # Create .env file if missing
    if not Path(".env").exists():
        env_content = """# TravelMind Environment Variables
GEMINI_API_KEY=your_gemini_api_key_here

# Weather Configuration (Free)
USE_WEATHER_API=false
SEASONAL_RECOMMENDATIONS=true

# Database Configuration
DATABASE_URL=sqlite:///./travelmind.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# ML Model Configuration
MODEL_PATH=./models/
RETRAIN_THRESHOLD=100

# Data Collection
MAX_HOTELS_PER_LOCATION=50
DATA_COLLECTION_INTERVAL_HOURS=24

# Recommendation Settings
TOP_K_RECOMMENDATIONS=10
SIMILARITY_THRESHOLD=0.7

# Cache Settings
CACHE_EXPIRY_HOURS=6
WEATHER_CACHE_DURATION=3600
"""
        with open(".env", 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔍 Testing Basic Functionality...")
    
    try:
        # Test configuration loading
        from src.utils.config import get_settings
        settings = get_settings()
        print("✅ Configuration loading")
        
        # Test free weather service
        from src.utils.free_weather import FreeWeatherService
        weather_service = FreeWeatherService()
        print("✅ Weather service initialization")
        
        # Test coordinate lookup
        coords = weather_service.get_coordinates_from_location("Paris, France")
        if coords:
            print("✅ Coordinate lookup")
        else:
            print("⚠️  Coordinate lookup failed (but this is non-critical)")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main error checking and fixing function"""
    print("🔧 TravelMind Error Checker and Fixer")
    print("="*50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    issues_found = 0
    
    # Create missing directories first
    create_missing_directories()
    
    # Fix common issues
    fix_common_issues()
    
    # Check imports
    if not check_imports():
        issues_found += 1
    
    # Check file structure
    if not check_file_structure():
        issues_found += 1
    
    # Check configuration
    if not check_configuration():
        issues_found += 1
    
    # Test weather service
    if not check_weather_service():
        issues_found += 1
    
    # Test basic functionality
    if not test_basic_functionality():
        issues_found += 1
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    if issues_found == 0:
        print("🎉 No issues found! Your TravelMind project is ready to run.")
        print("\n📋 Next Steps:")
        print("1. Set your Gemini API key in .env file")
        print("2. Run: python train.py")
        print("3. Start API: python -m src.main")
        print("4. Launch UI: streamlit run frontend/app.py")
    else:
        print(f"⚠️  Found {issues_found} issue(s). Please address them before running.")
        print("\n🔧 Common Solutions:")
        print("• Install packages: pip install -r requirements_free.txt")
        print("• Set API key in .env file")
        print("• Check file permissions")
        print("• Ensure you're in the TravelMind directory")

if __name__ == "__main__":
    main()
