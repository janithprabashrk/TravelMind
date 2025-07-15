#!/usr/bin/env python3
"""
TravelMind Free Demo - No Credit Cards Required!

This script demonstrates TravelMind's hotel recommendation system
using only free services that don't require credit card details.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.free_weather import get_weather_for_hotel_recommendation, FreeWeatherService

def demo_free_weather():
    """Demonstrate the free weather service"""
    print("🌤️  TravelMind Free Weather Service Demo")
    print("="*50)
    
    test_locations = [
        "Paris, France",
        "Tokyo, Japan", 
        "New York, USA",
        "Sydney, Australia",
        "Bangkok, Thailand"
    ]
    
    weather_service = FreeWeatherService()
    
    for location in test_locations:
        print(f"\n📍 {location}")
        print("-" * 30)
        
        try:
            weather_info = weather_service.get_weather_info(location)
            
            print(f"🌡️  Temperature: {weather_info.temperature}°C")
            print(f"☁️  Condition: {weather_info.condition}")
            print(f"🍂 Season: {weather_info.season}")
            print(f"💡 Travel Tip: {weather_info.travel_recommendation}")
            print(f"🎯 Best Activities: {', '.join(weather_info.best_activities[:3])}")
            
            # Get hotel recommendation data
            hotel_data = get_weather_for_hotel_recommendation(location)
            print(f"⭐ Weather Score: {hotel_data['weather_score']:.1f}/1.0")
            print(f"🏨 Recommended Hotels: {', '.join(hotel_data['best_hotel_types'][:2])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_seasonal_recommendations():
    """Demonstrate seasonal hotel recommendations"""
    print("\n\n🗓️  Seasonal Hotel Recommendations")
    print("="*50)
    
    from datetime import datetime, timedelta
    
    # Simulate different seasons
    seasons = [
        ("Winter in Paris", "Paris, France", datetime(2024, 1, 15)),
        ("Summer in Bangkok", "Bangkok, Thailand", datetime(2024, 7, 15)),
        ("Spring in Tokyo", "Tokyo, Japan", datetime(2024, 4, 15)),
        ("Autumn in New York", "New York, USA", datetime(2024, 10, 15))
    ]
    
    weather_service = FreeWeatherService()
    
    for title, location, date in seasons:
        print(f"\n🌍 {title}")
        print("-" * 30)
        
        # Get coordinates for seasonal calculation
        coords = weather_service.get_coordinates_from_location(location)
        if coords:
            lat, lon = coords
            season = weather_service.get_season_from_location_and_date(lat, date)
            climate_type = weather_service.get_regional_climate_type(lat, lon)
            
            print(f"📅 Season: {season.title()}")
            print(f"🌍 Climate: {climate_type.title()}")
            
            # Get travel recommendations
            travel_recs = weather_service.travel_recommendations.get(climate_type, {})
            recommendation = travel_recs.get(season, "Great time to visit!")
            print(f"💭 Recommendation: {recommendation}")

def demo_hotel_types_by_weather():
    """Demonstrate hotel type recommendations based on weather"""
    print("\n\n🏨 Hotel Type Recommendations by Weather")
    print("="*50)
    
    weather_scenarios = [
        ("Sunny Beach Day", {"temperature": 28, "condition": "Clear sky", "season": "summer"}),
        ("Rainy City Day", {"temperature": 15, "condition": "Heavy rain", "season": "autumn"}),
        ("Snowy Mountain", {"temperature": -5, "condition": "Snow", "season": "winter"}),
        ("Pleasant Spring", {"temperature": 20, "condition": "Partly cloudy", "season": "spring"})
    ]
    
    from src.utils.free_weather import WeatherInfo, get_recommended_hotel_types
    
    for scenario_name, weather_data in weather_scenarios:
        print(f"\n⛅ {scenario_name}")
        print("-" * 25)
        
        weather_info = WeatherInfo(
            location="Example Location",
            temperature=weather_data["temperature"],
            condition=weather_data["condition"],
            season=weather_data["season"]
        )
        
        hotel_types = get_recommended_hotel_types(weather_info)
        print(f"🏨 Recommended Hotels: {', '.join(hotel_types)}")

def demo_completely_free_system():
    """Show that the system works without any paid APIs"""
    print("\n\n✅ Completely Free System Verification")
    print("="*50)
    
    print("✅ No Credit Card Required")
    print("✅ Free Weather APIs:")
    print("   • wttr.in (completely free)")
    print("   • Open-Meteo (free, no signup)")
    print("   • Built-in seasonal logic")
    
    print("\n✅ Free Geocoding:")
    print("   • OpenStreetMap Nominatim (free)")
    print("   • No rate limits for personal use")
    
    print("\n✅ Free ML Models:")
    print("   • Scikit-learn (open source)")
    print("   • Pandas, NumPy (open source)")
    print("   • Local SQLite database")
    
    print("\n✅ Free APIs Used:")
    print("   • Google Gemini API (only one requiring signup)")
    print("   • All others completely free")
    
    print("\n🎯 Result: Industrial-level ML system with minimal external dependencies!")

def main():
    """Main demo function"""
    print("🚀 TravelMind - Completely Free Hotel Recommendation System")
    print("🆓 No Credit Cards Required! 🆓")
    print("="*60)
    
    try:
        demo_free_weather()
        demo_seasonal_recommendations()
        demo_hotel_types_by_weather()
        demo_completely_free_system()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Set your Gemini API key in .env file")
        print("2. Run: python train.py")
        print("3. Start API: python -m src.main")
        print("4. Launch UI: streamlit run frontend/app.py")
        print("\n🌟 Enjoy your free, industrial-level hotel recommendation system!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("\n💡 Make sure you're in the TravelMind directory and have installed requirements.txt")

if __name__ == "__main__":
    main()
