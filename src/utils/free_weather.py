"""
Free Weather Service for TravelMind

This module provides weather information without requiring paid APIs or credit cards.
Uses multiple free sources and built-in seasonal logic.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class WeatherInfo:
    """Weather information data class"""
    location: str
    temperature: float
    condition: str
    humidity: Optional[int] = None
    season: str = ""
    travel_recommendation: str = ""
    best_activities: List[str] = field(default_factory=list)

class FreeWeatherService:
    """
    Completely free weather service using multiple free sources:
    1. Built-in seasonal logic based on coordinates and date
    2. Free weather APIs that don't require credit cards
    3. Geographic and climatic patterns
    """
    
    def __init__(self):
        # Free weather APIs (no credit card required)
        self.free_apis = {
            "wttr": "https://wttr.in/{location}?format=j1",  # Completely free
            "openmeteo": "https://api.open-meteo.com/v1/forecast",  # Free, no signup
        }
        
        # Seasonal patterns by hemisphere and region
        self.seasonal_patterns = {
            "northern": {
                "winter": {"months": [12, 1, 2], "temp_range": (-10, 10)},
                "spring": {"months": [3, 4, 5], "temp_range": (5, 20)},
                "summer": {"months": [6, 7, 8], "temp_range": (15, 35)},
                "autumn": {"months": [9, 10, 11], "temp_range": (5, 25)}
            },
            "southern": {
                "summer": {"months": [12, 1, 2], "temp_range": (15, 35)},
                "autumn": {"months": [3, 4, 5], "temp_range": (5, 25)},
                "winter": {"months": [6, 7, 8], "temp_range": (-10, 10)},
                "spring": {"months": [9, 10, 11], "temp_range": (5, 20)}
            }
        }
        
        # Travel recommendations by season and region type
        self.travel_recommendations = {
            "tropical": {
                "dry_season": "Perfect time for beach resorts and outdoor activities",
                "wet_season": "Great for spa resorts and indoor cultural experiences"
            },
            "temperate": {
                "spring": "Ideal for city tours and cultural sites",
                "summer": "Perfect for outdoor activities and beach destinations",
                "autumn": "Excellent for scenic tours and wine regions",
                "winter": "Great for skiing and cozy mountain resorts"
            },
            "mediterranean": {
                "spring": "Perfect weather for sightseeing and outdoor dining",
                "summer": "Ideal for beach resorts and island hopping",
                "autumn": "Great for cultural tours and moderate activities",
                "winter": "Good for city breaks and indoor attractions"
            }
        }
    
    def get_weather_free_wttr(self, location: str) -> Optional[Dict]:
        """Get weather from wttr.in (completely free, no signup)"""
        try:
            url = self.free_apis["wttr"].format(location=location.replace(" ", "+"))
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_condition", [{}])[0]
                
                return {
                    "temperature": float(current.get("temp_C", 20)),
                    "condition": current.get("weatherDesc", [{}])[0].get("value", "Clear"),
                    "humidity": int(current.get("humidity", 50)),
                    "source": "wttr.in"
                }
        except Exception as e:
            logger.warning(f"wttr.in API failed for {location}: {e}")
        return None
    
    def get_weather_open_meteo(self, latitude: float, longitude: float) -> Optional[Dict]:
        """Get weather from Open-Meteo (free, no signup required)"""
        try:
            url = self.free_apis["openmeteo"]
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
                "temperature_unit": "celsius"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_weather", {})
                
                # Convert weather code to description
                weather_code = current.get("weathercode", 0)
                condition = self.weather_code_to_description(weather_code)
                
                return {
                    "temperature": current.get("temperature", 20),
                    "condition": condition,
                    "humidity": 50,  # Default value
                    "source": "open-meteo.com"
                }
        except Exception as e:
            logger.warning(f"Open-Meteo API failed for {latitude}, {longitude}: {e}")
        return None
    
    def weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to description"""
        codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
            67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        return codes.get(code, "Unknown")
    
    def get_season_from_location_and_date(self, latitude: float, date: Optional[datetime] = None) -> str:
        """Determine season based on latitude and current date"""
        if date is None:
            date = datetime.now()
        
        month = date.month
        hemisphere = "northern" if latitude >= 0 else "southern"
        
        for season, info in self.seasonal_patterns[hemisphere].items():
            if month in info["months"]:
                return season
        
        return "spring"  # Default
    
    def get_regional_climate_type(self, latitude: float, longitude: float) -> str:
        """Determine climate type based on coordinates"""
        abs_lat = abs(latitude)
        
        if abs_lat <= 23.5:  # Tropics
            return "tropical"
        elif abs_lat <= 40:  # Subtropical/Mediterranean
            return "mediterranean"
        else:  # Temperate
            return "temperate"
    
    def get_coordinates_from_location(self, location: str) -> Optional[tuple]:
        """Get approximate coordinates from location name (free service)"""
        try:
            # Use Nominatim (OpenStreetMap) - completely free
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    return (lat, lon)
        except Exception as e:
            logger.warning(f"Geocoding failed for {location}: {e}")
        
        return None
    
    def get_weather_info(self, location: str) -> WeatherInfo:
        """
        Get comprehensive weather information using only free sources
        """
        # Try to get coordinates
        coordinates = self.get_coordinates_from_location(location)
        
        # Try different free weather sources
        weather_data = None
        
        # Try wttr.in first (most reliable free source)
        weather_data = self.get_weather_free_wttr(location)
        
        # If that fails and we have coordinates, try Open-Meteo
        if not weather_data and coordinates:
            lat, lon = coordinates
            weather_data = self.get_weather_open_meteo(lat, lon)
        
        # If all APIs fail, use intelligent defaults based on location and season
        if not weather_data:
            if coordinates:
                lat, lon = coordinates
                season = self.get_season_from_location_and_date(lat)
                hemisphere = "northern" if lat >= 0 else "southern"
                
                # Estimate temperature based on season and latitude
                temp_range = self.seasonal_patterns[hemisphere][season]["temp_range"]
                avg_temp = sum(temp_range) / 2
                
                weather_data = {
                    "temperature": avg_temp,
                    "condition": "Clear",
                    "humidity": 50,
                    "source": "estimated"
                }
            else:
                # Ultimate fallback
                weather_data = {
                    "temperature": 22,
                    "condition": "Pleasant",
                    "humidity": 55,
                    "source": "default"
                }
        
        # Determine season and travel recommendations
        if coordinates:
            lat, lon = coordinates
            season = self.get_season_from_location_and_date(lat)
            climate_type = self.get_regional_climate_type(lat, lon)
            
            travel_rec = self.travel_recommendations.get(climate_type, {}).get(season, 
                "Great time to visit with pleasant weather")
            
            # Best activities based on weather and season
            activities = self.get_recommended_activities(weather_data["condition"], season, climate_type)
        else:
            season = "spring"
            travel_rec = "Good time for travel"
            activities = ["sightseeing", "dining", "cultural tours"]
        
        return WeatherInfo(
            location=location,
            temperature=weather_data["temperature"],
            condition=weather_data["condition"],
            humidity=weather_data.get("humidity"),
            season=season,
            travel_recommendation=travel_rec,
            best_activities=activities
        )
    
    def get_recommended_activities(self, condition: str, season: str, climate_type: str) -> List[str]:
        """Get recommended activities based on weather conditions"""
        activities = []
        
        condition_lower = condition.lower()
        
        if any(word in condition_lower for word in ["clear", "sunny", "pleasant"]):
            activities.extend(["outdoor dining", "sightseeing", "walking tours", "beach activities"])
        elif any(word in condition_lower for word in ["cloudy", "partly"]):
            activities.extend(["city tours", "museums", "shopping", "cultural sites"])
        elif any(word in condition_lower for word in ["rain", "shower"]):
            activities.extend(["museums", "spa treatments", "indoor attractions", "shopping malls"])
        elif any(word in condition_lower for word in ["snow", "cold"]):
            activities.extend(["winter sports", "cozy cafes", "indoor entertainment", "thermal baths"])
        
        # Add seasonal activities
        if season == "summer" and climate_type in ["tropical", "mediterranean"]:
            activities.extend(["swimming", "water sports", "island hopping"])
        elif season == "winter" and climate_type == "temperate":
            activities.extend(["skiing", "hot springs", "winter festivals"])
        
        return list(set(activities))[:6]  # Return unique activities, max 6

# Global instance
free_weather = FreeWeatherService()

def get_weather_for_hotel_recommendation(location: str) -> Dict:
    """
    Get weather information specifically for hotel recommendations
    This function is used by the hotel recommendation system
    """
    weather_info = free_weather.get_weather_info(location)
    
    return {
        "temperature": weather_info.temperature,
        "condition": weather_info.condition,
        "season": weather_info.season,
        "travel_recommendation": weather_info.travel_recommendation,
        "recommended_activities": weather_info.best_activities,
        "weather_score": calculate_weather_score(weather_info),
        "best_hotel_types": get_recommended_hotel_types(weather_info)
    }

def calculate_weather_score(weather_info: WeatherInfo) -> float:
    """Calculate a weather score for hotel recommendations (0-1)"""
    score = 0.5  # Base score
    
    # Temperature score (comfort zone: 18-28°C)
    temp = weather_info.temperature
    if 18 <= temp <= 28:
        score += 0.3
    elif 15 <= temp <= 32:
        score += 0.2
    elif 10 <= temp <= 35:
        score += 0.1
    
    # Condition score
    condition_lower = weather_info.condition.lower()
    if any(word in condition_lower for word in ["clear", "sunny", "pleasant"]):
        score += 0.2
    elif any(word in condition_lower for word in ["partly", "cloudy"]):
        score += 0.1
    elif any(word in condition_lower for word in ["rain", "storm"]):
        score -= 0.1
    
    return max(0.0, min(1.0, score))

def get_recommended_hotel_types(weather_info: WeatherInfo) -> List[str]:
    """Get recommended hotel types based on weather"""
    hotel_types = []
    
    temp = weather_info.temperature
    condition_lower = weather_info.condition.lower()
    season = weather_info.season
    
    # Temperature-based recommendations
    if temp >= 25:
        hotel_types.extend(["beach resort", "resort with pool", "air-conditioned hotel"])
    elif temp <= 10:
        hotel_types.extend(["cozy hotel", "hotel with heating", "mountain lodge"])
    else:
        hotel_types.extend(["city hotel", "boutique hotel"])
    
    # Weather condition recommendations
    if any(word in condition_lower for word in ["rain", "storm"]):
        hotel_types.extend(["hotel with spa", "indoor entertainment", "covered parking"])
    elif any(word in condition_lower for word in ["clear", "sunny"]):
        hotel_types.extend(["outdoor terrace", "rooftop restaurant", "garden hotel"])
    
    # Seasonal recommendations
    if season == "summer":
        hotel_types.extend(["beachfront", "pool hotel", "resort"])
    elif season == "winter":
        hotel_types.extend(["ski hotel", "thermal spa", "fireplace lounge"])
    
    return list(set(hotel_types))[:5]  # Return unique types, max 5
