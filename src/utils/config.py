"""
Configuration management for TravelMind
Handles environment variables and application settings
"""

import os
from typing import Optional
from dataclasses import dataclass

"""
Configuration management for TravelMind
Handles environment variables and application settings
"""

import os
from typing import Optional

class Settings:
    """Application settings from environment variables"""
    
    def __init__(self):
        # Load from environment variables
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        # Weather Configuration - Now completely free!
        self.USE_WEATHER_API = os.getenv("USE_WEATHER_API", "false").lower() == "true"
        self.SEASONAL_RECOMMENDATIONS = os.getenv("SEASONAL_RECOMMENDATIONS", "true").lower() == "true"
        self.FREE_WEATHER_SERVICE = os.getenv("FREE_WEATHER_SERVICE", "true").lower() == "true"
        
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./travelmind.db")
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
        
        # ML Model Configuration
        self.MODEL_PATH = os.getenv("MODEL_PATH", "./models/")
        self.RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "100"))
        
        # Data Collection
        self.MAX_HOTELS_PER_LOCATION = int(os.getenv("MAX_HOTELS_PER_LOCATION", "50"))
        self.DATA_COLLECTION_INTERVAL_HOURS = int(os.getenv("DATA_COLLECTION_INTERVAL_HOURS", "24"))
        
        # Recommendation Settings
        self.TOP_K_RECOMMENDATIONS = int(os.getenv("TOP_K_RECOMMENDATIONS", "10"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        
        # Cache Settings
        self.CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "6"))
        
        # Free Services Configuration
        self.USE_FREE_GEOCODING = True  # Use OpenStreetMap Nominatim (free)
        self.USE_FREE_WEATHER = True    # Use wttr.in and Open-Meteo (free)
        self.WEATHER_CACHE_DURATION = int(os.getenv("WEATHER_CACHE_DURATION", "3600"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "./logs/travelmind.log")
        
        # Validate required settings
        self._validate()
    
    def _validate(self):
        """Validate required settings"""
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your_gemini_api_key_here":
            print("⚠️  Warning: GEMINI_API_KEY not set. Some features may not work.")

# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings

def is_development() -> bool:
    """Check if running in development mode"""
    return settings.DEBUG

def get_database_url() -> str:
    """Get database URL"""
    return settings.DATABASE_URL

def get_model_path() -> str:
    """Get model storage path"""
    return settings.MODEL_PATH

def use_free_weather() -> bool:
    """Check if using free weather service"""
    return settings.FREE_WEATHER_SERVICE
