import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for TravelMind application"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./travelmind.db")
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / "models"
    DATA_PATH = BASE_DIR / "data"
    
    # ML Configuration
    RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 100))
    TOP_K_RECOMMENDATIONS = int(os.getenv("TOP_K_RECOMMENDATIONS", 10))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    
    # Data Collection
    MAX_HOTELS_PER_LOCATION = int(os.getenv("MAX_HOTELS_PER_LOCATION", 50))
    DATA_COLLECTION_INTERVAL_HOURS = int(os.getenv("DATA_COLLECTION_INTERVAL_HOURS", 24))
    
    # Cache
    CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", 6))
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_PATH.mkdir(exist_ok=True, parents=True)
        cls.DATA_PATH.mkdir(exist_ok=True, parents=True)
        (cls.BASE_DIR / "logs").mkdir(exist_ok=True, parents=True)

# Create directories on import
Config.create_directories()
