"""
TravelMind - AI-Powered Hotel Recommendation System

This package provides an industrial-level machine learning solution for hotel recommendations
based on user preferences, location, season, budget, and ratings.

Main components:
- Data collection and preprocessing
- Multiple ML recommendation algorithms
- RESTful API with FastAPI
- Web interface with Streamlit
- Comprehensive testing and deployment tools
"""

__version__ = "1.0.0"
__author__ = "TravelMind Team"
__email__ = "contact@travelmind.ai"

from .config import Config

__all__ = ["Config"]
