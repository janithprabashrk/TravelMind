"""
Machine learning models and algorithms for TravelMind.

This module contains:
- Hotel recommendation engine with multiple algorithms
- Feature engineering pipeline
- Model training and evaluation
"""

from .recommender import HotelRecommendationEngine
from .trainer import ModelTrainer
from .feature_engine import FeatureEngineering

__all__ = ["HotelRecommendationEngine", "ModelTrainer", "FeatureEngineering"]
