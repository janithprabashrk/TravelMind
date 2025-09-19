#!/usr/bin/env python3
"""
TravelMind Model Training Script

This script trains all ML models for the hotel recommendation system
using collected hotel data and free weather information.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_settings
from src.utils.free_weather import get_weather_for_hotel_recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Complete model training pipeline"""
    
    def __init__(self):
        self.settings = get_settings()
        self.models_dir = Path(self.settings.MODEL_PATH)
        self.data_dir = Path("./data")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Model trainer initialized")
        
    def create_sample_hotel_data(self) -> pd.DataFrame:
        """Create comprehensive sample hotel data for training"""
        logger.info("📊 Creating sample hotel data for training...")
        
        # Sample locations with diverse climates
        locations = [
            "Paris, France", "Tokyo, Japan", "New York, USA", "London, UK",
            "Bangkok, Thailand", "Sydney, Australia", "Dubai, UAE", "Singapore",
            "Barcelona, Spain", "Rome, Italy", "Amsterdam, Netherlands", "Berlin, Germany",
            "Los Angeles, USA", "Miami, USA", "Vancouver, Canada", "Toronto, Canada"
        ]
        
        sample_data = []
        
        for location in locations:
            # Get weather data for each location
            try:
                weather_data = get_weather_for_hotel_recommendation(location)
            except:
                weather_data = {
                    "temperature": 20, "season": "spring", "weather_score": 0.7,
                    "condition": "Pleasant", "recommended_activities": ["sightseeing"],
                    "best_hotel_types": ["city hotel"]
                }
            
            # Generate diverse hotels for each location
            hotel_types = ["budget", "mid-range", "luxury", "resort", "boutique"]
            
            for i, hotel_type in enumerate(hotel_types):
                for j in range(3):  # 3 hotels per type per location
                    hotel_id = len(sample_data) + 1
                    
                    # Base properties
                    base_rating = 3.0 + (i * 0.5) + (j * 0.1)
                    base_price = 50 + (i * 100) + (j * 20)
                    
                    # Seasonal adjustments
                    seasonal_multiplier = 1.0
                    if weather_data["season"] == "summer":
                        seasonal_multiplier = 1.2
                    elif weather_data["season"] == "winter":
                        seasonal_multiplier = 0.9
                    
                    # Weather-based adjustments
                    weather_bonus = weather_data["weather_score"] * 0.5
                    
                    hotel = {
                        # Basic Information
                        "hotel_id": hotel_id,
                        "name": f"{hotel_type.title()} Hotel {j+1}",
                        "location": location,
                        "address": f"{j+1} {hotel_type.title()} Street, {location}",
                        "property_type": hotel_type,
                        
                        # Ratings and Reviews
                        "rating": min(5.0, base_rating + weather_bonus),
                        "total_reviews": 100 + (i * 200) + (j * 50),
                        "review_rating": min(5.0, base_rating + weather_bonus + 0.1),
                        
                        # Pricing
                        "price_per_night": base_price * seasonal_multiplier,
                        "price_category": i + 1,  # 1-5 scale
                        
                        # Amenities (binary features)
                        "has_pool": 1 if hotel_type in ["luxury", "resort"] or j >= 1 else 0,
                        "has_spa": 1 if hotel_type in ["luxury", "resort"] else 0,
                        "has_wifi": 1,  # Almost all hotels have wifi
                        "has_restaurant": 1 if hotel_type != "budget" else 0,
                        "has_gym": 1 if hotel_type in ["mid-range", "luxury", "resort"] else 0,
                        "has_parking": 1 if j >= 1 else 0,
                        "has_ac": 1 if weather_data["temperature"] > 25 else 0,
                        "has_heating": 1 if weather_data["temperature"] < 15 else 0,
                        
                        # Property features
                        "is_resort": 1 if hotel_type == "resort" else 0,
                        "is_boutique": 1 if hotel_type == "boutique" else 0,
                        "is_business": 1 if hotel_type in ["mid-range", "luxury"] else 0,
                        "is_family_friendly": 1 if hotel_type in ["resort", "mid-range"] else 0,
                        
                        # Location features
                        "city_center_distance": 1.0 + (j * 2.0),
                        "airport_distance": 10.0 + (j * 5.0),
                        
                        # Weather features
                        "temperature": weather_data["temperature"],
                        "weather_score": weather_data["weather_score"],
                        "season": weather_data["season"],
                        
                        # Target variables for different models
                        "overall_score": min(5.0, base_rating + weather_bonus + (0.1 if j == 0 else 0)),
                        "value_score": max(1.0, 5.0 - (i * 0.8)),  # Budget hotels have higher value
                        "luxury_score": min(5.0, i + 1 + weather_bonus),
                        "family_score": 4.0 if hotel_type in ["resort", "mid-range"] else 2.5,
                        
                        # Additional features
                        "amenity_count": sum([
                            1 if hotel_type in ["luxury", "resort"] or j >= 1 else 0,  # pool
                            1 if hotel_type in ["luxury", "resort"] else 0,  # spa
                            1,  # wifi
                            1 if hotel_type != "budget" else 0,  # restaurant
                            1 if hotel_type in ["mid-range", "luxury", "resort"] else 0,  # gym
                        ]),
                        "collected_at": datetime.now().isoformat()
                    }
                    
                    sample_data.append(hotel)
        
        df = pd.DataFrame(sample_data)
        logger.info(f"✅ Created sample dataset with {len(df)} hotels across {len(locations)} locations")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for ML training"""
        logger.info("🔧 Preparing training data...")
        
        # Select features for training
        feature_columns = [
            'rating', 'price_per_night', 'total_reviews', 'amenity_count',
            'has_pool', 'has_spa', 'has_wifi', 'has_restaurant', 'has_gym',
            'is_resort', 'is_boutique', 'is_business', 'is_family_friendly',
            'temperature', 'weather_score', 'price_category'
        ]
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value
        
        X = df[feature_columns].fillna(0).values
        
        # Target variables
        y_overall = df['overall_score'].fillna(4.0).values if 'overall_score' in df.columns else df['rating'].fillna(4.0).values if 'rating' in df.columns else np.full(len(df), 4.0)
        y_value = df['value_score'].fillna(4.0).values if 'value_score' in df.columns else np.full(len(df), 4.0)
        y_luxury = df['luxury_score'].fillna(4.0).values if 'luxury_score' in df.columns else df['rating'].fillna(4.0).values if 'rating' in df.columns else np.full(len(df), 4.0)
        y_family = df['family_score'].fillna(4.0).values if 'family_score' in df.columns else np.full(len(df), 4.0)
        
        y_dict = {
            'overall': y_overall,
            'value': y_value,
            'luxury': y_luxury,
            'family': y_family
        }
        
        logger.info(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y_dict, feature_columns
    
    def train_simple_models(self, X: np.ndarray, y_dict: dict, feature_names: list):
        """Train simple but effective models"""
        logger.info("🤖 Training recommendation models...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            import joblib
            
            models = {}
            
            # Train different models for different recommendation types
            for model_name, y_target in y_dict.items():
                if model_name in ['overall', 'luxury', 'family']:
                    # Use Random Forest for complex patterns
                    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
                else:
                    # Use Ridge for simpler patterns
                    model = Ridge(alpha=1.0, random_state=42)
                
                model.fit(X, y_target)
                models[model_name] = model
                
                # Calculate simple score
                train_score = model.score(X, y_target)
                logger.info(f"✅ {model_name.title()} model trained (R² score: {train_score:.3f})")
            
            # Save individual model files with CORRECT NAMES that match the recommender expectations
            model_files = {
                'content_based_model.pkl': models['overall'],
                'collaborative_model.pkl': models['overall'],  # Use same for collaborative
                'value_based_model.pkl': models['value'],
                'luxury_model.pkl': models['luxury'],
                'family_model.pkl': models['family'],
                'hybrid_model.pkl': models['overall']  # Hybrid uses overall
            }
            
            for filename, model in model_files.items():
                model_path = self.models_dir / filename
                joblib.dump(model, model_path)
                logger.info(f"✅ Saved {filename}")
            
            # Save metadata
            metadata = {
                'feature_names': feature_names,
                'model_types': {name: type(model).__name__ for name, model in models.items()},
                'training_date': datetime.now().isoformat(),
                'num_samples': X.shape[0],
                'num_features': X.shape[1]
            }
            
            import json
            metadata_path = self.models_dir / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            return False
    
    def validate_models(self):
        """Validate that models can be loaded and used"""
        logger.info("🔍 Validating trained models...")
        
        try:
            import joblib
            
            # Check if model files exist
            expected_files = [
                'content_based_model.pkl', 'collaborative_model.pkl', 'value_based_model.pkl',
                'luxury_model.pkl', 'family_model.pkl', 'hybrid_model.pkl'
            ]
            
            missing_files = []
            for filename in expected_files:
                model_path = self.models_dir / filename
                if not model_path.exists():
                    missing_files.append(filename)
            
            if missing_files:
                logger.error(f"❌ Missing model files: {missing_files}")
                return False
            
            # Try loading and testing each model
            for filename in expected_files:
                model_path = self.models_dir / filename
                model = joblib.load(model_path)
                
                # Test prediction with dummy data
                dummy_features = np.array([[4.0, 150, 100, 5, 1, 0, 1, 1, 1, 0, 0, 1, 1, 22, 0.8, 3]])
                prediction = model.predict(dummy_features)
                
                logger.info(f"✅ {filename}: prediction = {prediction[0]:.2f}")
            
            logger.info("✅ All models validated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False

def main():
    """Main training function"""
    print("🚀 TravelMind Model Training")
    print("="*40)
    
    trainer = ModelTrainer()
    
    try:
        # Step 1: Create training data
        print("\n📊 Creating training data...")
        df = trainer.create_sample_hotel_data()
        
        # Save raw data
        data_file = trainer.data_dir / "training_data.csv"
        df.to_csv(data_file, index=False)
        print(f"✅ Training data saved to {data_file}")
        
        # Step 2: Prepare training data
        X, y_dict, feature_names = trainer.prepare_training_data(df)
        
        # Step 3: Train models
        success = trainer.train_simple_models(X, y_dict, feature_names)
        
        if success:
            # Step 4: Validate models
            if trainer.validate_models():
                print("\n🎉 Model training completed successfully!")
                print("\n📋 Next steps:")
                print("1. Start API: python -m src.main")
                print("2. Launch UI: streamlit run frontend/app.py")
                print("3. Test recommendations!")
                return True
            else:
                print("\n❌ Model validation failed.")
                return False
        else:
            print("\n❌ Model training failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
