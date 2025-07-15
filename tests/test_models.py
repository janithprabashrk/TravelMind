import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from src.models.recommender import HotelRecommendationEngine
from src.data.preprocessor import HotelDataPreprocessor
from src.models.feature_engine import FeatureEngineering

class TestHotelRecommendationEngine:
    """Test cases for hotel recommendation engine"""
    
    def setup_method(self):
        """Setup test data"""
        self.engine = HotelRecommendationEngine()
        
        # Create sample training data
        self.X_sample = np.random.random((50, 12))
        self.y_sample = {
            'overall_score': np.random.random(50) * 5,
            'value_score': np.random.random(50) * 5,
            'luxury_score': np.random.random(50) * 5,
            'family_score': np.random.random(50) * 5
        }
        self.feature_names = [f'feature_{i}' for i in range(12)]
        
        # Sample hotels for prediction
        self.sample_hotels = [
            {
                'name': 'Test Hotel 1',
                'location': 'paris, france',
                'star_rating': 4.0,
                'user_rating': 4.2,
                'min_price': 150,
                'max_price': 250,
                'avg_price': 200,
                'amenities': ['wifi', 'pool', 'gym'],
                'family_friendly': True,
                'pet_friendly': False
            },
            {
                'name': 'Test Hotel 2',
                'location': 'tokyo, japan',
                'star_rating': 3.5,
                'user_rating': 3.8,
                'min_price': 100,
                'max_price': 180,
                'avg_price': 140,
                'amenities': ['wifi', 'restaurant'],
                'family_friendly': False,
                'pet_friendly': True
            }
        ]
    
    def test_model_training(self):
        """Test model training"""
        self.engine.train_models(self.X_sample, self.y_sample, self.feature_names)
        assert self.engine.is_trained
        assert hasattr(self.engine, 'feature_names')
    
    def test_recommendations(self):
        """Test recommendation generation"""
        # Train models first
        self.engine.train_models(self.X_sample, self.y_sample, self.feature_names)
        
        user_preferences = {
            'budget_min': 100,
            'budget_max': 300,
            'min_rating': 3.0,
            'family_travel': True,
            'preferred_amenities': ['wifi', 'pool']
        }
        
        recommendations = self.engine.predict_recommendations(
            user_preferences, self.sample_hotels, 'hybrid', 2
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2
        
        if recommendations:
            rec = recommendations[0]
            assert 'recommendation_score' in rec
            assert 'recommendation_rank' in rec
            assert 'recommendation_type' in rec
    
    def test_model_persistence(self, tmp_path):
        """Test model saving and loading"""
        # Train models
        self.engine.train_models(self.X_sample, self.y_sample, self.feature_names)
        
        # Save models
        model_path = tmp_path / "test_models.pkl"
        self.engine.save_models(str(model_path))
        assert model_path.exists()
        
        # Load models in new instance
        new_engine = HotelRecommendationEngine()
        new_engine.load_models(str(model_path))
        assert new_engine.is_trained

class TestHotelDataPreprocessor:
    """Test cases for data preprocessor"""
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = HotelDataPreprocessor()
        
        self.sample_data = pd.DataFrame({
            'name': ['Hotel A', 'Hotel B', 'Hotel C'],
            'location': ['paris, france', 'tokyo, japan', 'new york, usa'],
            'star_rating': [4.0, 3.5, 4.5],
            'user_rating': [4.2, 3.8, 4.3],
            'min_price': [150, 100, 200],
            'max_price': [250, 180, 350],
            'amenities': [
                ['wifi', 'pool', 'gym'],
                ['wifi', 'restaurant'],
                ['wifi', 'pool', 'spa', 'business center']
            ],
            'family_friendly': [True, False, True],
            'pet_friendly': [False, True, False],
            'property_type': ['hotel', 'hotel', 'hotel'],
            'best_season': ['summer', 'spring', 'fall']
        })
    
    def test_data_cleaning(self):
        """Test data cleaning process"""
        cleaned_data = self.preprocessor.clean_data(self.sample_data.copy())
        
        # Check that data is cleaned
        assert not cleaned_data.empty
        assert 'avg_price' in cleaned_data.columns
        assert all(cleaned_data['star_rating'] >= 0)
        assert all(cleaned_data['star_rating'] <= 5)
    
    def test_feature_creation(self):
        """Test feature matrix creation"""
        cleaned_data = self.preprocessor.clean_data(self.sample_data.copy())
        X, feature_names = self.preprocessor.create_features(cleaned_data)
        
        assert isinstance(X, np.ndarray)
        assert len(feature_names) > 0
        assert X.shape[0] == len(self.sample_data)
        assert X.shape[1] == len(feature_names)
    
    def test_target_creation(self):
        """Test target variable creation"""
        cleaned_data = self.preprocessor.clean_data(self.sample_data.copy())
        targets = self.preprocessor.create_targets(cleaned_data)
        
        assert isinstance(targets, dict)
        assert len(targets) > 0
        
        for target_name, target_values in targets.items():
            assert isinstance(target_values, np.ndarray)
            assert len(target_values) == len(self.sample_data)

class TestFeatureEngineering:
    """Test cases for feature engineering"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_engineer = FeatureEngineering()
        
        self.sample_data = pd.DataFrame({
            'name': ['Hotel A', 'Hotel B'],
            'location': ['paris, france', 'tokyo, japan'],
            'star_rating': [4.0, 3.5],
            'user_rating': [4.2, 3.8],
            'min_price': [150, 100],
            'max_price': [250, 180],
            'amenities': [
                ['wifi', 'pool', 'gym', 'spa'],
                ['wifi', 'restaurant', 'parking']
            ],
            'description': [
                'Beautiful luxury hotel in the heart of Paris',
                'Modern business hotel with great amenities'
            ],
            'best_season': ['summer', 'spring'],
            'family_friendly': [True, False]
        })
    
    def test_location_features(self):
        """Test location feature engineering"""
        result = self.feature_engineer.engineer_location_features(self.sample_data.copy())
        
        assert 'country' in result.columns
        assert 'city' in result.columns
        assert 'region' in result.columns
    
    def test_price_features(self):
        """Test price feature engineering"""
        result = self.feature_engineer.engineer_price_features(self.sample_data.copy())
        
        assert 'avg_price' in result.columns
        assert 'price_range' in result.columns
        assert all(result['avg_price'] > 0)
    
    def test_amenity_features(self):
        """Test amenity feature engineering"""
        result = self.feature_engineer.engineer_amenity_features(self.sample_data.copy())
        
        assert 'total_amenities' in result.columns
        assert 'amenities_wellness' in result.columns
        assert 'amenities_tech' in result.columns
    
    def test_comprehensive_features(self):
        """Test all features together"""
        result = self.feature_engineer.apply_all_features(self.sample_data.copy())
        
        # Should have more columns than original
        assert result.shape[1] > self.sample_data.shape[1]
        assert result.shape[0] == self.sample_data.shape[0]

if __name__ == "__main__":
    pytest.main([__file__])
