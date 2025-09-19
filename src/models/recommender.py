import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import joblib

from ..utils.config import get_settings
from ..data.storage import DatabaseManager

logger = logging.getLogger(__name__)

# Get settings instance
settings = get_settings()

class HotelRecommendationEngine:
    """Advanced hotel recommendation system using multiple ML approaches"""
    
    def __init__(self):
        self.content_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.collaborative_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.value_model = Ridge(alpha=1.0, random_state=42)
        self.luxury_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.family_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Clustering for location-based recommendations
        self.location_clusters = KMeans(n_clusters=10, random_state=42)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Feature importance and similarity matrices
        self.feature_importance = {}
        self.hotel_similarity_matrix = None
        self.hotel_features = None
        self.hotel_ids = None
        
        # Model performance tracking
        self.model_scores = {}
        self.is_trained = False
        
        # Database connection
        self.db = DatabaseManager()
    
    def train_models(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                    feature_names: List[str], hotel_ids: Optional[List[int]] = None):
        """Train all recommendation models"""
        logger.info("Starting model training...")
        
        if X.shape[0] < 10:
            raise ValueError("Need at least 10 samples for training")
        
        self.feature_names = feature_names
        self.hotel_ids = hotel_ids or list(range(X.shape[0]))
        
        # Apply PCA for dimensionality reduction if needed
        if X.shape[1] > 50:
            logger.info("Applying PCA for dimensionality reduction...")
            if self.pca is None:
                self.pca = PCA(n_components=0.95)
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = X
            self.pca = None
        
        self.hotel_features = X_reduced
        
        # Train individual models
        self._train_content_model(X_reduced, y_dict)
        self._train_collaborative_model(X_reduced, y_dict)
        self._train_specialized_models(X_reduced, y_dict)
        
        # Create similarity matrix
        self._create_similarity_matrix(X_reduced)
        
        # Train location clusters
        self._train_location_clusters(X_reduced)
        
        self.is_trained = True
        logger.info("Model training completed successfully")
    
    def _train_content_model(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Train content-based recommendation model"""
        if 'overall_score' in y_dict:
            self.content_model.fit(X, y_dict['overall_score'])
            self.feature_importance['content'] = dict(zip(
                self.feature_names[:X.shape[1]], 
                self.content_model.feature_importances_
            ))
            logger.info("Content-based model trained")
    
    def _train_collaborative_model(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Train collaborative filtering model (simplified version)"""
        # For demo purposes, using overall score as target
        # In a real system, this would use user-item interaction data
        if 'overall_score' in y_dict:
            self.collaborative_model.fit(X, y_dict['overall_score'])
            logger.info("Collaborative filtering model trained")
    
    def _train_specialized_models(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Train specialized models for different use cases"""
        # Value-for-money model
        if 'value_score' in y_dict:
            self.value_model.fit(X, y_dict['value_score'])
            logger.info("Value model trained")
        
        # Luxury model
        if 'luxury_score' in y_dict:
            self.luxury_model.fit(X, y_dict['luxury_score'])
            logger.info("Luxury model trained")
        
        # Family-friendly model
        if 'family_score' in y_dict:
            self.family_model.fit(X, y_dict['family_score'])
            logger.info("Family model trained")
    
    def _create_similarity_matrix(self, X: np.ndarray):
        """Create hotel similarity matrix for content-based filtering"""
        self.hotel_similarity_matrix = cosine_similarity(X)
        logger.info(f"Similarity matrix created: {self.hotel_similarity_matrix.shape}")
    
    def _train_location_clusters(self, X: np.ndarray):
        """Train location-based clustering"""
        # Use a subset of features relevant to location
        if X.shape[1] > 5:
            location_features = X[:, :5]  # Use first 5 features as proxy
        else:
            location_features = X
        
        self.location_clusters.fit(location_features)
        logger.info("Location clustering model trained")
    
    def predict_recommendations(self, user_preferences: Dict[str, Any], 
                              candidate_hotels: List[Dict[str, Any]],
                              recommendation_type: str = 'hybrid',
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Generate hotel recommendations based on user preferences"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if not candidate_hotels:
            logger.warning("No candidate hotels provided")
            return []
        
        # Convert hotels to feature vectors
        hotel_features = self._hotels_to_features(candidate_hotels)
        
        if hotel_features.size == 0:
            logger.warning("Could not extract features from candidate hotels")
            return []
        
        # Apply PCA if it was used during training
        if self.pca is not None:
            hotel_features = self.pca.transform(hotel_features)
        
        # Calculate scores based on recommendation type
        if recommendation_type == 'content':
            scores = self._content_based_scores(hotel_features, user_preferences)
        elif recommendation_type == 'collaborative':
            scores = self._collaborative_scores(hotel_features, user_preferences)
        elif recommendation_type == 'value':
            scores = self._value_based_scores(hotel_features, user_preferences)
        elif recommendation_type == 'luxury':
            scores = self._luxury_based_scores(hotel_features, user_preferences)
        elif recommendation_type == 'family':
            scores = self._family_based_scores(hotel_features, user_preferences)
        else:  # hybrid
            scores = self._hybrid_scores(hotel_features, user_preferences)
        
        # Apply user preference filters
        scores = self._apply_preference_filters(scores, candidate_hotels, user_preferences)
        
        # Sort and get top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            if idx < len(candidate_hotels):
                rec = candidate_hotels[idx].copy()
                rec['recommendation_score'] = float(scores[idx])
                rec['recommendation_rank'] = i + 1
                rec['recommendation_type'] = recommendation_type
                recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} {recommendation_type} recommendations")
        return recommendations
    
    def _hotels_to_features(self, hotels: List[Dict[str, Any]]) -> np.ndarray:
        """Convert hotel dictionaries to feature vectors"""
        # This is a simplified version - in practice, you'd use the same
        # preprocessing pipeline as during training
        features = []
        
        for hotel in hotels:
            feature_vector = []
            
            # Basic features
            feature_vector.extend([
                hotel.get('star_rating', 3.0),
                hotel.get('user_rating', 3.0),
                hotel.get('min_price', 100.0),
                hotel.get('max_price', 200.0),
                hotel.get('avg_price', 150.0) if 'avg_price' in hotel else 
                    (hotel.get('min_price', 100.0) + hotel.get('max_price', 200.0)) / 2
            ])
            
            # Boolean features
            feature_vector.extend([
                1 if hotel.get('family_friendly', False) else 0,
                1 if hotel.get('pet_friendly', False) else 0
            ])
            
            # Amenity features (simplified)
            amenities = hotel.get('amenities', [])
            if isinstance(amenities, str):
                amenities = amenities.split(',')
            
            feature_vector.extend([
                1 if any('pool' in str(a).lower() for a in amenities) else 0,
                1 if any('wifi' in str(a).lower() for a in amenities) else 0,
                1 if any('gym' in str(a).lower() for a in amenities) else 0,
                1 if any('spa' in str(a).lower() for a in amenities) else 0,
                len(amenities)
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _content_based_scores(self, hotel_features: np.ndarray, 
                            user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate content-based recommendation scores"""
        try:
            return self.content_model.predict(hotel_features)
        except Exception as e:
            logger.warning(f"Content model prediction failed: {e}")
            return np.random.random(len(hotel_features))
    
    def _collaborative_scores(self, hotel_features: np.ndarray,
                            user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate collaborative filtering scores"""
        try:
            return self.collaborative_model.predict(hotel_features)
        except Exception as e:
            logger.warning(f"Collaborative model prediction failed: {e}")
            return np.random.random(len(hotel_features))
    
    def _value_based_scores(self, hotel_features: np.ndarray,
                          user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate value-for-money scores"""
        try:
            return self.value_model.predict(hotel_features)
        except Exception as e:
            logger.warning(f"Value model prediction failed: {e}")
            return np.random.random(len(hotel_features))
    
    def _luxury_based_scores(self, hotel_features: np.ndarray,
                           user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate luxury-focused scores"""
        try:
            return self.luxury_model.predict(hotel_features)
        except Exception as e:
            logger.warning(f"Luxury model prediction failed: {e}")
            return np.random.random(len(hotel_features))
    
    def _family_based_scores(self, hotel_features: np.ndarray,
                           user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate family-friendly scores"""
        try:
            return self.family_model.predict(hotel_features)
        except Exception as e:
            logger.warning(f"Family model prediction failed: {e}")
            return np.random.random(len(hotel_features))
    
    def _hybrid_scores(self, hotel_features: np.ndarray,
                      user_preferences: Dict[str, Any]) -> np.ndarray:
        """Calculate hybrid recommendation scores"""
        scores = np.zeros(len(hotel_features))
        
        # Weight different models based on user preferences
        weights = {
            'content': 0.3,
            'collaborative': 0.2,
            'value': 0.2,
            'luxury': 0.15,
            'family': 0.15
        }
        
        # Adjust weights based on user preferences
        if user_preferences.get('family_travel', False):
            weights['family'] += 0.2
            weights['content'] -= 0.1
            weights['luxury'] -= 0.1
        
        if user_preferences.get('business_travel', False):
            weights['luxury'] += 0.15
            weights['family'] -= 0.15
        
        budget_max = user_preferences.get('budget_max', 300)
        if budget_max < 150:  # Budget travel
            weights['value'] += 0.2
            weights['luxury'] -= 0.2
        elif budget_max > 400:  # Luxury travel
            weights['luxury'] += 0.2
            weights['value'] -= 0.2
        
        # Combine predictions
        try:
            scores += weights['content'] * self.content_model.predict(hotel_features)
        except:
            pass
        
        try:
            scores += weights['collaborative'] * self.collaborative_model.predict(hotel_features)
        except:
            pass
        
        try:
            scores += weights['value'] * self.value_model.predict(hotel_features)
        except:
            pass
        
        try:
            scores += weights['luxury'] * self.luxury_model.predict(hotel_features)
        except:
            pass
        
        try:
            scores += weights['family'] * self.family_model.predict(hotel_features)
        except:
            pass
        
        return scores
    
    def _apply_preference_filters(self, scores: np.ndarray, 
                                hotels: List[Dict[str, Any]], 
                                user_preferences: Dict[str, Any]) -> np.ndarray:
        """Apply user preference filters to scores"""
        filtered_scores = scores.copy()
        
        for i, hotel in enumerate(hotels):
            # Budget filter
            budget_min = user_preferences.get('budget_min', 0)
            budget_max = user_preferences.get('budget_max', float('inf'))
            hotel_price = hotel.get('avg_price', 
                                  (hotel.get('min_price', 0) + hotel.get('max_price', 0)) / 2)
            
            if not (budget_min <= hotel_price <= budget_max):
                filtered_scores[i] *= 0.3  # Heavily penalize out-of-budget hotels
            
            # Minimum rating filter
            min_rating = user_preferences.get('min_rating', 0)
            if hotel.get('user_rating', 0) < min_rating:
                filtered_scores[i] *= 0.5
            
            # Amenity preferences
            preferred_amenities = user_preferences.get('preferred_amenities', [])
            if preferred_amenities:
                hotel_amenities = hotel.get('amenities', [])
                if isinstance(hotel_amenities, str):
                    hotel_amenities = hotel_amenities.split(',')
                
                hotel_amenities_str = ' '.join(str(a).lower() for a in hotel_amenities)
                amenity_score = sum(1 for amenity in preferred_amenities 
                                  if amenity.lower() in hotel_amenities_str)
                
                if amenity_score > 0:
                    filtered_scores[i] *= (1 + amenity_score * 0.1)  # Boost matching amenities
        
        return filtered_scores
    
    def get_similar_hotels(self, hotel_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get hotels similar to a given hotel"""
        if self.hotel_similarity_matrix is None:
            raise ValueError("Similarity matrix not available")
        
        if hotel_id >= len(self.hotel_similarity_matrix):
            raise ValueError(f"Hotel ID {hotel_id} not found in similarity matrix")
        
        similarities = self.hotel_similarity_matrix[hotel_id]
        # Exclude the hotel itself
        similarities[hotel_id] = -1
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similar_hotels = [(idx, similarities[idx]) for idx in top_indices]
        
        return similar_hotels
    
    def save_models(self, filepath: str):
        """Save all trained models to disk"""
        model_data = {
            'content_model': self.content_model,
            'collaborative_model': self.collaborative_model,
            'value_model': self.value_model,
            'luxury_model': self.luxury_model,
            'family_model': self.family_model,
            'location_clusters': self.location_clusters,
            'pca': self.pca,
            'feature_importance': self.feature_importance,
            'hotel_similarity_matrix': self.hotel_similarity_matrix,
            'hotel_features': self.hotel_features,
            'hotel_ids': self.hotel_ids,
            'feature_names': getattr(self, 'feature_names', []),
            'model_scores': self.model_scores,
            'is_trained': self.is_trained,
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.content_model = model_data['content_model']
        self.collaborative_model = model_data['collaborative_model']
        self.value_model = model_data['value_model']
        self.luxury_model = model_data['luxury_model']
        self.family_model = model_data['family_model']
        self.location_clusters = model_data['location_clusters']
        self.pca = model_data.get('pca')
        self.feature_importance = model_data['feature_importance']
        self.hotel_similarity_matrix = model_data.get('hotel_similarity_matrix')
        self.hotel_features = model_data.get('hotel_features')
        self.hotel_ids = model_data.get('hotel_ids', [])
        self.feature_names = model_data.get('feature_names', [])
        self.model_scores = model_data.get('model_scores', {})
        self.is_trained = model_data.get('is_trained', False)
        
        logger.info(f"Models loaded from {filepath}")
    
    def get_feature_importance(self, model_type: str = 'content') -> Dict[str, float]:
        """Get feature importance for a specific model"""
        return self.feature_importance.get(model_type, {})
    
    def update_with_feedback(self, user_feedback: List[Dict[str, Any]]):
        """Update models based on user feedback (simplified online learning)"""
        # This is a placeholder for online learning implementation
        # In a production system, you would implement incremental learning
        logger.info(f"Received {len(user_feedback)} feedback entries for model update")
        
        # For now, just log the feedback
        for feedback in user_feedback:
            logger.info(f"User {feedback.get('user_id')} rated hotel {feedback.get('hotel_id')} "
                       f"with score {feedback.get('rating')}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from the training pipeline
    engine = HotelRecommendationEngine()
    
    # Example with dummy data
    import numpy as np
    X_dummy = np.random.random((100, 12))
    y_dummy = {
        'overall_score': np.random.random(100) * 5,
        'value_score': np.random.random(100) * 5,
        'luxury_score': np.random.random(100) * 5,
        'family_score': np.random.random(100) * 5
    }
    feature_names = [f'feature_{i}' for i in range(12)]
    
    try:
        engine.train_models(X_dummy, y_dummy, feature_names)
        print("Model training successful!")
        
        # Save models
        from pathlib import Path
        model_path = Path(settings.MODEL_PATH) / "recommendation_models.pkl"
        engine.save_models(str(model_path))
        print(f"Models saved to {model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
