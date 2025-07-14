import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pickle
import json
from pathlib import Path

from ..config import Config
from ..data.collector import HotelDataCollector
from ..data.preprocessor import HotelDataPreprocessor
from ..data.storage import DatabaseManager
from .recommender import HotelRecommendationEngine
from .feature_engine import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Comprehensive model training pipeline for hotel recommendation system"""
    
    def __init__(self):
        self.data_collector = HotelDataCollector()
        self.preprocessor = HotelDataPreprocessor()
        self.feature_engineer = FeatureEngineering()
        self.recommendation_engine = HotelRecommendationEngine()
        self.db = DatabaseManager()
        
        self.training_history = []
        self.model_metrics = {}
        
    async def collect_training_data(self, locations: List[str]) -> pd.DataFrame:
        """Collect training data from multiple locations"""
        logger.info(f"Starting data collection for {len(locations)} locations...")
        
        all_hotels = []
        
        for location in locations:
            try:
                logger.info(f"Collecting data for {location}...")
                location_data = await self.data_collector.collect_location_data(location)
                
                if location_data and 'hotels' in location_data:
                    hotels = location_data['hotels']
                    
                    # Save to database
                    self.db.insert_hotels(hotels)
                    
                    # Save raw data for backup
                    self.data_collector.save_to_json(location_data, location)
                    csv_file = self.data_collector.save_to_csv(hotels, location)
                    
                    all_hotels.extend(hotels)
                    logger.info(f"Collected {len(hotels)} hotels from {location}")
                
            except Exception as e:
                logger.error(f"Failed to collect data for {location}: {str(e)}")
                continue
        
        if not all_hotels:
            raise ValueError("No training data could be collected")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_hotels)
        logger.info(f"Total training data collected: {len(df)} hotels from {len(locations)} locations")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """Prepare data for model training"""
        logger.info("Preparing training data...")
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.apply_all_features(df)
        
        # Preprocess for ML
        X, y_dict, feature_names = self.preprocessor.prepare_for_training(df_engineered)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y_dict, feature_names
    
    def train_models(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                    feature_names: List[str], test_size: float = 0.2) -> Dict[str, Any]:
        """Train and evaluate all models"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train_dict, y_test_dict = self._split_data(X, y_dict, test_size)
        
        # Train recommendation engine
        self.recommendation_engine.train_models(X_train, y_train_dict, feature_names)
        
        # Evaluate models
        metrics = self._evaluate_models(X_test, y_test_dict)
        
        # Perform cross-validation
        cv_scores = self._cross_validate_models(X, y_dict)
        
        # Update training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'test_size': test_size,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'feature_names': feature_names
        }
        
        self.training_history.append(training_record)
        self.model_metrics = metrics
        
        logger.info("Model training completed successfully")
        return training_record
    
    def _split_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                   test_size: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Split data into training and testing sets"""
        # Use the first target variable for stratification
        primary_target = list(y_dict.values())[0]
        
        X_train, X_test, _, _ = train_test_split(
            X, primary_target, test_size=test_size, random_state=42
        )
        
        # Split all target variables
        y_train_dict = {}
        y_test_dict = {}
        
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_train), len(X))
        
        # Get indices from train_test_split
        all_indices = np.arange(len(X))
        X_train_full, X_test_full, indices_train, indices_test = train_test_split(
            X, all_indices, test_size=test_size, random_state=42
        )
        
        for target_name, target_values in y_dict.items():
            y_train_dict[target_name] = target_values[indices_train]
            y_test_dict[target_name] = target_values[indices_test]
        
        return X_train_full, X_test_full, y_train_dict, y_test_dict
    
    def _evaluate_models(self, X_test: np.ndarray, y_test_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance"""
        metrics = {}
        
        # Test each model if target data is available
        if 'overall_score' in y_test_dict:
            try:
                y_pred = self.recommendation_engine.content_model.predict(X_test)
                metrics['content_model'] = {
                    'mse': float(mean_squared_error(y_test_dict['overall_score'], y_pred)),
                    'mae': float(mean_absolute_error(y_test_dict['overall_score'], y_pred)),
                    'r2': float(r2_score(y_test_dict['overall_score'], y_pred))
                }
            except Exception as e:
                logger.warning(f"Content model evaluation failed: {e}")
        
        if 'value_score' in y_test_dict:
            try:
                y_pred = self.recommendation_engine.value_model.predict(X_test)
                metrics['value_model'] = {
                    'mse': float(mean_squared_error(y_test_dict['value_score'], y_pred)),
                    'mae': float(mean_absolute_error(y_test_dict['value_score'], y_pred)),
                    'r2': float(r2_score(y_test_dict['value_score'], y_pred))
                }
            except Exception as e:
                logger.warning(f"Value model evaluation failed: {e}")
        
        if 'luxury_score' in y_test_dict:
            try:
                y_pred = self.recommendation_engine.luxury_model.predict(X_test)
                metrics['luxury_model'] = {
                    'mse': float(mean_squared_error(y_test_dict['luxury_score'], y_pred)),
                    'mae': float(mean_absolute_error(y_test_dict['luxury_score'], y_pred)),
                    'r2': float(r2_score(y_test_dict['luxury_score'], y_pred))
                }
            except Exception as e:
                logger.warning(f"Luxury model evaluation failed: {e}")
        
        if 'family_score' in y_test_dict:
            try:
                y_pred = self.recommendation_engine.family_model.predict(X_test)
                metrics['family_model'] = {
                    'mse': float(mean_squared_error(y_test_dict['family_score'], y_pred)),
                    'mae': float(mean_absolute_error(y_test_dict['family_score'], y_pred)),
                    'r2': float(r2_score(y_test_dict['family_score'], y_pred))
                }
            except Exception as e:
                logger.warning(f"Family model evaluation failed: {e}")
        
        return metrics
    
    def _cross_validate_models(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation"""
        cv_scores = {}
        
        # Cross-validate content model
        if 'overall_score' in y_dict:
            try:
                scores = cross_val_score(
                    self.recommendation_engine.content_model, X, y_dict['overall_score'], 
                    cv=5, scoring='r2'
                )
                cv_scores['content_model'] = {
                    'mean_r2': float(np.mean(scores)),
                    'std_r2': float(np.std(scores))
                }
            except Exception as e:
                logger.warning(f"Content model CV failed: {e}")
        
        return cv_scores
    
    def save_models(self, model_path: Optional[str] = None) -> str:
        """Save all trained models and components"""
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = str(Config.MODEL_PATH / f"travelmind_models_{timestamp}.pkl")
        
        # Save recommendation engine
        self.recommendation_engine.save_models(str(model_path))
        
        # Save preprocessor
        preprocessor_path = str(model_path).replace('.pkl', '_preprocessor.pkl')
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Save training history
        history_path = str(model_path).replace('.pkl', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Models saved to {model_path}")
        return str(model_path)
    
    def load_models(self, model_path: str):
        """Load trained models"""
        self.recommendation_engine.load_models(model_path)
        
        # Load preprocessor
        preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
        if Path(preprocessor_path).exists():
            self.preprocessor.load_preprocessor(preprocessor_path)
        
        # Load training history
        history_path = model_path.replace('.pkl', '_history.json')
        if Path(history_path).exists():
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        logger.info(f"Models loaded from {model_path}")
    
    def hyperparameter_tuning(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        logger.info("Starting hyperparameter tuning...")
        
        best_params = {}
        
        # Tune content model (Random Forest)
        if 'overall_score' in y_dict:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(random_state=42)
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X, y_dict['overall_score'])
                
                best_params['content_model'] = grid_search.best_params_
                logger.info(f"Best content model params: {grid_search.best_params_}")
                
                # Update the model with best parameters
                self.recommendation_engine.content_model = grid_search.best_estimator_
                
            except Exception as e:
                logger.warning(f"Hyperparameter tuning for content model failed: {e}")
        
        return best_params
    
    def generate_model_report(self) -> Dict[str, Any]:
        """Generate comprehensive model performance report"""
        report = {
            'training_summary': {
                'total_training_sessions': len(self.training_history),
                'last_training': self.training_history[-1] if self.training_history else None,
                'current_metrics': self.model_metrics
            },
            'feature_importance': self.recommendation_engine.get_feature_importance('content'),
            'model_status': {
                'is_trained': self.recommendation_engine.is_trained,
                'models_available': ['content', 'collaborative', 'value', 'luxury', 'family']
            },
            'data_statistics': self.db.get_database_stats()
        }
        
        return report
    
    async def full_training_pipeline(self, locations: List[str], 
                                   retrain_existing: bool = False) -> Dict[str, Any]:
        """Execute complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        try:
            # Check if we have existing data
            existing_data = self.db.get_all_hotels_df()
            
            if existing_data.empty or retrain_existing:
                # Collect new data
                training_data = await self.collect_training_data(locations)
            else:
                logger.info(f"Using existing data: {len(existing_data)} hotels")
                training_data = existing_data
            
            # Prepare data
            X, y_dict, feature_names = self.prepare_training_data(training_data)
            
            # Hyperparameter tuning (optional)
            best_params = self.hyperparameter_tuning(X, y_dict)
            
            # Train models
            training_results = self.train_models(X, y_dict, feature_names)
            
            # Save models
            model_path = self.save_models()
            
            # Generate report
            report = self.generate_model_report()
            
            final_results = {
                'status': 'success',
                'training_results': training_results,
                'best_params': best_params,
                'model_path': model_path,
                'report': report
            }
            
            logger.info("Full training pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

# Example usage and testing
async def main():
    """Example usage of the training pipeline"""
    trainer = ModelTrainer()
    
    # Define training locations
    training_locations = [
        "Paris, France",
        "Tokyo, Japan", 
        "New York, USA",
        "London, UK",
        "Barcelona, Spain",
        "Sydney, Australia"
    ]
    
    try:
        # Run full training pipeline
        results = await trainer.full_training_pipeline(training_locations)
        
        print("Training Results:")
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            print(f"Model saved to: {results['model_path']}")
            print(f"Training metrics: {results['training_results']['metrics']}")
            print(f"Database statistics: {results['report']['data_statistics']}")
        else:
            print(f"Error: {results['error']}")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
