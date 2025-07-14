import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import logging
from typing import List, Dict, Tuple, Optional
import re
import json
from datetime import datetime
import pickle

from ..config import Config

logger = logging.getLogger(__name__)

class HotelDataPreprocessor:
    """Preprocesses hotel data for machine learning models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.amenity_encoder = MultiLabelBinarizer()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.target_columns = []
        
    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """Load and combine multiple CSV files"""
        dfs = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        if not dfs:
            raise ValueError("No data files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['name', 'location'], keep='first')
        logger.info(f"Removed {initial_rows - len(df)} duplicate records")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Standardize text fields
        df = self._standardize_text_fields(df)
        
        # Parse and clean numeric fields
        df = self._clean_numeric_fields(df)
        
        # Parse amenities and other list fields
        df = self._parse_list_fields(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing ratings with median
        numeric_columns = ['star_rating', 'user_rating', 'min_price', 'max_price']
        for col in numeric_columns:
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Fill missing text fields
        text_columns = ['description', 'best_season', 'property_type']
        for col in text_columns:
            if col in df.columns:
                df[col].fillna('Unknown', inplace=True)
        
        # Fill missing boolean fields
        bool_columns = ['family_friendly', 'pet_friendly']
        for col in bool_columns:
            if col in df.columns:
                df[col].fillna(False, inplace=True)
        
        return df
    
    def _standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text fields"""
        text_columns = ['name', 'location', 'description', 'best_season', 'property_type']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        return df
    
    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate numeric fields"""
        # Ensure ratings are in valid ranges
        if 'star_rating' in df.columns:
            df['star_rating'] = df['star_rating'].clip(0, 5)
        
        if 'user_rating' in df.columns:
            df['user_rating'] = df['user_rating'].clip(0, 5)
        
        # Ensure prices are positive
        price_columns = ['min_price', 'max_price']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Fix inverted price ranges
        if 'min_price' in df.columns and 'max_price' in df.columns:
            mask = df['min_price'] > df['max_price']
            df.loc[mask, ['min_price', 'max_price']] = df.loc[mask, ['max_price', 'min_price']].values
        
        return df
    
    def _parse_list_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse comma-separated list fields"""
        list_columns = ['amenities', 'room_types', 'nearby_attractions', 'business_facilities', 'accessibility']
        
        for col in list_columns:
            if col in df.columns:
                # Convert string representations back to lists
                df[col] = df[col].apply(self._parse_string_list)
        
        return df
    
    def _parse_string_list(self, value) -> List[str]:
        """Parse string representation of a list"""
        if pd.isna(value) or value == '':
            return []
        
        if isinstance(value, list):
            return value
        
        # Handle comma-separated strings
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(',') if item.strip()]
        
        return []
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML models"""
        # Price features
        if 'min_price' in df.columns and 'max_price' in df.columns:
            df['avg_price'] = (df['min_price'] + df['max_price']) / 2
            df['price_range'] = df['max_price'] - df['min_price']
            df['price_category'] = pd.cut(df['avg_price'], 
                                        bins=[0, 100, 200, 400, float('inf')], 
                                        labels=['budget', 'mid_range', 'luxury', 'ultra_luxury'])
        
        # Rating features
        if 'star_rating' in df.columns and 'user_rating' in df.columns:
            df['rating_difference'] = df['user_rating'] - df['star_rating']
            df['overall_score'] = (df['star_rating'] * 0.3 + df['user_rating'] * 0.7)
        
        # Amenity features
        if 'amenities' in df.columns:
            df['amenity_count'] = df['amenities'].apply(len)
            df['has_pool'] = df['amenities'].apply(lambda x: 'pool' in ' '.join(x).lower())
            df['has_wifi'] = df['amenities'].apply(lambda x: 'wifi' in ' '.join(x).lower())
            df['has_gym'] = df['amenities'].apply(lambda x: 'gym' in ' '.join(x).lower())
            df['has_spa'] = df['amenities'].apply(lambda x: 'spa' in ' '.join(x).lower())
            df['has_restaurant'] = df['amenities'].apply(lambda x: 'restaurant' in ' '.join(x).lower())
        
        # Location features
        if 'location' in df.columns:
            df['location_encoded'] = self._encode_location(df['location'])
        
        # Seasonal features
        if 'best_season' in df.columns:
            season_map = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4, 'unknown': 0}
            df['best_season_encoded'] = df['best_season'].map(season_map).fillna(0)
        
        return df
    
    def _encode_location(self, locations: pd.Series) -> pd.Series:
        """Encode location data"""
        # Extract country information
        countries = []
        for location in locations:
            parts = str(location).split(',')
            if len(parts) > 1:
                countries.append(parts[-1].strip())
            else:
                countries.append('unknown')
        
        if 'location' not in self.label_encoders:
            self.label_encoders['location'] = LabelEncoder()
            return self.label_encoders['location'].fit_transform(countries)
        else:
            return self.label_encoders['location'].transform(countries)
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Create feature matrix for ML models"""
        features = []
        feature_names = []
        
        # Numeric features
        numeric_features = ['star_rating', 'user_rating', 'min_price', 'max_price', 
                          'avg_price', 'price_range', 'rating_difference', 'overall_score',
                          'amenity_count', 'best_season_encoded', 'location_encoded']
        
        for feature in numeric_features:
            if feature in df.columns:
                features.append(df[feature].values.reshape(-1, 1))
                feature_names.append(feature)
        
        # Boolean features
        boolean_features = ['family_friendly', 'pet_friendly', 'has_pool', 'has_wifi', 
                          'has_gym', 'has_spa', 'has_restaurant']
        
        for feature in boolean_features:
            if feature in df.columns:
                features.append(df[feature].astype(int).values.reshape(-1, 1))
                feature_names.append(feature)
        
        # Categorical features (one-hot encoded)
        if 'property_type' in df.columns:
            if 'property_type' not in self.label_encoders:
                self.label_encoders['property_type'] = LabelEncoder()
                encoded = self.label_encoders['property_type'].fit_transform(df['property_type'])
            else:
                encoded = self.label_encoders['property_type'].transform(df['property_type'])
            
            features.append(encoded.reshape(-1, 1))
            feature_names.append('property_type_encoded')
        
        if 'price_category' in df.columns:
            if 'price_category' not in self.label_encoders:
                self.label_encoders['price_category'] = LabelEncoder()
                encoded = self.label_encoders['price_category'].fit_transform(df['price_category'].astype(str))
            else:
                encoded = self.label_encoders['price_category'].transform(df['price_category'].astype(str))
            
            features.append(encoded.reshape(-1, 1))
            feature_names.append('price_category_encoded')
        
        # Amenity features (multi-label binarized)
        if 'amenities' in df.columns:
            amenity_matrix = self.amenity_encoder.fit_transform(df['amenities'])
            features.append(amenity_matrix)
            amenity_feature_names = [f'amenity_{name}' for name in self.amenity_encoder.classes_]
            feature_names.extend(amenity_feature_names)
        
        # Combine all features
        if features:
            feature_matrix = np.hstack(features)
            self.feature_columns = feature_names
            return feature_matrix, feature_names
        else:
            raise ValueError("No features could be created from the data")
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create target variables for different recommendation scenarios"""
        targets = {}
        
        # Overall recommendation score (composite)
        if 'overall_score' in df.columns:
            targets['overall_score'] = df['overall_score'].values
        
        # Price value score (rating per price unit)
        if 'user_rating' in df.columns and 'avg_price' in df.columns:
            # Normalize price to 0-5 scale for comparison with rating
            normalized_price = 5 - (df['avg_price'] - df['avg_price'].min()) / (df['avg_price'].max() - df['avg_price'].min()) * 4
            targets['value_score'] = (df['user_rating'] + normalized_price) / 2
        
        # Luxury score
        if 'star_rating' in df.columns and 'avg_price' in df.columns:
            targets['luxury_score'] = (df['star_rating'] * 0.6 + 
                                     (df['avg_price'] - df['avg_price'].min()) / (df['avg_price'].max() - df['avg_price'].min()) * 5 * 0.4)
        
        # Family friendliness score
        family_score = np.zeros(len(df))
        if 'family_friendly' in df.columns:
            family_score += df['family_friendly'].astype(int) * 2
        if 'has_pool' in df.columns:
            family_score += df['has_pool'].astype(int)
        if 'amenities' in df.columns:
            family_score += df['amenities'].apply(lambda x: sum([1 for amenity in x if 'kid' in amenity.lower() or 'child' in amenity.lower()]))
        
        targets['family_score'] = family_score
        
        self.target_columns = list(targets.keys())
        return targets
    
    def fit_scalers(self, X: np.ndarray):
        """Fit the scalers on the training data"""
        self.scaler.fit(X)
        logger.info("Scalers fitted on training data")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scalers"""
        return self.scaler.transform(X)
    
    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """Prepare complete dataset for training"""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Create features and targets
        X, feature_names = self.create_features(df_clean)
        y_dict = self.create_targets(df_clean)
        
        # Fit and transform features
        self.fit_scalers(X)
        X_scaled = self.transform_features(X)
        
        logger.info(f"Prepared dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"Target variables: {list(y_dict.keys())}")
        
        return X_scaled, y_dict, feature_names
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'amenity_encoder': self.amenity_encoder,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load a fitted preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.amenity_encoder = preprocessor_data['amenity_encoder']
        self.imputer = preprocessor_data['imputer']
        self.feature_columns = preprocessor_data['feature_columns']
        self.target_columns = preprocessor_data['target_columns']
        
        logger.info(f"Preprocessor loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    preprocessor = HotelDataPreprocessor()
    
    # Example with dummy data
    sample_data = pd.DataFrame({
        'name': ['Hotel A', 'Hotel B'],
        'location': ['paris, france', 'tokyo, japan'],
        'star_rating': [4.0, 3.5],
        'user_rating': [4.2, 3.8],
        'min_price': [150, 100],
        'max_price': [250, 180],
        'amenities': [['wifi', 'pool', 'gym'], ['wifi', 'restaurant']],
        'family_friendly': [True, False],
        'pet_friendly': [False, True],
        'property_type': ['hotel', 'hotel'],
        'best_season': ['summer', 'spring']
    })
    
    try:
        X, y_dict, feature_names = preprocessor.prepare_for_training(sample_data)
        print(f"Features shape: {X.shape}")
        print(f"Feature names: {feature_names}")
        print(f"Targets: {list(y_dict.keys())}")
    except Exception as e:
        print(f"Error: {e}")
