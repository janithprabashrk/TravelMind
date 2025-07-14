import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import List, Dict, Tuple, Optional, Any
import re
from datetime import datetime, timedelta
import geopy.distance

from ..config import Config

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Advanced feature engineering for hotel recommendation system"""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.location_encoder = LabelEncoder()
        self.season_features = {}
        self.price_percentiles = {}
        
    def engineer_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        logger.info("Engineering location features...")
        
        # Extract country and city information
        df['country'] = df['location'].apply(self._extract_country)
        df['city'] = df['location'].apply(self._extract_city)
        
        # Encode geographical regions
        df['region'] = df['country'].apply(self._map_to_region)
        
        # Calculate location popularity (number of hotels in the area)
        location_counts = df['location'].value_counts()
        df['location_popularity'] = df['location'].map(location_counts)
        
        # Create location clusters based on similar characteristics
        df = self._create_location_clusters(df)
        
        return df
    
    def engineer_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive price-related features"""
        logger.info("Engineering price features...")
        
        # Basic price features
        if 'min_price' in df.columns and 'max_price' in df.columns:
            df['avg_price'] = (df['min_price'] + df['max_price']) / 2
            df['price_range'] = df['max_price'] - df['min_price']
            df['price_range_percentage'] = df['price_range'] / df['avg_price']
        
        # Price percentiles within location
        df = self._calculate_price_percentiles(df)
        
        # Price-to-rating ratio
        if 'user_rating' in df.columns and 'avg_price' in df.columns:
            df['price_per_rating_point'] = df['avg_price'] / (df['user_rating'] + 0.1)
            df['value_score'] = df['user_rating'] / (df['avg_price'] / 100 + 1)
        
        # Seasonal price analysis
        if 'seasonal_data' in df.columns:
            df = self._engineer_seasonal_price_features(df)
        
        return df
    
    def engineer_amenity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed amenity-based features"""
        logger.info("Engineering amenity features...")
        
        if 'amenities' not in df.columns:
            return df
        
        # Convert amenities to consistent format
        df['amenities_processed'] = df['amenities'].apply(self._process_amenities)
        
        # Basic amenity counts
        df['total_amenities'] = df['amenities_processed'].apply(len)
        
        # Categorize amenities
        amenity_categories = {
            'tech': ['wifi', 'internet', 'tv', 'computer', 'tablet'],
            'wellness': ['spa', 'massage', 'sauna', 'hot tub', 'jacuzzi'],
            'fitness': ['gym', 'fitness', 'exercise', 'yoga', 'pilates'],
            'dining': ['restaurant', 'bar', 'cafe', 'room service', 'kitchen'],
            'recreation': ['pool', 'swimming', 'tennis', 'golf', 'game room'],
            'business': ['conference', 'meeting', 'business center', 'printing'],
            'transport': ['parking', 'shuttle', 'airport', 'valet'],
            'family': ['playground', 'kids', 'children', 'babysitting', 'crib'],
            'luxury': ['concierge', 'butler', 'limousine', 'private', 'vip']
        }
        
        for category, keywords in amenity_categories.items():
            df[f'amenities_{category}'] = df['amenities_processed'].apply(
                lambda amenities: self._count_category_amenities(amenities, keywords)
            )
        
        # Unique amenity score (rarity of amenities)
        all_amenities = []
        for amenities in df['amenities_processed']:
            all_amenities.extend(amenities)
        
        amenity_frequency = pd.Series(all_amenities).value_counts()
        df['unique_amenity_score'] = df['amenities_processed'].apply(
            lambda amenities: sum(1 / (amenity_frequency.get(amenity, 1) + 1) for amenity in amenities)
        )
        
        return df
    
    def engineer_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rating-based features"""
        logger.info("Engineering rating features...")
        
        # Rating consistency
        if 'star_rating' in df.columns and 'user_rating' in df.columns:
            df['rating_consistency'] = np.abs(df['star_rating'] - df['user_rating'])
            df['rating_premium'] = df['user_rating'] - df['star_rating']
        
        # Rating within location context
        if 'location' in df.columns and 'user_rating' in df.columns:
            location_rating_stats = df.groupby('location')['user_rating'].agg(['mean', 'std'])
            df = df.merge(location_rating_stats, left_on='location', right_index=True, 
                         suffixes=('', '_location'))
            df['rating_vs_location_avg'] = df['user_rating'] - df['mean']
            df['rating_location_zscore'] = (df['user_rating'] - df['mean']) / (df['std'] + 0.1)
        
        # Rating categories
        if 'user_rating' in df.columns:
            df['rating_category'] = pd.cut(df['user_rating'], 
                                         bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
                                         labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
        
        return df
    
    def engineer_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create season-based features"""
        logger.info("Engineering seasonal features...")
        
        # Best season encoding
        if 'best_season' in df.columns:
            season_mapping = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
            df['best_season_encoded'] = df['best_season'].map(season_mapping).fillna(0)
        
        # Current season relevance (if we know current date)
        current_month = datetime.now().month
        current_season = self._month_to_season(current_month)
        
        if 'best_season' in df.columns:
            df['season_relevance'] = df['best_season'].apply(
                lambda x: 1.0 if x == current_season else 0.5
            )
        
        # Seasonal price volatility
        if 'seasonal_data' in df.columns:
            df = self._calculate_seasonal_volatility(df)
        
        return df
    
    def engineer_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text descriptions"""
        logger.info("Engineering text features...")
        
        if 'description' not in df.columns:
            return df
        
        # Basic text statistics
        df['description_length'] = df['description'].astype(str).apply(len)
        df['description_word_count'] = df['description'].astype(str).apply(lambda x: len(x.split()))
        
        # Sentiment indicators (simple keyword-based)
        positive_words = ['beautiful', 'amazing', 'excellent', 'luxury', 'comfortable', 'modern']
        negative_words = ['old', 'dated', 'small', 'noisy', 'basic']
        
        df['positive_word_count'] = df['description'].astype(str).apply(
            lambda x: sum(1 for word in positive_words if word in x.lower())
        )
        df['negative_word_count'] = df['description'].astype(str).apply(
            lambda x: sum(1 for word in negative_words if word in x.lower())
        )
        df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']
        
        # TF-IDF features (top keywords)
        try:
            descriptions = df['description'].fillna('').astype(str)
            tfidf_matrix = self.text_vectorizer.fit_transform(descriptions)
            # Convert sparse matrix to dense array
            from scipy.sparse import csr_matrix
            if hasattr(tfidf_matrix, 'toarray'):
                tfidf_array = tfidf_matrix.toarray()  # type: ignore
            else:
                tfidf_array = np.array(tfidf_matrix)
            tfidf_df = pd.DataFrame(tfidf_array, 
                                  columns=[f'tfidf_{word}' for word in self.text_vectorizer.get_feature_names_out()])
            df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
        except Exception as e:
            logger.warning(f"TF-IDF feature extraction failed: {e}")
        
        return df
    
    def engineer_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features combining multiple aspects"""
        logger.info("Engineering composite features...")
        
        # Overall value proposition
        if all(col in df.columns for col in ['user_rating', 'total_amenities', 'avg_price']):
            df['value_proposition'] = (
                df['user_rating'] * 0.4 +
                (df['total_amenities'] / df['total_amenities'].max()) * 5 * 0.3 +
                (1 - (df['avg_price'] / df['avg_price'].max())) * 5 * 0.3
            )
        
        # Luxury index
        luxury_indicators = ['amenities_luxury', 'star_rating', 'avg_price']
        available_indicators = [col for col in luxury_indicators if col in df.columns]
        
        if available_indicators:
            luxury_values = []
            for _, row in df.iterrows():
                values = []
                if 'amenities_luxury' in available_indicators and 'amenities_luxury' in row.index:
                    values.append(row['amenities_luxury'] / 5)
                if 'star_rating' in available_indicators and 'star_rating' in row.index:
                    values.append(row['star_rating'] / 5)
                if 'avg_price' in available_indicators and 'avg_price' in row.index:
                    values.append(row['avg_price'] / df['avg_price'].max())
                luxury_values.append(np.mean(values) * 5 if values else 0)
            
            df['luxury_index'] = luxury_values
        
        # Family-friendliness score
        family_indicators = ['amenities_family', 'amenities_recreation', 'family_friendly']
        available_family = [col for col in family_indicators if col in df.columns]
        
        if available_family:
            df['family_score'] = df[available_family].sum(axis=1)
        
        # Business travel suitability
        business_indicators = ['amenities_business', 'amenities_tech']
        available_business = [col for col in business_indicators if col in df.columns]
        
        if available_business:
            df['business_score'] = df[available_business].sum(axis=1)
        
        return df
    
    def _extract_country(self, location: str) -> str:
        """Extract country from location string"""
        parts = str(location).split(',')
        return parts[-1].strip().lower() if parts else 'unknown'
    
    def _extract_city(self, location: str) -> str:
        """Extract city from location string"""
        parts = str(location).split(',')
        return parts[0].strip().lower() if parts else 'unknown'
    
    def _map_to_region(self, country: str) -> str:
        """Map country to geographical region"""
        regions = {
            'europe': ['france', 'germany', 'italy', 'spain', 'uk', 'united kingdom', 'netherlands', 'belgium'],
            'asia': ['japan', 'china', 'india', 'thailand', 'singapore', 'korea', 'malaysia'],
            'north_america': ['usa', 'united states', 'canada', 'mexico'],
            'oceania': ['australia', 'new zealand'],
            'south_america': ['brazil', 'argentina', 'chile', 'peru'],
            'africa': ['south africa', 'egypt', 'morocco', 'kenya']
        }
        
        country_lower = country.lower()
        for region, countries in regions.items():
            if any(c in country_lower for c in countries):
                return region
        
        return 'other'
    
    def _create_location_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based clusters"""
        # Simple clustering based on region and price level
        if 'region' in df.columns and 'avg_price' in df.columns:
            df['price_level'] = pd.qcut(df['avg_price'], q=3, labels=['low', 'medium', 'high'])
            df['location_cluster'] = df['region'] + '_' + df['price_level'].astype(str)
        
        return df
    
    def _calculate_price_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price percentiles within location"""
        if 'location' in df.columns and 'avg_price' in df.columns:
            df['price_percentile_local'] = df.groupby('location')['avg_price'].rank(pct=True)
            df['price_percentile_global'] = df['avg_price'].rank(pct=True)
        
        return df
    
    def _engineer_seasonal_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from seasonal pricing data"""
        # This would parse the seasonal_data JSON to extract price variations
        # Simplified implementation
        df['seasonal_price_variance'] = 0.2  # Placeholder
        return df
    
    def _process_amenities(self, amenities) -> List[str]:
        """Process amenities into a clean list"""
        if pd.isna(amenities):
            return []
        
        if isinstance(amenities, list):
            return [str(a).lower().strip() for a in amenities]
        
        if isinstance(amenities, str):
            # Handle comma-separated or JSON-like strings
            amenities = amenities.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
            return [a.lower().strip() for a in amenities.split(',') if a.strip()]
        
        return []
    
    def _count_category_amenities(self, amenities: List[str], keywords: List[str]) -> int:
        """Count amenities in a specific category"""
        amenities_str = ' '.join(amenities)
        return sum(1 for keyword in keywords if keyword in amenities_str)
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _calculate_seasonal_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price volatility across seasons"""
        # Placeholder implementation
        df['seasonal_volatility'] = 0.15
        return df
    
    def apply_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        original_shape = df.shape
        
        # Apply all feature engineering steps
        df = self.engineer_location_features(df)
        df = self.engineer_price_features(df)
        df = self.engineer_amenity_features(df)
        df = self.engineer_rating_features(df)
        df = self.engineer_seasonal_features(df)
        df = self.engineer_text_features(df)
        df = self.engineer_composite_features(df)
        
        final_shape = df.shape
        logger.info(f"Feature engineering completed: {original_shape} -> {final_shape}")
        logger.info(f"Added {final_shape[1] - original_shape[1]} new features")
        
        return df

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'name': ['Hotel A', 'Hotel B', 'Hotel C'],
        'location': ['paris, france', 'tokyo, japan', 'new york, usa'],
        'star_rating': [4.0, 3.5, 4.5],
        'user_rating': [4.2, 3.8, 4.3],
        'min_price': [150, 100, 200],
        'max_price': [250, 180, 350],
        'amenities': [
            ['wifi', 'pool', 'gym', 'spa'],
            ['wifi', 'restaurant', 'parking'],
            ['wifi', 'pool', 'gym', 'business center', 'concierge']
        ],
        'description': [
            'Beautiful luxury hotel in the heart of Paris',
            'Modern business hotel with great amenities',
            'Premium accommodation in Manhattan'
        ],
        'best_season': ['summer', 'spring', 'fall'],
        'family_friendly': [True, False, True]
    })
    
    feature_engineer = FeatureEngineering()
    enhanced_data = feature_engineer.apply_all_features(sample_data)
    
    print(f"Original columns: {list(sample_data.columns)}")
    print(f"Enhanced columns: {list(enhanced_data.columns)}")
    print(f"Added {len(enhanced_data.columns) - len(sample_data.columns)} new features")
