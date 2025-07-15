import pytest
import sys
from pathlib import Path
import pandas as pd
import sqlite3

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.storage import DatabaseManager

class TestDatabaseManager:
    """Test cases for database operations"""
    
    def setup_method(self):
        """Setup test database"""
        # Use in-memory database for testing
        self.db = DatabaseManager(":memory:")
        
        self.sample_hotels = [
            {
                'name': 'Test Hotel 1',
                'location': 'paris, france',
                'star_rating': 4.0,
                'user_rating': 4.2,
                'min_price': 150,
                'max_price': 250,
                'amenities': ['wifi', 'pool', 'gym'],
                'family_friendly': True,
                'pet_friendly': False,
                'property_type': 'hotel'
            },
            {
                'name': 'Test Hotel 2',
                'location': 'tokyo, japan',
                'star_rating': 3.5,
                'user_rating': 3.8,
                'min_price': 100,
                'max_price': 180,
                'amenities': ['wifi', 'restaurant'],
                'family_friendly': False,
                'pet_friendly': True,
                'property_type': 'hotel'
            }
        ]
    
    def test_hotel_insertion(self):
        """Test hotel data insertion"""
        count = self.db.insert_hotels(self.sample_hotels)
        assert count == 2
        
        # Test duplicate insertion (should update)
        count = self.db.insert_hotels(self.sample_hotels)
        assert count == 2  # Should still be 2 (replaced)
    
    def test_hotel_retrieval(self):
        """Test hotel data retrieval"""
        # Insert test data
        self.db.insert_hotels(self.sample_hotels)
        
        # Get all hotels
        all_hotels = self.db.get_hotels()
        assert len(all_hotels) == 2
        
        # Test location filter
        paris_hotels = self.db.get_hotels(location="paris")
        assert len(paris_hotels) == 1
        assert paris_hotels[0]['name'] == 'Test Hotel 1'
        
        # Test rating filter
        high_rated = self.db.get_hotels(min_rating=4.0)
        assert len(high_rated) == 1
        
        # Test price filter
        budget_hotels = self.db.get_hotels(max_price=200)
        assert len(budget_hotels) >= 1
    
    def test_user_preferences(self):
        """Test user preferences storage"""
        preferences = {
            'location': 'paris, france',
            'budget_min': 100,
            'budget_max': 300,
            'preferred_amenities': ['wifi', 'pool'],
            'family_travel': True
        }
        
        pref_id = self.db.save_user_preferences("test_user", preferences)
        assert pref_id > 0
        
        # Retrieve preferences
        saved_prefs = self.db.get_user_preferences("test_user")
        assert len(saved_prefs) == 1
        assert saved_prefs[0]['location'] == 'paris, france'
    
    def test_recommendations_storage(self):
        """Test recommendations storage"""
        # Insert hotels first
        self.db.insert_hotels(self.sample_hotels)
        
        recommendations = [
            {'hotel_id': 1, 'score': 4.5, 'recommendation_type': 'hybrid'},
            {'hotel_id': 2, 'score': 3.8, 'recommendation_type': 'hybrid'}
        ]
        
        rec_ids = self.db.save_recommendations("test_user", recommendations)
        assert len(rec_ids) == 2
        assert all(rid > 0 for rid in rec_ids)
    
    def test_feedback_storage(self):
        """Test user feedback storage"""
        # Insert hotels and recommendations first
        self.db.insert_hotels(self.sample_hotels)
        recommendations = [{'hotel_id': 1, 'score': 4.5, 'recommendation_type': 'hybrid'}]
        rec_ids = self.db.save_recommendations("test_user", recommendations)
        
        feedback_id = self.db.save_user_feedback(
            "test_user", 1, rec_ids[0], 4.0, "Great hotel!"
        )
        assert feedback_id > 0
    
    def test_dataframe_export(self):
        """Test DataFrame export functionality"""
        # Insert test data
        self.db.insert_hotels(self.sample_hotels)
        
        # Get as DataFrame
        df = self.db.get_all_hotels_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'name' in df.columns
        assert 'location' in df.columns
    
    def test_database_stats(self):
        """Test database statistics"""
        # Insert test data
        self.db.insert_hotels(self.sample_hotels)
        
        stats = self.db.get_database_stats()
        assert isinstance(stats, dict)
        assert stats['total_hotels'] == 2
        assert stats['unique_locations'] >= 1

if __name__ == "__main__":
    pytest.main([__file__])
