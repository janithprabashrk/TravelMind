import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for TravelMind"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection"""
        self.db_path = db_path or Config.DATA_PATH / "travelmind.db"
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Hotels table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hotels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    address TEXT,
                    star_rating REAL,
                    user_rating REAL,
                    min_price REAL,
                    max_price REAL,
                    property_type TEXT,
                    amenities TEXT,  -- JSON string
                    room_types TEXT,  -- JSON string
                    description TEXT,
                    best_season TEXT,
                    nearby_attractions TEXT,  -- JSON string
                    contact_info TEXT,  -- JSON string
                    sustainability_rating REAL,
                    business_facilities TEXT,  -- JSON string
                    family_friendly BOOLEAN,
                    pet_friendly BOOLEAN,
                    accessibility TEXT,  -- JSON string
                    seasonal_data TEXT,  -- JSON string
                    collected_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, location)
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    location TEXT,
                    budget_min REAL,
                    budget_max REAL,
                    preferred_season TEXT,
                    preferred_amenities TEXT,  -- JSON string
                    property_type_preference TEXT,
                    family_travel BOOLEAN,
                    business_travel BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Recommendations history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    hotel_id INTEGER,
                    score REAL,
                    recommendation_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hotel_id) REFERENCES hotels (id)
                )
            """)
            
            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    hotel_id INTEGER,
                    recommendation_id INTEGER,
                    rating REAL,
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (hotel_id) REFERENCES hotels (id),
                    FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hotels_location ON hotels(location)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hotels_rating ON hotels(user_rating)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hotels_price ON hotels(min_price, max_price)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)")
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    def insert_hotels(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotel data into database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            inserted_count = 0
            for hotel in hotels:
                try:
                    # Convert lists and dicts to JSON strings
                    amenities = json.dumps(hotel.get('amenities', []))
                    room_types = json.dumps(hotel.get('room_types', []))
                    nearby_attractions = json.dumps(hotel.get('nearby_attractions', []))
                    contact_info = json.dumps(hotel.get('contact_info', {}))
                    business_facilities = json.dumps(hotel.get('business_facilities', []))
                    accessibility = json.dumps(hotel.get('accessibility', []))
                    seasonal_data = json.dumps(hotel.get('seasonal_data', {}))
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO hotels (
                            name, location, address, star_rating, user_rating,
                            min_price, max_price, property_type, amenities, room_types,
                            description, best_season, nearby_attractions, contact_info,
                            sustainability_rating, business_facilities, family_friendly,
                            pet_friendly, accessibility, seasonal_data, collected_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        hotel.get('name'),
                        hotel.get('location'),
                        hotel.get('address'),
                        hotel.get('star_rating'),
                        hotel.get('user_rating'),
                        hotel.get('min_price'),
                        hotel.get('max_price'),
                        hotel.get('property_type'),
                        amenities,
                        room_types,
                        hotel.get('description'),
                        hotel.get('best_season'),
                        nearby_attractions,
                        contact_info,
                        hotel.get('sustainability_rating'),
                        business_facilities,
                        hotel.get('family_friendly'),
                        hotel.get('pet_friendly'),
                        accessibility,
                        seasonal_data,
                        hotel.get('collected_at', datetime.now().isoformat())
                    ))
                    inserted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error inserting hotel {hotel.get('name', 'Unknown')}: {str(e)}")
            
            conn.commit()
            logger.info(f"Inserted {inserted_count} hotels into database")
            return inserted_count
    
    def get_hotels(self, 
                   location: Optional[str] = None,
                   min_rating: Optional[float] = None,
                   max_price: Optional[float] = None,
                   amenities: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve hotels with optional filters"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM hotels WHERE 1=1"
            params = []
            
            if location:
                query += " AND location LIKE ?"
                params.append(f"%{location.lower()}%")
            
            if min_rating:
                query += " AND user_rating >= ?"
                params.append(min_rating)
            
            if max_price:
                query += " AND min_price <= ?"
                params.append(max_price)
            
            if amenities:
                for amenity in amenities:
                    query += " AND amenities LIKE ?"
                    params.append(f"%{amenity.lower()}%")
            
            query += " ORDER BY user_rating DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            hotels = []
            for row in rows:
                hotel = dict(row)
                # Parse JSON fields back to Python objects
                try:
                    hotel['amenities'] = json.loads(hotel['amenities'] or '[]')
                    hotel['room_types'] = json.loads(hotel['room_types'] or '[]')
                    hotel['nearby_attractions'] = json.loads(hotel['nearby_attractions'] or '[]')
                    hotel['contact_info'] = json.loads(hotel['contact_info'] or '{}')
                    hotel['business_facilities'] = json.loads(hotel['business_facilities'] or '[]')
                    hotel['accessibility'] = json.loads(hotel['accessibility'] or '[]')
                    hotel['seasonal_data'] = json.loads(hotel['seasonal_data'] or '{}')
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON for hotel {hotel['name']}")
                
                hotels.append(hotel)
            
            return hotels
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> int:
        """Save user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_preferences (
                    user_id, location, budget_min, budget_max, preferred_season,
                    preferred_amenities, property_type_preference, family_travel, business_travel
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                preferences.get('location'),
                preferences.get('budget_min'),
                preferences.get('budget_max'),
                preferences.get('preferred_season'),
                json.dumps(preferences.get('preferred_amenities', [])),
                preferences.get('property_type_preference'),
                preferences.get('family_travel', False),
                preferences.get('business_travel', False)
            ))
            
            preference_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Saved preferences for user {user_id}")
            return preference_id or 0
    
    def get_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user preferences history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM user_preferences 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            preferences = []
            
            for row in rows:
                pref = dict(row)
                try:
                    pref['preferred_amenities'] = json.loads(pref['preferred_amenities'] or '[]')
                except json.JSONDecodeError:
                    pref['preferred_amenities'] = []
                preferences.append(pref)
            
            return preferences
    
    def save_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]]) -> List[int]:
        """Save recommendation results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            recommendation_ids = []
            for rec in recommendations:
                cursor.execute("""
                    INSERT INTO recommendations (user_id, hotel_id, score, recommendation_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    rec.get('hotel_id'),
                    rec.get('score'),
                    rec.get('recommendation_type', 'general')
                ))
                
                recommendation_ids.append(cursor.lastrowid)
            
            conn.commit()
            logger.info(f"Saved {len(recommendations)} recommendations for user {user_id}")
            return recommendation_ids
    
    def save_user_feedback(self, user_id: str, hotel_id: int, recommendation_id: int, 
                          rating: float, feedback_text: str = "") -> int:
        """Save user feedback on recommendations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_feedback (user_id, hotel_id, recommendation_id, rating, feedback_text)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, hotel_id, recommendation_id, rating, feedback_text))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Saved feedback from user {user_id} for hotel {hotel_id}")
            return feedback_id or 0
    
    def get_all_hotels_df(self) -> pd.DataFrame:
        """Get all hotels as a pandas DataFrame for ML processing"""
        hotels = self.get_hotels()
        if not hotels:
            return pd.DataFrame()
        
        df = pd.DataFrame(hotels)
        logger.info(f"Retrieved {len(df)} hotels as DataFrame")
        return df
    
    def get_user_feedback_df(self) -> pd.DataFrame:
        """Get user feedback as DataFrame for model improvement"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT uf.*, h.name as hotel_name, h.location, h.user_rating as hotel_rating
                FROM user_feedback uf
                JOIN hotels h ON uf.hotel_id = h.id
                ORDER BY uf.created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            return df
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count hotels
            cursor.execute("SELECT COUNT(*) FROM hotels")
            stats['total_hotels'] = cursor.fetchone()[0]
            
            # Count unique locations
            cursor.execute("SELECT COUNT(DISTINCT location) FROM hotels")
            stats['unique_locations'] = cursor.fetchone()[0]
            
            # Count user preferences
            cursor.execute("SELECT COUNT(*) FROM user_preferences")
            stats['user_preferences'] = cursor.fetchone()[0]
            
            # Count recommendations
            cursor.execute("SELECT COUNT(*) FROM recommendations")
            stats['total_recommendations'] = cursor.fetchone()[0]
            
            # Count feedback
            cursor.execute("SELECT COUNT(*) FROM user_feedback")
            stats['total_feedback'] = cursor.fetchone()[0]
            
            return stats
    
    def export_to_csv(self, table_name: str, filepath: str):
        """Export table data to CSV"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {table_name} to {filepath}")
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

# Example usage
if __name__ == "__main__":
    db = DatabaseManager()
    
    # Example hotel data
    sample_hotels = [
        {
            'name': 'Sample Hotel',
            'location': 'paris, france',
            'star_rating': 4.0,
            'user_rating': 4.2,
            'min_price': 150,
            'max_price': 250,
            'amenities': ['wifi', 'pool', 'gym'],
            'family_friendly': True,
            'pet_friendly': False
        }
    ]
    
    # Insert sample data
    db.insert_hotels(sample_hotels)
    
    # Get statistics
    stats = db.get_database_stats()
    print("Database statistics:", stats)
