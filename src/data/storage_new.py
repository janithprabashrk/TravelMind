import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

try:
    from ..utils.config import get_settings
except ImportError:
    # Fallback if config import fails
    class MockSettings:
        DATABASE_URL = "sqlite:///travelmind.db"
    def get_settings():
        return MockSettings()

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simple database manager for TravelMind"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection"""
        try:
            settings = get_settings()
            self.db_url = settings.DATABASE_URL
        except:
            self.db_url = "sqlite:///travelmind.db"
        
        if 'mysql' in self.db_url.lower():
            self._init_mysql()
        else:
            self._init_sqlite()
    
    def _init_mysql(self):
        """Initialize MySQL connection"""
        try:
            import pymysql
            from sqlalchemy import create_engine, text
            
            self.engine = create_engine(self.db_url, echo=False)
            self.db_type = 'mysql'
            self._create_mysql_tables()
            logger.info("MySQL database initialized")
            
        except Exception as e:
            logger.error(f"MySQL initialization failed: {e}, falling back to SQLite")
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite connection"""
        db_file = "travelmind.db"
        self.db_path = Path(db_file)
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db_type = 'sqlite'
        self._create_sqlite_tables()
        logger.info(f"SQLite database initialized: {self.db_path}")
    
    def _create_mysql_tables(self):
        """Create MySQL tables"""
        from sqlalchemy import text
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hotels (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    location VARCHAR(255) NOT NULL,
                    address TEXT,
                    star_rating DECIMAL(2,1),
                    user_rating DECIMAL(2,1),
                    min_price DECIMAL(10,2),
                    max_price DECIMAL(10,2),
                    property_type VARCHAR(100),
                    amenities JSON,
                    room_types JSON,
                    description TEXT,
                    best_season VARCHAR(50),
                    nearby_attractions JSON,
                    contact_info JSON,
                    sustainability_rating DECIMAL(2,1),
                    business_facilities JSON,
                    family_friendly BOOLEAN,
                    pet_friendly BOOLEAN,
                    accessibility JSON,
                    seasonal_data JSON,
                    collected_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_hotel (name, location)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(255),
                    location VARCHAR(255),
                    budget_min DECIMAL(10,2),
                    budget_max DECIMAL(10,2),
                    preferred_season VARCHAR(50),
                    preferred_amenities TEXT,
                    property_type_preference VARCHAR(100),
                    family_travel BOOLEAN,
                    business_travel BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            conn.commit()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        cursor = self.connection.cursor()
        
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
                amenities TEXT,
                room_types TEXT,
                description TEXT,
                best_season TEXT,
                nearby_attractions TEXT,
                contact_info TEXT,
                sustainability_rating REAL,
                business_facilities TEXT,
                family_friendly BOOLEAN,
                pet_friendly BOOLEAN,
                accessibility TEXT,
                seasonal_data TEXT,
                collected_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, location)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                location TEXT,
                budget_min REAL,
                budget_max REAL,
                preferred_season TEXT,
                preferred_amenities TEXT,
                property_type_preference TEXT,
                family_travel BOOLEAN,
                business_travel BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    def get_hotels(self, location: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
        """Get hotels from database"""
        if self.db_type == 'mysql':
            return self._get_hotels_mysql(location, **filters)
        else:
            return self._get_hotels_sqlite(location, **filters)
    
    def _get_hotels_sqlite(self, location: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
        """Get hotels from SQLite"""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM hotels"
        params = []
        
        if location:
            query += " WHERE location LIKE ?"
            params.append(f"%{location}%")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        hotels = []
        
        for row in rows:
            hotel = dict(zip(columns, row))
            # Parse JSON fields
            for field in ['amenities', 'room_types', 'nearby_attractions', 'contact_info', 
                         'business_facilities', 'accessibility', 'seasonal_data']:
                if hotel.get(field):
                    try:
                        hotel[field] = json.loads(hotel[field])
                    except:
                        hotel[field] = []
            hotels.append(hotel)
        
        return hotels
    
    def _get_hotels_mysql(self, location: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
        """Get hotels from MySQL"""
        from sqlalchemy import text
        
        query = "SELECT * FROM hotels"
        params = {}
        
        if location:
            query += " WHERE location LIKE :location"
            params['location'] = f"%{location}%"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()
            
            hotels = []
            for row in rows:
                hotel = dict(zip(columns, row))
                hotels.append(hotel)
        
        return hotels
    
    def get_all_hotels_df(self) -> pd.DataFrame:
        """Get all hotels as pandas DataFrame"""
        hotels = self.get_hotels()
        return pd.DataFrame(hotels)
    
    def insert_hotels(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotel data"""
        if self.db_type == 'mysql':
            return self._insert_hotels_mysql(hotels)
        else:
            return self._insert_hotels_sqlite(hotels)
    
    def _insert_hotels_sqlite(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotels into SQLite"""
        cursor = self.connection.cursor()
        inserted_count = 0
        
        for hotel in hotels:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO hotels (
                        name, location, address, star_rating, user_rating,
                        min_price, max_price, property_type, amenities, room_types,
                        description, best_season, nearby_attractions, contact_info,
                        sustainability_rating, business_facilities, family_friendly,
                        pet_friendly, accessibility, seasonal_data, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hotel.get('name'), hotel.get('location'), hotel.get('address'),
                    hotel.get('star_rating'), hotel.get('user_rating'),
                    hotel.get('min_price'), hotel.get('max_price'),
                    hotel.get('property_type'),
                    json.dumps(hotel.get('amenities', [])),
                    json.dumps(hotel.get('room_types', [])),
                    hotel.get('description'), hotel.get('best_season'),
                    json.dumps(hotel.get('nearby_attractions', [])),
                    json.dumps(hotel.get('contact_info', {})),
                    hotel.get('sustainability_rating'),
                    json.dumps(hotel.get('business_facilities', [])),
                    hotel.get('family_friendly'), hotel.get('pet_friendly'),
                    json.dumps(hotel.get('accessibility', [])),
                    json.dumps(hotel.get('seasonal_data', {})),
                    hotel.get('collected_at')
                ))
                inserted_count += 1
            except Exception as e:
                logger.error(f"Error inserting hotel {hotel.get('name', 'Unknown')}: {e}")
        
        self.connection.commit()
        return inserted_count
    
    def _insert_hotels_mysql(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotels into MySQL"""
        from sqlalchemy import text
        inserted_count = 0
        
        with self.engine.connect() as conn:
            for hotel in hotels:
                try:
                    conn.execute(text("""
                        INSERT INTO hotels (
                            name, location, address, star_rating, user_rating,
                            min_price, max_price, property_type, amenities, room_types,
                            description, best_season, nearby_attractions, contact_info,
                            sustainability_rating, business_facilities, family_friendly,
                            pet_friendly, accessibility, seasonal_data, collected_at
                        ) VALUES (
                            :name, :location, :address, :star_rating, :user_rating,
                            :min_price, :max_price, :property_type, :amenities, :room_types,
                            :description, :best_season, :nearby_attractions, :contact_info,
                            :sustainability_rating, :business_facilities, :family_friendly,
                            :pet_friendly, :accessibility, :seasonal_data, :collected_at
                        ) ON DUPLICATE KEY UPDATE
                            user_rating = VALUES(user_rating),
                            min_price = VALUES(min_price),
                            max_price = VALUES(max_price)
                    """), {
                        'name': hotel.get('name'),
                        'location': hotel.get('location'),
                        'address': hotel.get('address'),
                        'star_rating': hotel.get('star_rating'),
                        'user_rating': hotel.get('user_rating'),
                        'min_price': hotel.get('min_price'),
                        'max_price': hotel.get('max_price'),
                        'property_type': hotel.get('property_type'),
                        'amenities': json.dumps(hotel.get('amenities', [])),
                        'room_types': json.dumps(hotel.get('room_types', [])),
                        'description': hotel.get('description'),
                        'best_season': hotel.get('best_season'),
                        'nearby_attractions': json.dumps(hotel.get('nearby_attractions', [])),
                        'contact_info': json.dumps(hotel.get('contact_info', {})),
                        'sustainability_rating': hotel.get('sustainability_rating'),
                        'business_facilities': json.dumps(hotel.get('business_facilities', [])),
                        'family_friendly': hotel.get('family_friendly'),
                        'pet_friendly': hotel.get('pet_friendly'),
                        'accessibility': json.dumps(hotel.get('accessibility', [])),
                        'seasonal_data': json.dumps(hotel.get('seasonal_data', {})),
                        'collected_at': hotel.get('collected_at')
                    })
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting hotel {hotel.get('name', 'Unknown')}: {e}")
            
            conn.commit()
        
        return inserted_count

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        if self.db_type == 'mysql':
            return self._get_mysql_stats()
        else:
            return self._get_sqlite_stats()
    
    def _get_mysql_stats(self) -> Dict[str, int]:
        """Get MySQL database stats"""
        from sqlalchemy import text
        stats = {}
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result]
                
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats[table] = count
                
                stats['total_hotels'] = stats.get('hotels', 0)
        except Exception as e:
            logger.error(f"Error getting MySQL stats: {e}")
            stats = {'total_hotels': 0}
        
        return stats
    
    def _get_sqlite_stats(self) -> Dict[str, int]:
        """Get SQLite database stats"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        stats = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[table] = count
        
        stats['total_hotels'] = stats.get('hotels', 0)
        return stats

    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> int:
        """Save user preferences"""
        if self.db_type == 'mysql':
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO user_preferences (
                        user_id, location, budget_min, budget_max, preferred_season,
                        preferred_amenities, property_type_preference, family_travel, business_travel
                    ) VALUES (
                        :user_id, :location, :budget_min, :budget_max, :preferred_season,
                        :preferred_amenities, :property_type_preference, :family_travel, :business_travel
                    )
                """), {
                    'user_id': user_id,
                    'location': preferences.get('location'),
                    'budget_min': preferences.get('budget_range', [0, 1000])[0],
                    'budget_max': preferences.get('budget_range', [0, 1000])[1],
                    'preferred_season': preferences.get('preferred_season'),
                    'preferred_amenities': json.dumps(preferences.get('preferred_amenities', [])),
                    'property_type_preference': preferences.get('property_type_preference'),
                    'family_travel': preferences.get('family_travel', False),
                    'business_travel': preferences.get('business_travel', False)
                })
                conn.commit()
                return result.lastrowid or 0
        else:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO user_preferences (
                    user_id, location, budget_min, budget_max, preferred_season,
                    preferred_amenities, property_type_preference, family_travel, business_travel
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, preferences.get('location'),
                preferences.get('budget_range', [0, 1000])[0],
                preferences.get('budget_range', [0, 1000])[1],
                preferences.get('preferred_season'),
                json.dumps(preferences.get('preferred_amenities', [])),
                preferences.get('property_type_preference'),
                preferences.get('family_travel', False),
                preferences.get('business_travel', False)
            ))
            self.connection.commit()
            return cursor.lastrowid or 0

    def get_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user preferences"""
        if self.db_type == 'mysql':
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM user_preferences WHERE user_id = :user_id"), 
                                    {'user_id': user_id})
                return [dict(row._mapping) for row in result]
        else:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
