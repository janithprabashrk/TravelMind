#!/usr/bin/env python3
"""
Universal Database Manager for TravelMind
Supports both SQLite and MySQL/PostgreSQL
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class UniversalDatabaseManager:
    """Database manager that works with SQLite, MySQL, and PostgreSQL"""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize database connection"""
        if db_url is None:
            from ..utils.config import get_settings
            settings = get_settings()
            db_url = settings.DATABASE_URL
        
        self.db_url = db_url
        self.db_type = self._detect_db_type(db_url)
        self.engine = None
        self.Session = None
        self._init_connection()
    
    def _detect_db_type(self, db_url: str) -> str:
        """Detect database type from URL"""
        if db_url.startswith('sqlite'):
            return 'sqlite'
        elif db_url.startswith('mysql') or db_url.startswith('mariadb'):
            return 'mysql'
        elif db_url.startswith('postgresql'):
            return 'postgresql'
        else:
            return 'unknown'
    
    def _init_connection(self):
        """Initialize database connection"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            if self.db_type == 'sqlite':
                self.engine = create_engine(self.db_url, echo=False)
            elif self.db_type == 'mysql':
                self.engine = create_engine(self.db_url, echo=False, pool_pre_ping=True)
            elif self.db_type == 'postgresql':
                self.engine = create_engine(self.db_url, echo=False, pool_pre_ping=True)
            
            self.Session = sessionmaker(bind=self.engine)
            self._create_tables()
            
        except ImportError:
            # Fallback to native SQLite for basic operations
            if self.db_type == 'sqlite':
                self._init_sqlite_fallback()
            else:
                raise Exception(f"SQLAlchemy required for {self.db_type} databases")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _init_sqlite_fallback(self):
        """Fallback SQLite initialization"""
        import sqlite3
        db_path = self.db_url.replace('sqlite:///', '').replace('./', '')
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._create_sqlite_tables()
    
    def _create_tables(self):
        """Create tables using SQLAlchemy"""
        from sqlalchemy import text
        
        if self.engine is None:
            raise Exception("Database engine not initialized")
        
        with self.engine.connect() as conn:
            # Hotels table
            if self.db_type == 'mysql':
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
            else:
                # SQLite/PostgreSQL version
                conn.execute(text("""
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
                """))
            
            # Other tables...
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
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
            """ if self.db_type == 'mysql' else """
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
            """))
            
            conn.commit()
            logger.info(f"Database tables created for {self.db_type}")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables (fallback)"""
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
        
        self.connection.commit()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        if hasattr(self, 'connection'):  # SQLite fallback
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            stats = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count
            
            return stats
        else:  # SQLAlchemy
            if self.engine is None:
                raise Exception("Database engine not initialized")
            
            from sqlalchemy import text
            stats = {}
            
            with self.engine.connect() as conn:
                if self.db_type == 'mysql':
                    result = conn.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in result]
                else:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                    tables = [row[0] for row in result]
                
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats[table] = count
            
            return stats
    
    def insert_hotels(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotel data"""
        if hasattr(self, 'connection'):  # SQLite fallback
            return self._insert_hotels_sqlite(hotels)
        else:  # SQLAlchemy
            return self._insert_hotels_sqlalchemy(hotels)
    
    def _insert_hotels_sqlite(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotels using SQLite"""
        cursor = self.connection.cursor()
        inserted_count = 0
        
        for hotel in hotels:
            try:
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
                    hotel.get('name'), hotel.get('location'), hotel.get('address'),
                    hotel.get('star_rating'), hotel.get('user_rating'),
                    hotel.get('min_price'), hotel.get('max_price'),
                    hotel.get('property_type'), amenities, room_types,
                    hotel.get('description'), hotel.get('best_season'),
                    nearby_attractions, contact_info,
                    hotel.get('sustainability_rating'), business_facilities,
                    hotel.get('family_friendly'), hotel.get('pet_friendly'),
                    accessibility, seasonal_data, hotel.get('collected_at')
                ))
                
                inserted_count += 1
            except Exception as e:
                logger.error(f"Error inserting hotel {hotel.get('name', 'Unknown')}: {e}")
        
        self.connection.commit()
        return inserted_count
    
    def _insert_hotels_sqlalchemy(self, hotels: List[Dict[str, Any]]) -> int:
        """Insert hotels using SQLAlchemy"""
        from sqlalchemy import text
        inserted_count = 0
        
        if self.engine is None:
            raise Exception("Database engine not initialized")
        
        with self.engine.connect() as conn:
            for hotel in hotels:
                try:
                    if self.db_type == 'mysql':
                        # MySQL with JSON columns
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
                                max_price = VALUES(max_price),
                                updated_at = CURRENT_TIMESTAMP
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
