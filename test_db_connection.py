#!/usr/bin/env python3
"""
Quick test to check if our database connection is working
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)

try:
    # Test simple MySQL connection first
    import pymysql
    import sqlalchemy
    from sqlalchemy import create_engine, text
    
    # Try connecting to MySQL
    DATABASE_URL = "mysql+pymysql://root@localhost:3306/travelmind_db"
    print(f"Testing connection to: {DATABASE_URL}")
    
    engine = create_engine(DATABASE_URL, echo=False)
    
    with engine.connect() as conn:
        # Test connection
        result = conn.execute(text("SELECT 1"))
        print("✅ MySQL connection successful!")
        
        # Create database if it doesn't exist
        try:
            conn.execute(text("CREATE DATABASE IF NOT EXISTS travelmind_db"))
            print("✅ Database 'travelmind_db' ready")
        except Exception as e:
            print(f"⚠️  Database creation note: {e}")
        
        # Test basic table creation
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✅ Test table creation successful")
        
        # Test insert
        conn.execute(text("INSERT INTO test_table (name) VALUES ('test_entry')"))
        
        # Test select
        result = conn.execute(text("SELECT COUNT(*) FROM test_table"))
        count = result.scalar()
        print(f"✅ Test table has {count} entries")
        
        # Clean up test table
        conn.execute(text("DROP TABLE test_table"))
        
        conn.commit()
        print("✅ All database tests passed!")

except Exception as e:
    print(f"❌ Database test failed: {e}")
    print("\nTrying SQLite fallback...")
    
    # Test SQLite fallback
    try:
        import sqlite3
        db_path = "travelmind_test.db"
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("INSERT INTO test_table (name) VALUES ('test_entry')")
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        print(f"✅ SQLite fallback working - {count} entries")
        
        cursor.execute("DROP TABLE test_table")
        connection.commit()
        connection.close()
        
        # Clean up test file
        os.remove(db_path)
        print("✅ SQLite fallback tests passed!")
        
    except Exception as sqlite_error:
        print(f"❌ SQLite fallback also failed: {sqlite_error}")
