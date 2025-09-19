#!/usr/bin/env python3
"""
TravelMind Database Setup Script
This script will initialize and configure your database
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_settings
from src.data.storage import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_sqlite_database():
    """Setup SQLite database (recommended for development)"""
    print("🗄️ Setting up SQLite Database...")
    
    settings = get_settings()
    db_path = Path("travelmind.db")
    
    print(f"📍 Database will be created at: {db_path.absolute()}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Database is automatically initialized in __init__
        print("✅ SQLite database setup complete!")
        print(f"📊 Database file: {db_path.absolute()}")
        
        # Test connection by getting stats
        stats = db_manager.get_database_stats()
        print("✅ Database connection test successful!")
        print(f"📊 Database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        logger.error(f"Database setup error: {e}")
        return False

def setup_postgresql_database():
    """Setup PostgreSQL database (for production)"""
    print("🐘 PostgreSQL Database Setup Instructions")
    print("=" * 50)
    
    print("""
📋 Prerequisites:
1. Install PostgreSQL server
2. Create a database and user
3. Install psycopg2: pip install psycopg2-binary

🔧 Setup Steps:

1. Install PostgreSQL:
   - Windows: Download from https://www.postgresql.org/download/windows/
   - Run installer and note the password for 'postgres' user

2. Create database and user:
   - Open pgAdmin or use command line
   - Connect as 'postgres' user
   - Run these commands:
   
   CREATE DATABASE travelmind_db;
   CREATE USER travelmind_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE travelmind_db TO travelmind_user;

3. Update your .env file:
   DATABASE_URL=postgresql://travelmind_user:your_secure_password@localhost:5432/travelmind_db

4. Install Python driver:
   pip install psycopg2-binary

5. Run this script again to create tables
""")

def setup_mysql_database():
    """Setup MySQL database (alternative option)"""
    print("🐬 MySQL Database Setup Instructions")
    print("=" * 50)
    
    print("""
📋 Prerequisites:
1. Install MySQL server
2. Create a database and user
3. Install PyMySQL: pip install PyMySQL

🔧 Setup Steps:

1. Install MySQL:
   - Windows: Download from https://dev.mysql.com/downloads/mysql/
   - Run installer and set root password

2. Create database and user:
   - Open MySQL Workbench or command line
   - Connect as 'root' user
   - Run these commands:
   
   CREATE DATABASE travelmind_db;
   CREATE USER 'travelmind_user'@'localhost' IDENTIFIED BY 'your_secure_password';
   GRANT ALL PRIVILEGES ON travelmind_db.* TO 'travelmind_user'@'localhost';
   FLUSH PRIVILEGES;

3. Update your .env file:
   DATABASE_URL=mysql+pymysql://travelmind_user:your_secure_password@localhost:3306/travelmind_db

4. Install Python driver:
   pip install PyMySQL

5. Run this script again to create tables
""")

def test_database_connection():
    """Test the current database connection"""
    print("🔍 Testing Database Connection...")
    
    try:
        db_manager = DatabaseManager()
        
        # Test by getting database stats
        stats = db_manager.get_database_stats()
        print("✅ Database connection successful!")
        print(f"📊 Database Statistics:")
        for table, count in stats.items():
            print(f"   📄 {table}: {count} records")
        
        return True
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def show_current_config():
    """Show current database configuration"""
    print("⚙️ Current Database Configuration")
    print("=" * 40)
    
    settings = get_settings()
    db_url = settings.DATABASE_URL
    
    print(f"📍 Database URL: {db_url}")
    
    if db_url.startswith("sqlite"):
        db_file = db_url.replace("sqlite:///", "").replace("./", "")
        db_path = Path(db_file)
        print(f"📁 Database Type: SQLite")
        print(f"📄 Database File: {db_path.absolute()}")
        print(f"💾 File Exists: {'Yes' if db_path.exists() else 'No'}")
        if db_path.exists():
            size = db_path.stat().st_size
            print(f"📊 File Size: {size} bytes")
    elif db_url.startswith("postgresql"):
        print(f"📁 Database Type: PostgreSQL")
        print(f"🔗 Connection String: {db_url}")
    elif db_url.startswith("mysql"):
        print(f"📁 Database Type: MySQL")
        print(f"🔗 Connection String: {db_url}")
    else:
        print(f"❓ Unknown database type")

def backup_sqlite_database():
    """Create a backup of the SQLite database"""
    print("💾 Creating Database Backup...")
    
    db_path = Path("travelmind.db")
    if not db_path.exists():
        print("❌ Database file not found")
        return False
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"travelmind_backup_{timestamp}.db")
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def main():
    """Main database setup function"""
    print("🗄️ TravelMind Database Configuration")
    print("=" * 50)
    
    while True:
        print("\n🔧 Database Setup Options:")
        print("1. Show current configuration")
        print("2. Setup SQLite database (recommended for development)")
        print("3. Show PostgreSQL setup instructions (for production)")
        print("4. Show MySQL setup instructions (alternative)")
        print("5. Test database connection")
        print("6. Create SQLite backup")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            show_current_config()
            
        elif choice == "2":
            success = setup_sqlite_database()
            if success:
                print("\n✅ SQLite setup complete!")
                print("💡 Your .env file is already configured for SQLite")
                
        elif choice == "3":
            setup_postgresql_database()
            
        elif choice == "4":
            setup_mysql_database()
            
        elif choice == "5":
            test_database_connection()
            
        elif choice == "6":
            backup_sqlite_database()
            
        elif choice == "7":
            print("👋 Database setup complete!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
