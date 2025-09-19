#!/usr/bin/env python3
"""
Quick XAMPP MySQL verification for TravelMind
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_xampp_status():
    """Check XAMPP MySQL status"""
    print("🔍 XAMPP MySQL Status Check")
    print("=" * 30)
    
    # Check if MySQL port is open
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 3306))
        sock.close()
        
        if result == 0:
            print("✅ MySQL is running on port 3306")
        else:
            print("❌ MySQL is not running on port 3306")
            print("📋 Please start MySQL in XAMPP Control Panel")
            return False
    except Exception as e:
        print(f"❌ Error checking MySQL: {e}")
        return False
    
    # Check PyMySQL
    try:
        import pymysql
        print("✅ PyMySQL driver is installed")
    except ImportError:
        print("❌ PyMySQL driver not installed")
        print("📦 Run: pip install PyMySQL")
        return False
    
    # Check current database configuration
    try:
        from src.utils.config import get_settings
        settings = get_settings()
        db_url = settings.DATABASE_URL
        print(f"⚙️ Current DATABASE_URL: {db_url}")
        
        if "mysql" not in db_url.lower():
            print("⚠️ DATABASE_URL is not configured for MySQL")
            print("📝 Update your .env file with MySQL connection string")
            return False
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False
    
    # Test database connection
    try:
        from src.data.storage import DatabaseManager
        db = DatabaseManager()
        stats = db.get_database_stats()
        print("✅ Database connection successful!")
        print("📊 Tables found:")
        for table, count in stats.items():
            print(f"   📄 {table}: {count} records")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Common fixes:")
        print("   1. Create 'travelmind_db' database in phpMyAdmin")
        print("   2. Check username/password in .env file")
        print("   3. Ensure MySQL is running in XAMPP")
        return False

def quick_setup_guide():
    """Show quick setup guide"""
    print("\n📋 Quick XAMPP Setup Guide:")
    print("=" * 30)
    print("1. 🚀 Start XAMPP Control Panel")
    print("2. ▶️  Click 'Start' next to MySQL")
    print("3. 🌐 Click 'Admin' next to MySQL (opens phpMyAdmin)")
    print("4. ➕ Create new database: 'travelmind_db'")
    print("5. 📦 Install PyMySQL: pip install PyMySQL")
    print("6. ⚙️  Update .env DATABASE_URL")
    print("7. 🧪 Run this script again to verify")

if __name__ == "__main__":
    print("🎯 TravelMind XAMPP Verification")
    print("=" * 40)
    
    success = check_xampp_status()
    
    if success:
        print("\n🎉 XAMPP MySQL is ready for TravelMind!")
        print("\n🚀 You can now start your system:")
        print("   python -m src.main")
        print("   streamlit run frontend/app.py")
    else:
        quick_setup_guide()
        print("\n🔧 Run 'python setup_xampp.py' for guided setup")
