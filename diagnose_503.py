#!/usr/bin/env python3
"""
TravelMind 503 Error Diagnostic Script
This will identify and fix the Service Unavailable error
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")
    
    dependencies = {
        'PyMySQL': 'pymysql',
        'SQLAlchemy': 'sqlalchemy',
        'FastAPI': 'fastapi',
        'Streamlit': 'streamlit',
        'Pandas': 'pandas',
        'Scikit-learn': 'sklearn'
    }
    
    missing = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Missing!")
            missing.append(name)
    
    return missing

def test_mysql_connection():
    """Test MySQL connection specifically"""
    print("\n🔍 Testing MySQL Connection...")
    
    try:
        import pymysql
        print("✅ PyMySQL imported")
        
        # Test direct connection
        connection = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='',
            database='travelmind_db'
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result and result[0] == 1:
            print("✅ Direct MySQL connection successful")
            return True
        else:
            print("❌ MySQL connection test failed")
            return False
            
    except ImportError:
        print("❌ PyMySQL not installed")
        return False
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        print("💡 Common fixes:")
        print("   1. Start MySQL in XAMPP Control Panel")
        print("   2. Create 'travelmind_db' database in phpMyAdmin")
        print("   3. Check if root user has no password")
        return False

def test_database_manager():
    """Test the database manager initialization"""
    print("\n🔍 Testing Database Manager...")
    
    try:
        # Test current DatabaseManager
        from src.data.storage import DatabaseManager
        db = DatabaseManager()
        print("✅ DatabaseManager imported and initialized")
        
        stats = db.get_database_stats()
        print(f"✅ Database stats retrieved: {stats}")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager failed: {e}")
        print("\n🔧 Trying UniversalDatabaseManager...")
        
        try:
            from src.data.mysql_storage import UniversalDatabaseManager
            db = UniversalDatabaseManager()
            print("✅ UniversalDatabaseManager initialized")
            
            stats = db.get_database_stats()
            print(f"✅ Database stats: {stats}")
            return True
            
        except Exception as e2:
            print(f"❌ UniversalDatabaseManager also failed: {e2}")
            traceback.print_exc()
            return False

def test_recommendation_system():
    """Test the recommendation system"""
    print("\n🔍 Testing Recommendation System...")
    
    try:
        from src.models.recommender import HotelRecommendationSystem
        recommender = HotelRecommendationSystem()
        print("✅ Recommendation system initialized")
        
        # Test with minimal preferences
        user_preferences = {
            "budget_range": (100, 300),
            "preferred_amenities": ["wifi"],
            "travel_purpose": "leisure"
        }
        
        recommendations = recommender.get_recommendations(
            location="Paris",
            user_preferences=user_preferences,
            algorithm="hybrid"
        )
        
        print(f"✅ Recommendations retrieved: {len(recommendations)} results")
        return True
        
    except Exception as e:
        print(f"❌ Recommendation system failed: {e}")
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoint availability"""
    print("\n🔍 Testing API Components...")
    
    try:
        from src.main import app
        print("✅ FastAPI app imported")
        
        # Check if we can access the app
        if hasattr(app, 'routes'):
            routes = [route.path for route in app.routes]
            recommendation_routes = [r for r in routes if 'recommendation' in r]
            print(f"✅ Found {len(recommendation_routes)} recommendation routes")
            print(f"📋 Routes: {recommendation_routes}")
        
        return True
        
    except Exception as e:
        print(f"❌ API components failed: {e}")
        traceback.print_exc()
        return False

def suggest_fixes(missing_deps, mysql_ok, db_ok, rec_ok, api_ok):
    """Suggest fixes based on test results"""
    print("\n🔧 Suggested Fixes:")
    print("=" * 30)
    
    if missing_deps:
        print(f"📦 Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    if not mysql_ok:
        print("🐬 MySQL Issues:")
        print("   1. Start MySQL in XAMPP Control Panel")
        print("   2. Open phpMyAdmin: http://localhost/phpmyadmin")
        print("   3. Create database: 'travelmind_db'")
        print("   4. Install PyMySQL: pip install PyMySQL")
    
    if not db_ok:
        print("🗄️ Database Manager Issues:")
        print("   1. Update storage.py to use UniversalDatabaseManager")
        print("   2. Install SQLAlchemy: pip install SQLAlchemy")
        print("   3. Check database URL in .env file")
    
    if not rec_ok:
        print("🤖 Recommendation System Issues:")
        print("   1. Ensure models are trained: python train.py")
        print("   2. Check model files exist in ./models/")
        print("   3. Add sample hotel data to database")
    
    if not api_ok:
        print("🚀 API Issues:")
        print("   1. Check FastAPI installation")
        print("   2. Verify all imports in main.py")
        print("   3. Check for circular import issues")

def auto_fix_issues():
    """Attempt to automatically fix common issues"""
    print("\n🔧 Attempting Automatic Fixes...")
    
    # Fix 1: Install PyMySQL if missing
    try:
        import pymysql
    except ImportError:
        print("📦 Installing PyMySQL...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMySQL"])
        print("✅ PyMySQL installed")
    
    # Fix 2: Update storage.py to use correct database manager
    storage_file = Path("src/data/storage.py")
    if storage_file.exists():
        print("🔄 Updating storage.py to use MySQL...")
        
        # Create a backup
        backup_file = storage_file.with_suffix('.py.backup')
        import shutil
        shutil.copy2(storage_file, backup_file)
        
        # Replace imports in files that use DatabaseManager
        fix_database_imports()
    
    # Fix 3: Ensure models exist
    models_dir = Path("models")
    if not models_dir.exists() or len(list(models_dir.glob("*.pkl"))) < 6:
        print("🤖 Training models...")
        import subprocess
        subprocess.run([sys.executable, "train.py"])

def fix_database_imports():
    """Fix database imports to use UniversalDatabaseManager"""
    print("🔄 Fixing database imports...")
    
    files_to_update = [
        "src/models/recommender.py",
        "src/data/collector.py",
        "src/main.py"
    ]
    
    for file_path in files_to_update:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    content = f.read()
                
                # Replace imports
                old_import = "from ..data.storage import DatabaseManager"
                new_import = "from ..data.mysql_storage import UniversalDatabaseManager as DatabaseManager"
                
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    
                    with open(path, 'w') as f:
                        f.write(content)
                    
                    print(f"✅ Updated {file_path}")
                
            except Exception as e:
                print(f"❌ Failed to update {file_path}: {e}")

def main():
    """Main diagnostic function"""
    print("🚨 TravelMind 503 Error Diagnostic")
    print("=" * 40)
    
    # Run tests
    missing_deps = test_dependencies()
    mysql_ok = test_mysql_connection()
    db_ok = test_database_manager()
    rec_ok = test_recommendation_system()
    api_ok = test_api_endpoints()
    
    # Summary
    print("\n📊 Diagnostic Summary:")
    print("=" * 25)
    print(f"Dependencies: {'✅ OK' if not missing_deps else '❌ Issues'}")
    print(f"MySQL: {'✅ OK' if mysql_ok else '❌ Issues'}")
    print(f"Database: {'✅ OK' if db_ok else '❌ Issues'}")
    print(f"Recommendations: {'✅ OK' if rec_ok else '❌ Issues'}")
    print(f"API: {'✅ OK' if api_ok else '❌ Issues'}")
    
    # Suggest fixes
    suggest_fixes(missing_deps, mysql_ok, db_ok, rec_ok, api_ok)
    
    # Ask for auto-fix
    if not all([mysql_ok, db_ok, rec_ok, api_ok]):
        print("\n🔧 Attempt automatic fixes? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            auto_fix_issues()
            print("\n✅ Auto-fixes applied. Please restart your API server.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Diagnostic cancelled")
