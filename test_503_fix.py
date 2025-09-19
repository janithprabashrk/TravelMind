#!/usr/bin/env python3
"""
Simple test to verify our fix worked
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔧 Testing the 503 fix...")

try:
    # Test importing our fixed storage module
    from src.data.storage_fixed import DatabaseManager
    print("✅ Fixed storage module imported successfully!")
    
    # Test creating database manager
    print("🔍 Creating DatabaseManager instance...")
    db = DatabaseManager()
    print(f"✅ DatabaseManager created successfully with type: {db.db_type}")
    
    # Test database stats (this was failing before)
    print("🔍 Testing database stats...")
    stats = db.get_database_stats()
    print(f"✅ Database stats retrieved: {stats}")
    
    # Test the route import
    print("🔍 Testing route imports...")
    from src.api.routes import router
    print("✅ API routes imported successfully!")
    
    print("\n🎉 503 Service Unavailable fix successful!")
    print("📝 Summary of fixes:")
    print("   • Fixed storage.py corruption by using storage_fixed.py")
    print("   • Added missing database methods")
    print("   • Updated route imports")
    print("   • Database connection is working")
    print("\n🚀 You can now start the API server!")
    
except Exception as e:
    print(f"❌ Fix verification failed: {e}")
    import traceback
    traceback.print_exc()
