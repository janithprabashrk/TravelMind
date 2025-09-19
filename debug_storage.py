#!/usr/bin/env python3
"""Debug script to test storage module import"""

print("🔍 Testing storage module import...")

try:
    from src.data.storage import DatabaseManager
    print("✅ Storage module imported successfully!")
    
    # Test creating a database manager
    print("🔍 Testing DatabaseManager initialization...")
    db = DatabaseManager()
    print(f"✅ DatabaseManager created with type: {db.db_type}")
    
    # Test getting stats
    print("🔍 Testing database stats...")
    stats = db.get_database_stats()
    print(f"✅ Database stats: {stats}")
    
    print("🎉 All storage tests passed!")
    
except Exception as e:
    print(f"❌ Storage test failed: {e}")
    import traceback
    traceback.print_exc()
