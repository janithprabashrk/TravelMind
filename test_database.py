#!/usr/bin/env python3
"""
Simple database test script
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_database():
    print("🗄️ Testing TravelMind Database...")
    
    try:
        from src.data.storage import DatabaseManager
        
        # Initialize database
        print("📋 Initializing database...")
        db = DatabaseManager()
        
        # Get stats
        print("📊 Getting database statistics...")
        stats = db.get_database_stats()
        
        print("✅ Database is working!")
        print(f"📄 Database file: {db.db_path}")
        print("📊 Tables and record counts:")
        
        for table, count in stats.items():
            print(f"   📄 {table}: {count} records")
        
        # Test inserting a sample hotel
        print("\n🏨 Testing hotel insertion...")
        sample_hotels = [{
            'name': 'Test Hotel',
            'location': 'Test City',
            'address': '123 Test Street',
            'star_rating': 4.0,
            'user_rating': 4.2,
            'min_price': 100.0,
            'max_price': 200.0,
            'property_type': 'Hotel',
            'amenities': ['wifi', 'pool'],
            'room_types': ['standard', 'deluxe'],
            'description': 'A test hotel',
            'best_season': 'summer',
            'nearby_attractions': ['Beach', 'Museum'],
            'contact_info': {'phone': '123-456-7890'},
            'sustainability_rating': 3.5,
            'business_facilities': ['conference room'],
            'family_friendly': True,
            'pet_friendly': False,
            'accessibility': ['wheelchair accessible'],
            'seasonal_data': {'summer': {'price_multiplier': 1.2}},
            'collected_at': '2025-07-15 12:00:00'
        }]
        
        inserted = db.insert_hotels(sample_hotels)
        print(f"✅ Inserted {inserted} test hotel(s)")
        
        # Get updated stats
        updated_stats = db.get_database_stats()
        print("\n📊 Updated statistics:")
        for table, count in updated_stats.items():
            print(f"   📄 {table}: {count} records")
        
        print("\n🎉 Database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)
