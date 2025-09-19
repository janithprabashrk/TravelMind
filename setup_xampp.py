#!/usr/bin/env python3
"""
XAMPP MySQL Setup Script for TravelMind
This script will help you configure TravelMind to work with XAMPP's MySQL
"""

import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_xampp_running():
    """Check if XAMPP MySQL is running"""
    print("🔍 Checking XAMPP MySQL status...")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 3306))
        sock.close()
        
        if result == 0:
            print("✅ MySQL is running on port 3306")
            return True
        else:
            print("❌ MySQL is not running on port 3306")
            return False
    except Exception as e:
        print(f"❌ Error checking MySQL: {e}")
        return False

def install_mysql_driver():
    """Install PyMySQL driver"""
    print("📦 Installing PyMySQL driver...")
    
    try:
        import pymysql
        print("✅ PyMySQL already installed")
        return True
    except ImportError:
        print("📥 Installing PyMySQL...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMySQL"])
            print("✅ PyMySQL installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyMySQL: {e}")
            return False

def test_mysql_connection(db_url):
    """Test MySQL connection"""
    print("🔍 Testing MySQL connection...")
    
    try:
        # Update .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace DATABASE_URL
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('DATABASE_URL='):
                    new_lines.append(f'DATABASE_URL={db_url}')
                else:
                    new_lines.append(line)
            
            with open(env_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print(f"✅ Updated .env with: {db_url}")
        
        # Test connection
        from src.data.storage import DatabaseManager
        db = DatabaseManager()
        
        # This will create tables if they don't exist
        stats = db.get_database_stats()
        
        print("✅ MySQL connection successful!")
        print("📊 Database tables created:")
        for table, count in stats.items():
            print(f"   📄 {table}: {count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Main XAMPP setup function"""
    print("🎯 TravelMind XAMPP MySQL Setup")
    print("=" * 40)
    
    print("\n📋 Before starting, make sure:")
    print("1. XAMPP is installed")
    print("2. XAMPP Control Panel is open")
    print("3. MySQL service is started in XAMPP")
    print("4. You have created 'travelmind_db' database in phpMyAdmin")
    
    input("\nPress Enter when ready...")
    
    # Step 1: Check if MySQL is running
    if not check_xampp_running():
        print("\n❌ XAMPP MySQL is not running!")
        print("📋 Please:")
        print("1. Open XAMPP Control Panel")
        print("2. Click 'Start' next to MySQL")
        print("3. Wait for it to turn green")
        print("4. Run this script again")
        return False
    
    # Step 2: Install MySQL driver
    if not install_mysql_driver():
        return False
    
    # Step 3: Choose database configuration
    print("\n🔧 Database Configuration Options:")
    print("1. Use root user (no password) - Quick setup")
    print("2. Use root user with password")
    print("3. Use custom user (more secure)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        db_url = "mysql+pymysql://root@localhost:3306/travelmind_db"
        print("📝 Using root user with no password")
        
    elif choice == "2":
        password = input("Enter root password: ").strip()
        db_url = f"mysql+pymysql://root:{password}@localhost:3306/travelmind_db"
        print("📝 Using root user with password")
        
    elif choice == "3":
        username = input("Enter username: ").strip()
        password = input("Enter password: ").strip()
        db_url = f"mysql+pymysql://{username}:{password}@localhost:3306/travelmind_db"
        print(f"📝 Using custom user: {username}")
        
    else:
        print("❌ Invalid choice")
        return False
    
    # Step 4: Test connection
    success = test_mysql_connection(db_url)
    
    if success:
        print("\n🎉 XAMPP MySQL setup complete!")
        print("\n📋 Next steps:")
        print("1. python -m src.main  # Start API")
        print("2. streamlit run frontend/app.py  # Start web interface")
        print("\n💡 Database URL in .env:")
        print(f"   DATABASE_URL={db_url}")
        
        # Test with sample data
        print("\n🧪 Want to add sample hotel data? (y/n)")
        add_sample = input().strip().lower()
        
        if add_sample == 'y':
            add_sample_data()
        
        return True
    else:
        print("\n❌ Setup failed. Common issues:")
        print("1. Database 'travelmind_db' doesn't exist - create it in phpMyAdmin")
        print("2. Wrong username/password")
        print("3. MySQL not running in XAMPP")
        return False

def add_sample_data():
    """Add sample hotel data"""
    print("🏨 Adding sample hotel data...")
    
    try:
        from src.data.storage import DatabaseManager
        
        db = DatabaseManager()
        
        sample_hotels = [
            {
                'name': 'Grand Palace Hotel',
                'location': 'Paris, France',
                'address': '123 Champs-Élysées',
                'star_rating': 5.0,
                'user_rating': 4.8,
                'min_price': 250.0,
                'max_price': 500.0,
                'property_type': 'Luxury Hotel',
                'amenities': ['wifi', 'spa', 'pool', 'restaurant', 'gym'],
                'room_types': ['standard', 'deluxe', 'suite'],
                'description': 'Luxury hotel in the heart of Paris',
                'best_season': 'spring',
                'nearby_attractions': ['Eiffel Tower', 'Louvre Museum'],
                'contact_info': {'phone': '+33-1-2345-6789', 'email': 'info@grandpalace.fr'},
                'sustainability_rating': 4.2,
                'business_facilities': ['conference room', 'business center'],
                'family_friendly': True,
                'pet_friendly': False,
                'accessibility': ['wheelchair accessible', 'elevator'],
                'seasonal_data': {'spring': {'price_multiplier': 1.2}, 'summer': {'price_multiplier': 1.5}},
                'collected_at': '2025-07-15 12:00:00'
            },
            {
                'name': 'Budget Traveler Inn',
                'location': 'Paris, France',
                'address': '456 Rue de la Paix',
                'star_rating': 2.0,
                'user_rating': 3.8,
                'min_price': 50.0,
                'max_price': 100.0,
                'property_type': 'Budget Hotel',
                'amenities': ['wifi', 'breakfast'],
                'room_types': ['standard', 'twin'],
                'description': 'Affordable accommodation for budget travelers',
                'best_season': 'all',
                'nearby_attractions': ['Metro Station', 'Local Market'],
                'contact_info': {'phone': '+33-1-9876-5432'},
                'sustainability_rating': 3.0,
                'business_facilities': [],
                'family_friendly': True,
                'pet_friendly': True,
                'accessibility': ['ground floor rooms'],
                'seasonal_data': {'winter': {'price_multiplier': 0.8}},
                'collected_at': '2025-07-15 12:00:00'
            }
        ]
        
        inserted = db.insert_hotels(sample_hotels)
        print(f"✅ Added {inserted} sample hotels!")
        
        # Show updated stats
        stats = db.get_database_stats()
        print("📊 Updated database:")
        for table, count in stats.items():
            print(f"   📄 {table}: {count} records")
            
    except Exception as e:
        print(f"❌ Failed to add sample data: {e}")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
