# 🗄️ TravelMind Database Configuration Guide

Your TravelMind system supports multiple database options. Here's how to configure each:

## 📋 Current Configuration (Recommended for Development)

Your `.env` file currently has:
```
DATABASE_URL=sqlite:///./travelmind.db
```

This is **perfect for development and testing**! SQLite requires no additional setup.

## 🚀 Quick Setup (SQLite)

**Option 1: Run setup script**
```bash
python setup_database.py
```
Then choose option 2.

**Option 2: Manual initialization**
```bash
python -c "from src.data.storage import DatabaseManager; db = DatabaseManager(); print('Database initialized!')"
```

## 🏢 Production Database Options

### 🐘 PostgreSQL (Recommended for Production)

1. **Install PostgreSQL**
   - Download: https://www.postgresql.org/download/
   - Windows: Use the installer
   - Note the password for 'postgres' user

2. **Create Database**
   ```sql
   -- Connect to PostgreSQL as postgres user
   CREATE DATABASE travelmind_db;
   CREATE USER travelmind_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE travelmind_db TO travelmind_user;
   ```

3. **Install Python Driver**
   ```bash
   pip install psycopg2-binary
   ```

4. **Update .env file**
   ```env
   DATABASE_URL=postgresql://travelmind_user:your_secure_password@localhost:5432/travelmind_db
   ```

### 🐬 MySQL (Alternative)

1. **Install MySQL**
   - Download: https://dev.mysql.com/downloads/mysql/
   - Set root password during installation

2. **Create Database**
   ```sql
   -- Connect to MySQL as root
   CREATE DATABASE travelmind_db;
   CREATE USER 'travelmind_user'@'localhost' IDENTIFIED BY 'your_secure_password';
   GRANT ALL PRIVILEGES ON travelmind_db.* TO 'travelmind_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Install Python Driver**
   ```bash
   pip install PyMySQL
   ```

4. **Update .env file**
   ```env
   DATABASE_URL=mysql+pymysql://travelmind_user:your_secure_password@localhost:3306/travelmind_db
   ```

### 🎯 XAMPP MySQL (Perfect for Local Development)

XAMPP includes MySQL/MariaDB which is perfect for TravelMind development!

1. **Start XAMPP Services**
   - Open XAMPP Control Panel
   - Start **Apache** and **MySQL** services
   - Click **Admin** next to MySQL to open phpMyAdmin

2. **Create Database via phpMyAdmin**
   - Go to http://localhost/phpmyadmin
   - Click "New" to create a database
   - Database name: `travelmind_db`
   - Collation: `utf8mb4_general_ci`
   - Click "Create"

3. **Create User (Optional - for security)**
   ```sql
   -- In phpMyAdmin SQL tab, run:
   CREATE USER 'travelmind_user'@'localhost' IDENTIFIED BY 'travelmind123';
   GRANT ALL PRIVILEGES ON travelmind_db.* TO 'travelmind_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

4. **Install Python MySQL Driver**
   ```bash
   pip install PyMySQL
   ```

5. **Update your .env file**
   ```env
   # For root user (simpler):
   DATABASE_URL=mysql+pymysql://root@localhost:3306/travelmind_db
   
   # Or for custom user (more secure):
   DATABASE_URL=mysql+pymysql://travelmind_user:travelmind123@localhost:3306/travelmind_db
   ```

6. **Test Connection**
   ```bash
   python -c "from src.data.storage import DatabaseManager; db = DatabaseManager(); print('✅ XAMPP MySQL connected!')"
   ```

## 🔧 Database Operations

### Initialize Database
```bash
python setup_database.py
```

### Test Connection
```bash
python -c "from src.data.storage import DatabaseManager; db = DatabaseManager(); print('Stats:', db.get_database_stats())"
```

### Backup Database (SQLite only)
```bash
python -c "from src.data.storage import DatabaseManager; db = DatabaseManager(); db.backup_database('backup.db'); print('Backup created!')"
```

### View Database Stats
```bash
python -c "
from src.data.storage import DatabaseManager
db = DatabaseManager()
stats = db.get_database_stats()
print('📊 Database Statistics:')
for table, count in stats.items():
    print(f'   📄 {table}: {count} records')
"
```

## 🛠️ Database Schema

Your TravelMind database includes these tables:

- **hotels** - Hotel information and amenities
- **user_preferences** - User travel preferences
- **recommendations** - AI recommendation history
- **user_feedback** - User ratings and feedback

## 🔍 Troubleshooting

### SQLite Issues
- ✅ No installation required
- ✅ Database file created automatically
- ⚠️ Single-user access only
- 📁 Database file: `travelmind.db`

### PostgreSQL Issues
- Check if PostgreSQL service is running
- Verify username/password in connection string
- Ensure database exists
- Test connection: `psql -U travelmind_user -d travelmind_db`

### MySQL Issues
- Check if MySQL service is running
- Verify username/password in connection string
- Ensure database exists
- Test connection: `mysql -u travelmind_user -p travelmind_db`

## 📊 Performance Tips

### For Development
- SQLite is perfect - fast and simple

### For Production
- Use PostgreSQL for better performance
- Enable connection pooling
- Regular backups
- Monitor database size
- Consider read replicas for high traffic

## 🔐 Security Notes

1. **Never commit passwords** to version control
2. **Use environment variables** for credentials
3. **Limit database user permissions**
4. **Regular security updates**
5. **Backup regularly**

---

**Need help?** Run `python setup_database.py` for an interactive setup guide!
