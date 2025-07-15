#!/usr/bin/env python3
"""
TravelMind Deployment Verification Script

This script verifies that all components of the TravelMind system
are properly installed and configured for deployment.
"""

import os
import sys
import subprocess
import importlib
import sqlite3
from pathlib import Path

class DeploymentVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        if version < (3, 8):
            self.errors.append(f"Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
        else:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.errors.append("requirements.txt not found")
            return
        
        with open(requirements_file, 'r') as f:
            packages = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
        
        missing_packages = []
        for package in packages:
            try:
                importlib.import_module(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
        else:
            print(f"✅ All {len(packages)} required packages installed")
    
    def check_project_structure(self):
        """Verify project directory structure"""
        required_dirs = [
            "src", "src/api", "src/data", "src/models", "src/utils",
            "frontend", "tests", "docker", "data", "models", "logs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.errors.append(f"Missing directories: {', '.join(missing_dirs)}")
        else:
            print("✅ Project structure - OK")
    
    def check_configuration(self):
        """Check configuration files"""
        config_files = [".env.example", "config.yaml"]
        
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                self.warnings.append(f"Configuration file {config_file} not found")
        
        # Check if .env exists
        if not (self.project_root / ".env").exists():
            self.warnings.append(".env file not found - copy from .env.example and configure")
        else:
            print("✅ Configuration files - OK")
    
    def check_database(self):
        """Verify database setup"""
        db_path = self.project_root / "data" / "travelmind.db"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['hotels', 'user_preferences', 'recommendations', 'feedback']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                self.warnings.append(f"Database tables not found: {', '.join(missing_tables)}")
                self.warnings.append("Run 'python -c \"from src.data.storage import DatabaseManager; DatabaseManager()\"' to initialize")
            else:
                print("✅ Database structure - OK")
            
            conn.close()
            
        except Exception as e:
            self.warnings.append(f"Database check failed: {str(e)}")
    
    def check_models(self):
        """Check if ML models are trained"""
        models_dir = self.project_root / "models"
        
        expected_models = [
            "content_model.pkl", "collaborative_model.pkl",
            "hybrid_model.pkl", "value_model.pkl", "luxury_model.pkl"
        ]
        
        missing_models = []
        for model_file in expected_models:
            if not (models_dir / model_file).exists():
                missing_models.append(model_file)
        
        if missing_models:
            self.warnings.append(f"ML models not found: {', '.join(missing_models)}")
            self.warnings.append("Run 'python train.py' to train models")
        else:
            print("✅ ML models - OK")
    
    def check_api_server(self):
        """Test if API server can be imported"""
        try:
            sys.path.insert(0, str(self.project_root))
            from src.main import app
            print("✅ FastAPI application - OK")
        except ImportError as e:
            self.errors.append(f"Cannot import FastAPI app: {str(e)}")
    
    def check_frontend(self):
        """Test if frontend can be imported"""
        try:
            frontend_path = self.project_root / "frontend" / "app.py"
            if frontend_path.exists():
                print("✅ Streamlit frontend - OK")
            else:
                self.errors.append("Frontend app.py not found")
        except Exception as e:
            self.errors.append(f"Frontend check failed: {str(e)}")
    
    def check_docker(self):
        """Check Docker configuration"""
        docker_files = ["Dockerfile", "docker-compose.yml"]
        
        for docker_file in docker_files:
            if not (self.project_root / docker_file).exists():
                self.warnings.append(f"Docker file {docker_file} not found")
        
        # Check if Docker is available
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker available")
            else:
                self.warnings.append("Docker not available")
        except FileNotFoundError:
            self.warnings.append("Docker not installed")
    
    def generate_startup_commands(self):
        """Generate startup commands for the user"""
        print("\n" + "="*60)
        print("🚀 STARTUP COMMANDS")
        print("="*60)
        
        print("\n1. Environment Setup:")
        print("   # Windows:")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        print("")
        print("   # Unix/Linux/MacOS:")
        print("   python -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        
        print("\n2. Configuration:")
        print("   cp .env.example .env")
        print("   # Edit .env with your API keys")
        
        print("\n3. Initialize Database & Train Models:")
        print("   python -c \"from src.data.storage import DatabaseManager; DatabaseManager()\"")
        print("   python train.py")
        
        print("\n4. Start Services:")
        print("   # Terminal 1 - API Server:")
        print("   python -m src.main")
        print("")
        print("   # Terminal 2 - Web Interface:")
        print("   streamlit run frontend/app.py")
        
        print("\n5. Access Applications:")
        print("   • Web Interface: http://localhost:8501")
        print("   • API Documentation: http://localhost:8000/docs")
        print("   • API Health: http://localhost:8000/api/v1/health")
        
        print("\n6. Docker Deployment (Alternative):")
        print("   docker-compose up --build")
    
    def run_verification(self):
        """Run all verification checks"""
        print("🔍 TravelMind Deployment Verification")
        print("="*50)
        
        self.check_python_version()
        self.check_dependencies()
        self.check_project_structure()
        self.check_configuration()
        self.check_database()
        self.check_models()
        self.check_api_server()
        self.check_frontend()
        self.check_docker()
        
        # Report results
        print("\n" + "="*50)
        print("📊 VERIFICATION RESULTS")
        print("="*50)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n🎉 ALL CHECKS PASSED!")
            print("Your TravelMind installation is ready for deployment!")
        elif not self.errors:
            print("\n✅ READY FOR DEPLOYMENT")
            print("Address warnings above for optimal performance.")
        else:
            print("\n🔧 ISSUES FOUND")
            print("Please fix errors before deployment.")
        
        self.generate_startup_commands()

if __name__ == "__main__":
    verifier = DeploymentVerifier()
    verifier.run_verification()
