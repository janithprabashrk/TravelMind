#!/usr/bin/env python3
"""
TravelMind - Quick Start Script
Sets up the environment and provides quick commands
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main setup function"""
    print("🏨 TravelMind - AI Hotel Recommendation System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        return
    
    print("✅ Python version check passed")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("\n📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/MacOS
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install requirements
    if Path("requirements.txt").exists():
        print("\n📋 Installing requirements...")
        try:
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            print("✅ Requirements installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return
    
    # Check for .env file
    if not Path(".env").exists():
        print("\n📝 Creating .env file from template...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your API keys:")
            print("   - GEMINI_API_KEY (required for data collection)")
            print("   - OPENWEATHER_API_KEY (optional for weather data)")
        else:
            print("❌ .env.example not found")
    
    # Create necessary directories
    dirs_to_create = ["data", "models", "logs"]
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✅ Created directories: {', '.join(dirs_to_create)}")
    
    print("\n🎯 Next Steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run training: python train.py")
    print("3. Start API: python -m src.main")
    print("4. Start UI: streamlit run frontend/app.py")
    
    print("\n🚀 Quick Commands:")
    if os.name == 'nt':  # Windows
        print("   Activate venv: venv\\Scripts\\activate")
    else:
        print("   Activate venv: source venv/bin/activate")
    
    print("   Train models: python train.py")
    print("   Start API: python -m src.main")
    print("   Start UI: streamlit run frontend/app.py")
    print("   Run tests: python -m pytest tests/")

if __name__ == "__main__":
    main()
