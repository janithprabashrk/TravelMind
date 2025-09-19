#!/usr/bin/env python3
"""
TravelMind Quick Start Script
Run this to start the complete system
"""

import subprocess
import sys
import time
from pathlib import Path

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting TravelMind API Server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, "-m", "src.main"], cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\n🛑 API Server stopped")

def start_frontend():
    """Start the Streamlit frontend"""
    print("🖥️ Starting TravelMind Web Interface...")
    print("🌐 App will be available at: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    try:
        subprocess.run(["streamlit", "run", "frontend/app.py"], cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped")

def main():
    print("🌟 TravelMind - AI Hotel Recommendation System")
    print("=" * 50)
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists() or len(list(models_dir.glob("*.pkl"))) < 6:
        print("⚠️  Models not found. Creating them now...")
        subprocess.run([sys.executable, "train.py"])
        print("✅ Models created!")
    
    print("\n🚀 Choose what to start:")
    print("1. API Server only (FastAPI)")
    print("2. Web Interface only (Streamlit)")
    print("3. Run verification test")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            start_api()
            break
        elif choice == "2":
            start_frontend()
            break
        elif choice == "3":
            print("🧪 Running verification test...")
            subprocess.run([sys.executable, "test_fix.py"])
            continue
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
