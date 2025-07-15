#!/usr/bin/env python3
"""
TravelMind - Training Script
Run this script to collect data and train ML models
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.trainer import ModelTrainer
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main training function"""
    print("🏨 TravelMind - AI Hotel Recommendation System")
    print("=" * 50)
    
    # Check if API keys are configured
    if not Config.GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in the .env file")
        return
    
    # Training locations
    training_locations = [
        "Paris, France",
        "Tokyo, Japan", 
        "New York, USA",
        "London, UK",
        "Barcelona, Spain",
        "Sydney, Australia",
        "Rome, Italy",
        "Bangkok, Thailand",
        "Amsterdam, Netherlands",
        "Dubai, UAE"
    ]
    
    print(f"🎯 Training with {len(training_locations)} locations:")
    for i, location in enumerate(training_locations, 1):
        print(f"   {i}. {location}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        print("\n🚀 Starting full training pipeline...")
        
        # Run training
        results = await trainer.full_training_pipeline(training_locations)
        
        if results["status"] == "success":
            print("\n✅ Training completed successfully!")
            print(f"📁 Models saved to: {results['model_path']}")
            
            # Display metrics
            metrics = results["training_results"]["metrics"]
            print("\n📊 Model Performance:")
            for model_name, model_metrics in metrics.items():
                print(f"   {model_name}:")
                print(f"      R² Score: {model_metrics.get('r2', 0):.3f}")
                print(f"      MAE: {model_metrics.get('mae', 0):.3f}")
            
            # Database stats
            db_stats = results["report"]["data_statistics"]
            print(f"\n📈 Database Statistics:")
            print(f"   Total Hotels: {db_stats['total_hotels']}")
            print(f"   Unique Locations: {db_stats['unique_locations']}")
            
            print("\n🎉 TravelMind is ready to provide recommendations!")
            print("   Run: python -m src.main  (to start API)")
            print("   Run: streamlit run frontend/app.py  (to start UI)")
            
        else:
            print(f"\n❌ Training failed: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        logger.exception("Training error details:")

if __name__ == "__main__":
    asyncio.run(main())
