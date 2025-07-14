# TravelMind - AI-Powered Hotel Recommendation System

## Overview
TravelMind is an industrial-level machine learning project that provides personalized hotel recommendations based on user preferences, location, season, budget, and ratings. The system leverages Google Gemini API for hotel data collection and uses advanced ML algorithms for recommendation.

## Features
- 🏨 Location-based hotel and villa discovery
- 🤖 ML-powered recommendation engine
- 📊 Multi-factor analysis (season, ratings, budget, experience)
- 💾 Persistent model storage (PKL files)
- 🌐 Free and open-source implementation
- 📈 Industrial-grade architecture

## Tech Stack
- **Backend**: Python, FastAPI
- **ML Framework**: Scikit-learn, Pandas, NumPy
- **API**: Google Gemini API
- **Database**: SQLite (lightweight, no cost)
- **Frontend**: Streamlit (for demo interface)
- **Model Storage**: Pickle files
- **Deployment**: Docker support

## Architecture
```
TravelMind/
├── src/
│   ├── data/
│   │   ├── collector.py      # Gemini API integration
│   │   ├── preprocessor.py   # Data cleaning & preparation
│   │   └── storage.py        # Database operations
│   ├── models/
│   │   ├── recommender.py    # ML recommendation engine
│   │   ├── feature_engine.py # Feature engineering
│   │   └── trainer.py        # Model training pipeline
│   ├── api/
│   │   ├── routes.py         # API endpoints
│   │   └── schemas.py        # Data validation
│   └── utils/
│       ├── config.py         # Configuration management
│       └── helpers.py        # Utility functions
├── models/                   # Saved PKL models
├── data/                     # Raw and processed data
├── tests/                    # Unit tests
├── frontend/                 # Streamlit interface
├── docker/                   # Docker configuration
└── requirements.txt
```

## Installation & Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Gemini API key in `.env`
4. Run data collection: `python src/data/collector.py`
5. Train models: `python src/models/trainer.py`
6. Start API: `uvicorn src.api.main:app --reload`
7. Launch UI: `streamlit run frontend/app.py`

## Model Features
- **Content-based filtering**: Based on hotel amenities and features
- **Collaborative filtering**: User behavior patterns
- **Hybrid approach**: Combines multiple recommendation strategies
- **Seasonal adjustments**: Weather and tourism patterns
- **Budget optimization**: Price-performance analysis
- **Rating analysis**: Multi-source rating aggregation

## Free Resources Used
- Gemini API (free tier)
- OpenWeatherMap API (free tier)
- SQLite database
- Open-source Python libraries
- Docker (free for personal use)
