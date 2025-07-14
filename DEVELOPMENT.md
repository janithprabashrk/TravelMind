# Development Guide

## Project Structure

```
TravelMind/
├── src/                      # Source code
│   ├── api/                  # FastAPI routes and schemas
│   ├── data/                 # Data collection and storage
│   ├── models/               # ML models and algorithms
│   ├── config.py            # Configuration management
│   └── main.py              # FastAPI application
├── frontend/                 # Streamlit frontend
├── tests/                   # Unit tests
├── docker/                  # Docker configurations
├── data/                    # Data storage
├── models/                  # Trained models
└── logs/                    # Application logs
```

## Development Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd TravelMind

# Run setup script
python setup.py

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/Linux/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

Required environment variables:
- `GEMINI_API_KEY`: Google Gemini API key (required)
- `OPENWEATHER_API_KEY`: OpenWeatherMap API key (optional)

### 3. Database Setup

The application uses SQLite by default. The database will be automatically initialized when you first run the application.

## API Development

### Adding New Endpoints

1. Define request/response schemas in `src/api/schemas.py`
2. Implement route handlers in `src/api/routes.py`
3. Add route to the main router in `src/main.py`

### Example: Adding a new endpoint

```python
# In schemas.py
class NewRequest(BaseModel):
    parameter: str

class NewResponse(BaseModel):
    result: str

# In routes.py
@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Implementation
    return NewResponse(result="success")
```

## ML Model Development

### Adding New Models

1. Create model class in `src/models/`
2. Integrate with `HotelRecommendationEngine`
3. Update training pipeline in `src/models/trainer.py`

### Model Training Pipeline

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
results = await trainer.full_training_pipeline(locations)
```

## Data Collection

### Adding New Data Sources

1. Extend `HotelDataCollector` in `src/data/collector.py`
2. Update preprocessing pipeline
3. Modify database schema if needed

### Custom Data Collection

```python
from src.data.collector import HotelDataCollector

collector = HotelDataCollector()
data = await collector.collect_location_data("Location")
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest tests/ --cov=src
```

### Writing Tests

Follow the existing test patterns in `tests/` directory. Use pytest fixtures for setup.

## Deployment

### Local Development

```bash
# Start API server
python -m src.main

# Start frontend (in another terminal)
streamlit run frontend/app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual containers
docker build -t travelmind-api .
docker run -p 8000:8000 travelmind-api
```

### Production Deployment

1. Set `DEBUG=False` in environment
2. Use production database (PostgreSQL recommended)
3. Set up reverse proxy (nginx)
4. Use process manager (systemd, supervisor)

## Configuration

### Application Settings

Modify `src/config.py` for application-wide settings.

### Model Configuration

Model hyperparameters can be adjusted in the respective model classes.

## Monitoring and Logging

### Logging

Logs are written to `logs/` directory. Configure logging levels in `src/main.py`.

### Health Checks

Use `/api/v1/health` endpoint for health monitoring.

### Metrics

System metrics available at `/api/v1/status`.

## Performance Optimization

### Database Optimization

1. Use database indexes for frequently queried columns
2. Implement connection pooling for production
3. Consider read replicas for heavy read workloads

### Model Optimization

1. Use model caching for frequent predictions
2. Implement batch prediction for multiple requests
3. Consider model quantization for deployment

### API Optimization

1. Implement response caching
2. Use async/await for I/O operations
3. Add request rate limiting

## Security Considerations

### API Security

1. Implement authentication for admin endpoints
2. Add input validation and sanitization
3. Use HTTPS in production
4. Implement rate limiting

### Data Security

1. Encrypt sensitive data at rest
2. Use secure connections for external APIs
3. Implement audit logging
4. Regular security updates

## Contribution Guidelines

### Code Style

1. Follow PEP 8 for Python code
2. Use type hints where possible
3. Add docstrings for all functions
4. Run linting before commits

### Pull Request Process

1. Create feature branch from main
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit pull request with description

### Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for refactoring

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure Gemini API key is set correctly
2. **Database Errors**: Check file permissions and disk space
3. **Model Training Fails**: Verify sufficient data and memory
4. **Import Errors**: Check virtual environment activation

### Debug Mode

Set `DEBUG=True` in environment for detailed error messages.

### Performance Issues

1. Monitor resource usage
2. Check database query performance
3. Profile ML model inference time
4. Review API response times

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Google Gemini API](https://ai.google.dev/)

## Support

For issues and questions:
1. Check existing documentation
2. Search through issues
3. Create new issue with detailed description
4. Include error logs and system information
