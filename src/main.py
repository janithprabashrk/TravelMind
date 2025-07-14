from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import logging
from datetime import datetime
import traceback
import uvicorn

from .api.routes import router, admin_router
from .config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TravelMind API",
    description="AI-Powered Hotel Recommendation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if Config.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting TravelMind API...")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"API will be available at http://{Config.API_HOST}:{Config.API_PORT}")
    
    # Create necessary directories
    Config.create_directories()
    
    logger.info("TravelMind API started successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down TravelMind API...")

# Include routers
app.include_router(router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["Administration"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TravelMind API",
        "version": "1.0.0",
        "description": "AI-Powered Hotel Recommendation System",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "recommendations": "/api/v1/recommendations",
            "hotels": "/api/v1/hotels",
            "feedback": "/api/v1/feedback",
            "status": "/api/v1/status",
            "admin": "/api/v1/admin/"
        }
    }

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TravelMind API",
        version="1.0.0",
        description="""
        ## TravelMind - AI-Powered Hotel Recommendation System
        
        This API provides intelligent hotel recommendations based on user preferences, 
        location, season, budget, and experience ratings using advanced machine learning algorithms.
        
        ### Features
        - 🤖 **AI-Powered Recommendations**: Multiple ML algorithms for personalized suggestions
        - 🏨 **Comprehensive Hotel Data**: Detailed information including amenities, ratings, and pricing
        - 📊 **Multi-Factor Analysis**: Considers season, budget, ratings, and user experience
        - 🎯 **Specialized Recommendations**: Content-based, collaborative, value, luxury, and family-friendly options
        - 📈 **Real-time Learning**: Continuous improvement based on user feedback
        
        ### Usage
        1. **Get Recommendations**: POST to `/api/v1/recommendations` with user preferences
        2. **Search Hotels**: GET `/api/v1/hotels` with filters
        3. **Submit Feedback**: POST to `/api/v1/feedback` to improve recommendations
        4. **Check Status**: GET `/api/v1/status` for system information
        
        ### Admin Functions
        - **Train Models**: POST to `/api/v1/admin/train` to retrain ML models
        - **Collect Data**: POST to `/api/v1/admin/collect-data` to gather new hotel data
        - **Database Stats**: GET `/api/v1/admin/database/stats` for analytics
        """,
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Development server function
def run_dev_server():
    """Run development server"""
    uvicorn.run(
        "src.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG,
        log_level="info" if Config.DEBUG else "warning"
    )

if __name__ == "__main__":
    run_dev_server()
