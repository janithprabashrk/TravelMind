from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import asyncio
import uuid

from .schemas import (
    RecommendationRequest, RecommendationResponse, UserFeedback,
    DataCollectionRequest, TrainingRequest, TrainingResponse,
    SystemStatus, HealthCheck, ErrorResponse, DatabaseStats,
    TrainingStatus, UserPreferences
)
from ..models.trainer import ModelTrainer
from ..models.recommender import HotelRecommendationEngine
from ..data.storage import DatabaseManager
from ..data.collector import HotelDataCollector
from ..config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Global instances
trainer = ModelTrainer()
db = DatabaseManager()
collector = HotelDataCollector()

# Background training tasks
active_training_tasks = {}

# Create routers
router = APIRouter()
admin_router = APIRouter(prefix="/admin")

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        stats = db.get_database_stats()
        db_status = "healthy" if stats['total_hotels'] >= 0 else "unhealthy"
        
        # Check model status
        model_status = "healthy" if trainer.recommendation_engine.is_trained else "not_trained"
        
        return HealthCheck(
            components={
                "database": db_status,
                "models": model_status,
                "api": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        db_stats = db.get_database_stats()
        
        from .schemas import ModelStatus, ModelMetrics
        
        def convert_metrics(metrics_dict):
            if metrics_dict and isinstance(metrics_dict, dict):
                return ModelMetrics(**metrics_dict)
            return None
        
        models_status = [
            ModelStatus(
                model_name="content_model",
                is_trained=trainer.recommendation_engine.is_trained,
                metrics=convert_metrics(trainer.model_metrics.get('content_model'))
            ),
            ModelStatus(
                model_name="value_model", 
                is_trained=trainer.recommendation_engine.is_trained,
                metrics=convert_metrics(trainer.model_metrics.get('value_model'))
            ),
            ModelStatus(
                model_name="luxury_model",
                is_trained=trainer.recommendation_engine.is_trained,
                metrics=convert_metrics(trainer.model_metrics.get('luxury_model'))
            ),
            ModelStatus(
                model_name="family_model",
                is_trained=trainer.recommendation_engine.is_trained,
                metrics=convert_metrics(trainer.model_metrics.get('family_model'))
            )
        ]
        
        return SystemStatus(
            total_hotels=db_stats['total_hotels'],
            unique_locations=db_stats['unique_locations'],
            models_status=models_status
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get hotel recommendations based on user preferences"""
    start_time = datetime.now()
    
    try:
        if not trainer.recommendation_engine.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Recommendation models are not trained yet. Please train models first."
            )
        
        # Get candidate hotels from database
        candidate_hotels = db.get_hotels(
            location=request.user_preferences.location,
            min_rating=request.user_preferences.min_rating,
            max_price=request.user_preferences.budget_max
        )
        
        if not candidate_hotels:
            raise HTTPException(
                status_code=404,
                detail=f"No hotels found for location: {request.user_preferences.location}"
            )
        
        # Convert user preferences to dict
        user_prefs_dict = request.user_preferences.dict()
        
        # Get recommendations
        recommendation_type = request.recommendation_type
        if recommendation_type == "value_based":
            recommendation_type = "value"
            
        recommendations = trainer.recommendation_engine.predict_recommendations(
            user_preferences=user_prefs_dict,
            candidate_hotels=candidate_hotels,
            recommendation_type=recommendation_type,
            top_k=request.top_k
        )
        
        # Convert to response format
        recommendation_results = []
        for rec in recommendations:
            hotel_data = rec.copy()
            
            # Extract recommendation metadata
            rec_score = hotel_data.pop('recommendation_score', 0.0)
            rec_rank = hotel_data.pop('recommendation_rank', 0)
            rec_type = hotel_data.pop('recommendation_type', recommendation_type)
            
            # Create recommendation result
            rec_result = {
                "hotel": hotel_data,
                "recommendation_score": rec_score,
                "recommendation_rank": rec_rank,
                "recommendation_type": rec_type,
                "reasoning": _generate_reasoning(hotel_data, user_prefs_dict, rec_score)
            }
            
            # Add similar hotels if requested
            if request.include_similar:
                try:
                    similar = trainer.recommendation_engine.get_similar_hotels(
                        hotel_data.get('id', rec_rank), top_k=3
                    )
                    # Convert similar hotels (simplified)
                    rec_result["similar_hotels"] = []
                except:
                    rec_result["similar_hotels"] = []
            
            recommendation_results.append(rec_result)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            recommendations=recommendation_results,
            total_found=len(candidate_hotels),
            user_preferences=request.user_preferences,
            recommendation_type=request.recommendation_type,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback: UserFeedback):
    """Submit user feedback on recommendations"""
    try:
        feedback_id = db.save_user_feedback(
            user_id=feedback.user_id,
            hotel_id=feedback.hotel_id,
            recommendation_id=feedback.recommendation_id or 0,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text or ""
        )
        
        return {"status": "success", "feedback_id": feedback_id}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hotels")
async def search_hotels(
    location: Optional[str] = Query(None, description="Location to search"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    amenities: Optional[str] = Query(None, description="Comma-separated amenities"),
    limit: Optional[int] = Query(50, ge=1, le=100, description="Maximum results")
):
    """Search hotels with filters"""
    try:
        amenity_list = amenities.split(',') if amenities else None
        
        hotels = db.get_hotels(
            location=location,
            min_rating=min_rating,
            max_price=max_price,
            amenities=amenity_list,
            limit=limit
        )
        
        return {
            "hotels": hotels,
            "count": len(hotels),
            "filters_applied": {
                "location": location,
                "min_rating": min_rating,
                "max_price": max_price,
                "amenities": amenity_list
            }
        }
        
    except Exception as e:
        logger.error(f"Hotel search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@admin_router.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training process"""
    try:
        training_id = str(uuid.uuid4())
        
        # Start training in background
        background_tasks.add_task(
            _background_training_task,
            training_id,
            request.locations,
            request.retrain_existing,
            request.hyperparameter_tuning
        )
        
        # Store training task info
        active_training_tasks[training_id] = {
            "status": "started",
            "started_at": datetime.now(),
            "locations": request.locations
        }
        
        return TrainingResponse(
            status="started",
            message="Training started in background",
            training_id=training_id,
            estimated_duration_minutes=len(request.locations) * 5  # Rough estimate
        )
        
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/training/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get training status"""
    if training_id not in active_training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    task_info = active_training_tasks[training_id]
    
    return TrainingStatus(
        status=task_info["status"],
        started_at=task_info.get("started_at"),
        completed_at=task_info.get("completed_at"),
        message=task_info.get("message"),
        metrics=task_info.get("metrics")
    )

@admin_router.post("/collect-data")
async def collect_data(request: DataCollectionRequest, background_tasks: BackgroundTasks):
    """Collect hotel data for specified locations"""
    try:
        collection_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            _background_data_collection_task,
            collection_id,
            request.locations,
            request.force_refresh
        )
        
        return {
            "status": "started",
            "collection_id": collection_id,
            "locations": request.locations,
            "message": "Data collection started in background"
        }
        
    except Exception as e:
        logger.error(f"Data collection start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/database/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = db.get_database_stats()
        return DatabaseStats(**stats)
        
    except Exception as e:
        logger.error(f"Database stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/models/load")
async def load_models(model_path: str):
    """Load pre-trained models"""
    try:
        trainer.load_models(model_path)
        return {"status": "success", "message": f"Models loaded from {model_path}"}
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _background_training_task(training_id: str, locations: List[str], 
                                  retrain_existing: bool, hyperparameter_tuning: bool):
    """Background training task"""
    try:
        active_training_tasks[training_id]["status"] = "training"
        
        # Run training pipeline
        results = await trainer.full_training_pipeline(locations, retrain_existing)
        
        if results["status"] == "success":
            active_training_tasks[training_id].update({
                "status": "completed",
                "completed_at": datetime.now(),
                "message": "Training completed successfully",
                "metrics": results["training_results"]["metrics"]
            })
        else:
            active_training_tasks[training_id].update({
                "status": "failed",
                "completed_at": datetime.now(),
                "message": f"Training failed: {results.get('error', 'Unknown error')}"
            })
            
    except Exception as e:
        logger.error(f"Background training failed: {e}")
        active_training_tasks[training_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "message": f"Training failed: {str(e)}"
        })

async def _background_data_collection_task(collection_id: str, locations: List[str], force_refresh: bool):
    """Background data collection task"""
    try:
        collected_count = 0
        
        for location in locations:
            try:
                location_data = await collector.collect_location_data(location)
                if location_data and 'hotels' in location_data:
                    db.insert_hotels(location_data['hotels'])
                    collected_count += len(location_data['hotels'])
                    
            except Exception as e:
                logger.error(f"Failed to collect data for {location}: {e}")
                continue
        
        logger.info(f"Data collection {collection_id} completed: {collected_count} hotels collected")
        
    except Exception as e:
        logger.error(f"Background data collection failed: {e}")

def _generate_reasoning(hotel: Dict[str, Any], user_prefs: Dict[str, Any], score: float) -> str:
    """Generate explanation for recommendation"""
    reasons = []
    
    # Price match
    budget_max = user_prefs.get('budget_max')
    hotel_price = hotel.get('avg_price', hotel.get('min_price', 0))
    
    if budget_max and hotel_price <= budget_max:
        reasons.append("fits your budget")
    
    # Rating
    rating = hotel.get('user_rating', 0)
    if rating >= 4.0:
        reasons.append("highly rated by guests")
    
    # Amenities match
    preferred_amenities = user_prefs.get('preferred_amenities', [])
    hotel_amenities = hotel.get('amenities', [])
    
    if preferred_amenities and hotel_amenities:
        matches = [a for a in preferred_amenities if any(a.lower() in ha.lower() for ha in hotel_amenities)]
        if matches:
            reasons.append(f"has {', '.join(matches[:2])}")
    
    # Family/business travel
    if user_prefs.get('family_travel') and hotel.get('family_friendly'):
        reasons.append("family-friendly")
    
    if user_prefs.get('business_travel') and any('business' in str(a).lower() for a in hotel.get('amenities', [])):
        reasons.append("business facilities available")
    
    if reasons:
        return f"Recommended because it {', '.join(reasons[:3])}."
    else:
        return f"Good match with a score of {score:.1f}/5.0"
