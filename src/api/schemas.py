from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class SeasonEnum(str, Enum):
    spring = "spring"
    summer = "summer"
    fall = "fall"
    winter = "winter"

class PropertyTypeEnum(str, Enum):
    hotel = "hotel"
    villa = "villa"
    resort = "resort"
    apartment = "apartment"
    bed_and_breakfast = "bed_and_breakfast"

class RecommendationTypeEnum(str, Enum):
    hybrid = "hybrid"
    content = "content"
    collaborative = "collaborative"
    value_based = "value"
    luxury = "luxury"
    family = "family"

# Request schemas
class UserPreferences(BaseModel):
    location: str = Field(..., description="Target location for hotel search")
    budget_min: Optional[float] = Field(None, ge=0, description="Minimum budget per night")
    budget_max: Optional[float] = Field(None, ge=0, description="Maximum budget per night")
    preferred_season: Optional[SeasonEnum] = Field(None, description="Preferred travel season")
    preferred_amenities: Optional[List[str]] = Field(default=[], description="List of preferred amenities")
    property_type_preference: Optional[PropertyTypeEnum] = Field(None, description="Preferred property type")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum hotel rating")
    family_travel: Optional[bool] = Field(False, description="Is this for family travel")
    business_travel: Optional[bool] = Field(False, description="Is this for business travel")
    group_size: Optional[int] = Field(1, ge=1, description="Number of travelers")
    
    @validator('budget_max')
    def budget_max_greater_than_min(cls, v, values):
        if v is not None and values.get('budget_min') is not None:
            if v < values['budget_min']:
                raise ValueError('budget_max must be greater than or equal to budget_min')
        return v

class RecommendationRequest(BaseModel):
    user_preferences: UserPreferences
    recommendation_type: RecommendationTypeEnum = RecommendationTypeEnum.hybrid
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    include_similar: bool = Field(False, description="Include similar hotels for each recommendation")

class UserFeedback(BaseModel):
    user_id: str
    hotel_id: int
    recommendation_id: Optional[int] = None
    rating: float = Field(..., ge=0, le=5, description="User rating for the hotel")
    feedback_text: Optional[str] = Field(None, description="Optional text feedback")

class DataCollectionRequest(BaseModel):
    locations: List[str] = Field(..., description="List of locations to collect data for")
    force_refresh: bool = Field(False, description="Force refresh of existing data")

# Response schemas
class ContactInfo(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None

class SeasonalData(BaseModel):
    min_price: float
    max_price: float
    occupancy_rate: float
    avg_rating: float

class Hotel(BaseModel):
    id: Optional[int] = None
    name: str
    location: str
    address: Optional[str] = None
    star_rating: float
    user_rating: float
    min_price: float
    max_price: float
    avg_price: Optional[float] = None
    property_type: Optional[str] = None
    amenities: List[str] = []
    room_types: Optional[List[str]] = []
    description: Optional[str] = None
    best_season: Optional[str] = None
    nearby_attractions: Optional[List[str]] = []
    contact_info: Optional[ContactInfo] = None
    sustainability_rating: Optional[float] = None
    business_facilities: Optional[List[str]] = []
    family_friendly: Optional[bool] = None
    pet_friendly: Optional[bool] = None
    accessibility: Optional[List[str]] = []
    seasonal_data: Optional[Dict[str, SeasonalData]] = None
    collected_at: Optional[datetime] = None

class RecommendationResult(BaseModel):
    hotel: Hotel
    recommendation_score: float = Field(..., description="Recommendation score (0-5)")
    recommendation_rank: int = Field(..., description="Rank in the recommendation list")
    recommendation_type: str = Field(..., description="Type of recommendation algorithm used")
    reasoning: Optional[str] = Field(None, description="Explanation for the recommendation")
    similar_hotels: Optional[List['RecommendationResult']] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationResult]
    total_found: int
    user_preferences: UserPreferences
    recommendation_type: RecommendationTypeEnum
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None

class ModelMetrics(BaseModel):
    mse: float
    mae: float
    r2: float

class ModelStatus(BaseModel):
    model_name: str
    is_trained: bool
    last_trained: Optional[datetime] = None
    metrics: Optional[ModelMetrics] = None

class SystemStatus(BaseModel):
    total_hotels: int
    unique_locations: int
    models_status: List[ModelStatus]
    last_data_collection: Optional[datetime] = None
    api_version: str = "1.0.0"

class TrainingStatus(BaseModel):
    status: str  # 'training', 'completed', 'failed'
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, ModelMetrics]] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Training and management schemas
class TrainingRequest(BaseModel):
    locations: List[str]
    retrain_existing: bool = Field(False, description="Whether to retrain with existing data")
    hyperparameter_tuning: bool = Field(True, description="Whether to perform hyperparameter tuning")

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_id: Optional[str] = None
    estimated_duration_minutes: Optional[int] = None

class DatabaseStats(BaseModel):
    total_hotels: int
    unique_locations: int
    user_preferences: int
    total_recommendations: int
    total_feedback: int

class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=lambda: {
        "database": "healthy",
        "models": "healthy",
        "api": "healthy"
    })

# Enable forward references
RecommendationResult.model_rebuild()
