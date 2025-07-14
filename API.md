# API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently, no authentication is required for most endpoints. Admin endpoints may require authentication in production.

## Rate Limiting
API requests are currently not rate-limited in development. Consider implementing rate limiting for production use.

---

## Endpoints

### Health Check
**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "models": "healthy",
    "api": "healthy"
  }
}
```

---

### System Status
**GET** `/status`

Get comprehensive system information.

**Response:**
```json
{
  "total_hotels": 1250,
  "unique_locations": 25,
  "models_status": [
    {
      "model_name": "content_model",
      "is_trained": true,
      "last_trained": "2024-01-15T09:00:00Z",
      "metrics": {
        "mse": 0.123,
        "mae": 0.234,
        "r2": 0.856
      }
    }
  ],
  "last_data_collection": "2024-01-15T08:00:00Z",
  "api_version": "1.0.0"
}
```

---

### Get Recommendations
**POST** `/recommendations`

Get personalized hotel recommendations.

**Request Body:**
```json
{
  "user_preferences": {
    "location": "Paris, France",
    "budget_min": 100,
    "budget_max": 300,
    "preferred_season": "summer",
    "preferred_amenities": ["wifi", "pool", "gym"],
    "property_type_preference": "hotel",
    "min_rating": 3.5,
    "family_travel": true,
    "business_travel": false,
    "group_size": 2
  },
  "recommendation_type": "hybrid",
  "top_k": 10,
  "include_similar": false
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "hotel": {
        "id": 123,
        "name": "Grand Hotel Paris",
        "location": "Paris, France",
        "star_rating": 4.5,
        "user_rating": 4.3,
        "min_price": 180,
        "max_price": 280,
        "avg_price": 230,
        "amenities": ["wifi", "pool", "gym", "spa"],
        "description": "Luxury hotel in central Paris",
        "family_friendly": true,
        "pet_friendly": false
      },
      "recommendation_score": 4.2,
      "recommendation_rank": 1,
      "recommendation_type": "hybrid",
      "reasoning": "Recommended because it fits your budget, highly rated by guests, has wifi, pool."
    }
  ],
  "total_found": 45,
  "user_preferences": { /* ... */ },
  "recommendation_type": "hybrid",
  "generated_at": "2024-01-15T10:30:00Z",
  "processing_time_ms": 125.5
}
```

---

### Search Hotels
**GET** `/hotels`

Search hotels with filters.

**Query Parameters:**
- `location` (optional): Location to search
- `min_rating` (optional): Minimum rating (0-5)
- `max_price` (optional): Maximum price per night
- `amenities` (optional): Comma-separated amenities
- `limit` (optional): Maximum results (1-100, default: 50)

**Example:**
```
GET /hotels?location=Tokyo&min_rating=4.0&max_price=200&amenities=wifi,pool&limit=20
```

**Response:**
```json
{
  "hotels": [
    {
      "id": 456,
      "name": "Tokyo Bay Hotel",
      "location": "Tokyo, Japan",
      "star_rating": 4.0,
      "user_rating": 4.1,
      "min_price": 120,
      "max_price": 180,
      "amenities": ["wifi", "pool", "restaurant"]
    }
  ],
  "count": 15,
  "filters_applied": {
    "location": "Tokyo",
    "min_rating": 4.0,
    "max_price": 200,
    "amenities": ["wifi", "pool"]
  }
}
```

---

### Submit Feedback
**POST** `/feedback`

Submit user feedback on recommendations.

**Request Body:**
```json
{
  "user_id": "user123",
  "hotel_id": 456,
  "recommendation_id": 789,
  "rating": 4.5,
  "feedback_text": "Great hotel with excellent service!"
}
```

**Response:**
```json
{
  "status": "success",
  "feedback_id": 101
}
```

---

## Admin Endpoints

### Start Training
**POST** `/admin/train`

Start model training process.

**Request Body:**
```json
{
  "locations": ["Paris, France", "Tokyo, Japan", "New York, USA"],
  "retrain_existing": false,
  "hyperparameter_tuning": true
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Training started in background",
  "training_id": "550e8400-e29b-41d4-a716-446655440000",
  "estimated_duration_minutes": 15
}
```

---

### Training Status
**GET** `/admin/training/{training_id}`

Get training process status.

**Response:**
```json
{
  "status": "completed",
  "progress": 100.0,
  "message": "Training completed successfully",
  "started_at": "2024-01-15T09:00:00Z",
  "completed_at": "2024-01-15T09:15:00Z",
  "metrics": {
    "content_model": {
      "mse": 0.123,
      "mae": 0.234,
      "r2": 0.856
    }
  }
}
```

---

### Collect Data
**POST** `/admin/collect-data`

Start data collection for specified locations.

**Request Body:**
```json
{
  "locations": ["Barcelona, Spain", "Rome, Italy"],
  "force_refresh": false
}
```

**Response:**
```json
{
  "status": "started",
  "collection_id": "550e8400-e29b-41d4-a716-446655440001",
  "locations": ["Barcelona, Spain", "Rome, Italy"],
  "message": "Data collection started in background"
}
```

---

### Database Statistics
**GET** `/admin/database/stats`

Get database statistics.

**Response:**
```json
{
  "total_hotels": 1250,
  "unique_locations": 25,
  "user_preferences": 45,
  "total_recommendations": 890,
  "total_feedback": 234
}
```

---

### Load Models
**POST** `/admin/models/load`

Load pre-trained models.

**Query Parameters:**
- `model_path`: Path to the model file

**Response:**
```json
{
  "status": "success",
  "message": "Models loaded from /path/to/models.pkl"
}
```

---

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (resource not found)
- **422**: Validation Error (invalid request format)
- **500**: Internal Server Error
- **503**: Service Unavailable (models not trained)

---

## Data Models

### UserPreferences
```json
{
  "location": "string (required)",
  "budget_min": "number (optional, >= 0)",
  "budget_max": "number (optional, >= budget_min)",
  "preferred_season": "enum: spring|summer|fall|winter",
  "preferred_amenities": "array of strings",
  "property_type_preference": "enum: hotel|villa|resort|apartment|bed_and_breakfast",
  "min_rating": "number (0-5)",
  "family_travel": "boolean",
  "business_travel": "boolean",
  "group_size": "integer (>= 1)"
}
```

### Hotel
```json
{
  "id": "integer",
  "name": "string",
  "location": "string",
  "address": "string",
  "star_rating": "number (0-5)",
  "user_rating": "number (0-5)",
  "min_price": "number",
  "max_price": "number",
  "avg_price": "number",
  "property_type": "string",
  "amenities": "array of strings",
  "room_types": "array of strings",
  "description": "string",
  "best_season": "string",
  "nearby_attractions": "array of strings",
  "contact_info": {
    "phone": "string",
    "email": "string", 
    "website": "string"
  },
  "sustainability_rating": "number",
  "business_facilities": "array of strings",
  "family_friendly": "boolean",
  "pet_friendly": "boolean",
  "accessibility": "array of strings",
  "seasonal_data": "object",
  "collected_at": "datetime"
}
```

---

## Rate Limiting

For production deployment, consider implementing rate limiting:

- **General endpoints**: 100 requests per minute
- **Recommendation endpoint**: 10 requests per minute
- **Admin endpoints**: 5 requests per minute

---

## Pagination

For endpoints returning large datasets, implement pagination:

```json
{
  "items": [...],
  "total": 1000,
  "page": 1,
  "page_size": 50,
  "has_next": true,
  "has_prev": false
}
```

---

## WebSocket Support

Future versions may include WebSocket support for real-time updates:

- Training progress updates
- Live recommendation updates
- System status changes

---

## SDK Examples

### Python
```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/api/v1/recommendations",
    json={
        "user_preferences": {
            "location": "Paris, France",
            "budget_max": 300,
            "min_rating": 4.0
        },
        "recommendation_type": "hybrid",
        "top_k": 5
    }
)

recommendations = response.json()["recommendations"]
```

### JavaScript
```javascript
// Get recommendations
fetch('http://localhost:8000/api/v1/recommendations', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_preferences: {
      location: 'Paris, France',
      budget_max: 300,
      min_rating: 4.0
    },
    recommendation_type: 'hybrid',
    top_k: 5
  })
})
.then(response => response.json())
.then(data => console.log(data.recommendations));
```

### cURL
```bash
# Get recommendations
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_preferences": {
      "location": "Paris, France",
      "budget_max": 300,
      "min_rating": 4.0
    },
    "recommendation_type": "hybrid",
    "top_k": 5
  }'
```
