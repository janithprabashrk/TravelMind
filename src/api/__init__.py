"""
API module for TravelMind providing RESTful endpoints.

This module includes:
- FastAPI route handlers
- Request/response schemas
- API documentation
"""

from .routes import router, admin_router
from .schemas import *

__all__ = ["router", "admin_router"]
