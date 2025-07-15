import google.generativeai as genai
import json
import requests
import time
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime
import asyncio
import random
from pathlib import Path

from ..utils.config import get_settings
from ..utils.free_weather import get_weather_for_hotel_recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotelDataCollector:
    """Collects hotel data using Gemini API and free weather sources"""
    
    def __init__(self):
        """Initialize the data collector"""
        self.settings = get_settings()
        
        if not self.settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Create data directory if it doesn't exist
        self.data_path = Path("./data")
        self.data_path.mkdir(exist_ok=True)
        
        logger.info("HotelDataCollector initialized with free weather service")
    
    def get_location_coordinates(self, location: str) -> Optional[Dict[str, float]]:
        """Get latitude and longitude for a location using free Nominatim service"""
        try:
            # Use free Nominatim service from OpenStreetMap
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location,
                "format": "json",
                "limit": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "latitude": float(data[0]["lat"]),
                        "longitude": float(data[0]["lon"]),
                        "display_name": data[0]["display_name"]
                    }
        except Exception as e:
            logger.error(f"Geocoding error for {location}: {str(e)}")
        return None
    
    def get_weather_data(self, location: str) -> Dict:
        """Get weather data for a location using free weather service"""
        try:
            weather_info = get_weather_for_hotel_recommendation(location)
            logger.info(f"Retrieved free weather data for {location}")
            return weather_info
        except Exception as e:
            logger.warning(f"Free weather service failed for {location}: {e}")
            return {
                "temperature": 22,
                "condition": "Pleasant",
                "season": "spring",
                "travel_recommendation": "Good time to visit",
                "recommended_activities": ["sightseeing", "dining"],
                "weather_score": 0.7,
                "best_hotel_types": ["city hotel", "boutique hotel"]
            }
    
    def generate_hotel_prompt(self, location: str, hotel_type: str = "hotels") -> str:
        """Generate prompt for Gemini API to get hotel information"""
        return f"""
        Please provide detailed information about {self.settings.MAX_HOTELS_PER_LOCATION} popular {hotel_type} in {location}.
        
        For each property, include the following information in JSON format:
        
        {{
            "name": "Hotel Name",
            "address": "Full address",
            "rating": 4.5,
            "price_range": "$$$ (150-300 USD per night)",
            "property_type": "hotel/resort/villa/apartment",
            "amenities": ["wifi", "pool", "spa", "restaurant", "gym"],
            "room_types": ["standard", "deluxe", "suite"],
            "description": "Brief description of the property",
            "highlights": ["key features", "unique selling points"],
            "contact_info": {{
                "phone": "+1-234-567-8900",
                "email": "info@hotel.com",
                "website": "https://hotel.com"
            }},
            "location_details": {{
                "neighborhood": "District/Area name",
                "nearby_attractions": ["attraction1", "attraction2"],
                "transportation": "Access to public transport",
                "distance_to_center": "2.5 km"
            }},
            "guest_reviews": {{
                "total_reviews": 1250,
                "average_rating": 4.3,
                "recent_feedback": "Recent guest comments"
            }},
            "pricing_details": {{
                "low_season": "100-200 USD",
                "high_season": "200-400 USD",
                "currency": "USD"
            }},
            "booking_info": {{
                "availability": "Available",
                "cancellation_policy": "Free cancellation up to 24 hours",
                "check_in": "15:00",
                "check_out": "11:00"
            }}
        }}
        
        Please ensure all information is realistic and detailed. Focus on popular, well-reviewed properties.
        Return only a valid JSON array of hotel objects.
        """
    
    def parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini API response and extract hotel data"""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Try to find JSON content
            start_idx = cleaned_text.find('[')
            end_idx = cleaned_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_text = cleaned_text[start_idx:end_idx]
                hotels = json.loads(json_text)
                
                # Validate and clean the data
                validated_hotels = []
                for hotel in hotels:
                    if isinstance(hotel, dict) and 'name' in hotel:
                        # Ensure required fields exist
                        hotel.setdefault('rating', 4.0)
                        hotel.setdefault('price_range', '$$ (100-250 USD per night)')
                        hotel.setdefault('amenities', ['wifi', 'restaurant'])
                        hotel.setdefault('property_type', 'hotel')
                        
                        validated_hotels.append(hotel)
                
                return validated_hotels
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Response parsing error: {str(e)}")
        
        return []
    
    def collect_hotels_for_location(self, location: str, hotel_types: Optional[List[str]] = None) -> List[Dict]:
        """Collect hotel data for a specific location"""
        if hotel_types is None:
            hotel_types = ["hotels", "resorts", "villas", "apartments"]
        
        all_hotels = []
        
        # Get location coordinates and weather
        coordinates = self.get_location_coordinates(location)
        weather_data = self.get_weather_data(location)
        
        for hotel_type in hotel_types:
            try:
                logger.info(f"Collecting {hotel_type} data for {location}")
                
                # Generate prompt
                prompt = self.generate_hotel_prompt(location, hotel_type)
                
                # Get response from Gemini
                response = self.model.generate_content(prompt)
                
                if response.text:
                    hotels = self.parse_gemini_response(response.text)
                    
                    # Add metadata to each hotel
                    for hotel in hotels:
                        hotel['location'] = location
                        hotel['property_category'] = hotel_type
                        hotel['collected_at'] = datetime.now().isoformat()
                        
                        # Add coordinates if available
                        if coordinates:
                            hotel['coordinates'] = coordinates
                        
                        # Add weather information
                        hotel['weather_info'] = weather_data
                        
                        # Add derived features for ML
                        hotel['features'] = self.extract_features(hotel, weather_data)
                    
                    all_hotels.extend(hotels)
                    logger.info(f"Collected {len(hotels)} {hotel_type} for {location}")
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting {hotel_type} for {location}: {str(e)}")
                continue
        
        return all_hotels
    
    def extract_features(self, hotel: Dict, weather_data: Dict) -> Dict:
        """Extract features for machine learning"""
        features = {}
        
        # Price features
        price_range = hotel.get('price_range', '').lower()
        if '$' in price_range:
            dollar_count = price_range.count('$')
            features['price_category'] = dollar_count  # 1-4 scale
        else:
            features['price_category'] = 2  # Default
        
        # Rating features
        features['rating'] = float(hotel.get('rating', 4.0))
        
        # Amenity features
        amenities = hotel.get('amenities', [])
        features['amenity_count'] = len(amenities)
        features['has_pool'] = 'pool' in [a.lower() for a in amenities]
        features['has_spa'] = 'spa' in [a.lower() for a in amenities]
        features['has_wifi'] = 'wifi' in [a.lower() for a in amenities]
        features['has_restaurant'] = 'restaurant' in [a.lower() for a in amenities]
        features['has_gym'] = 'gym' in [a.lower() for a in amenities]
        
        # Property type features
        prop_type = hotel.get('property_type', 'hotel').lower()
        features['is_resort'] = 'resort' in prop_type
        features['is_villa'] = 'villa' in prop_type
        features['is_apartment'] = 'apartment' in prop_type
        features['is_hotel'] = 'hotel' in prop_type
        
        # Weather-based features
        features['temperature'] = weather_data.get('temperature', 22)
        features['weather_score'] = weather_data.get('weather_score', 0.7)
        features['season'] = weather_data.get('season', 'spring')
        
        # Review features
        guest_reviews = hotel.get('guest_reviews', {})
        features['total_reviews'] = guest_reviews.get('total_reviews', 100)
        features['review_rating'] = guest_reviews.get('average_rating', features['rating'])
        
        return features
    
    def save_hotels_json(self, hotels: List[Dict], location: str):
        """Save hotel data to JSON file"""
        filename = self.data_path / f"{location.replace(' ', '_').lower()}_hotels.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(hotels, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(hotels)} hotels to {filename}")
    
    def save_hotels_csv(self, hotels: List[Dict], location: str):
        """Save hotel data to CSV file"""
        if not hotels:
            return
        
        # Flatten the data for CSV
        flattened_data = []
        for hotel in hotels:
            flat_hotel = {}
            
            # Basic information
            flat_hotel.update({
                'name': hotel.get('name', ''),
                'location': hotel.get('location', ''),
                'address': hotel.get('address', ''),
                'rating': hotel.get('rating', 0),
                'price_range': hotel.get('price_range', ''),
                'property_type': hotel.get('property_type', ''),
                'property_category': hotel.get('property_category', ''),
                'description': hotel.get('description', ''),
                'collected_at': hotel.get('collected_at', '')
            })
            
            # Amenities as comma-separated string
            amenities = hotel.get('amenities', [])
            flat_hotel['amenities'] = ', '.join(amenities) if amenities else ''
            
            # Contact information
            contact = hotel.get('contact_info', {})
            flat_hotel.update({
                'phone': contact.get('phone', ''),
                'email': contact.get('email', ''),
                'website': contact.get('website', '')
            })
            
            # Location details
            location_details = hotel.get('location_details', {})
            flat_hotel.update({
                'neighborhood': location_details.get('neighborhood', ''),
                'distance_to_center': location_details.get('distance_to_center', ''),
                'transportation': location_details.get('transportation', '')
            })
            
            # Reviews
            reviews = hotel.get('guest_reviews', {})
            flat_hotel.update({
                'total_reviews': reviews.get('total_reviews', 0),
                'review_rating': reviews.get('average_rating', 0)
            })
            
            # Weather information
            weather = hotel.get('weather_info', {})
            flat_hotel.update({
                'temperature': weather.get('temperature', 0),
                'weather_condition': weather.get('condition', ''),
                'season': weather.get('season', ''),
                'weather_score': weather.get('weather_score', 0)
            })
            
            # Features
            features = hotel.get('features', {})
            flat_hotel.update(features)
            
            flattened_data.append(flat_hotel)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened_data)
        filename = self.data_path / f"{location.replace(' ', '_').lower()}_hotels.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(hotels)} hotels to {filename}")
    
    async def collect_multiple_locations(self, locations: List[str]) -> Dict[str, List[Dict]]:
        """Collect hotel data for multiple locations"""
        all_location_data = {}
        
        for location in locations:
            try:
                logger.info(f"Starting data collection for {location}")
                hotels = self.collect_hotels_for_location(location)
                
                if hotels:
                    all_location_data[location] = hotels
                    
                    # Save data
                    self.save_hotels_json(hotels, location)
                    self.save_hotels_csv(hotels, location)
                    
                    logger.info(f"Completed data collection for {location}: {len(hotels)} properties")
                else:
                    logger.warning(f"No data collected for {location}")
                
                # Rate limiting between locations
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error processing {location}: {str(e)}")
                continue
        
        return all_location_data

# Example usage
if __name__ == "__main__":
    collector = HotelDataCollector()
    
    # Test locations
    test_locations = ["Paris, France", "Tokyo, Japan", "New York, USA"]
    
    async def main():
        data = await collector.collect_multiple_locations(test_locations)
        print(f"Collected data for {len(data)} locations")
    
    asyncio.run(main())
