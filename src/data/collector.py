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

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI not available. Install with: pip install google-generativeai")

from ..utils.config import get_settings
from ..utils.free_weather import get_weather_for_hotel_recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotelDataCollector:
    """Collects hotel data using Gemini API and other free sources"""
    
    def __init__(self):
        """Initialize the data collector"""
        self.settings = get_settings()
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not available. Install with: pip install google-generativeai")
        
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
    
    def generate_hotel_prompt(self, location: str, hotel_type: str = "hotels") -> str:
        """Generate prompt for Gemini API to get hotel information"""
        return f"""
        Please provide detailed information about {self.settings.MAX_HOTELS_PER_LOCATION} popular {hotel_type} in {location}.
        For each hotel, provide the following information in JSON format:
        
        {{
            "hotels": [
                {{
                    "name": "Hotel Name",
                    "address": "Full address",
                    "star_rating": 4.5,
                    "user_rating": 4.2,
                    "price_range": "$$$ (150-250 USD per night)",
                    "amenities": ["WiFi", "Pool", "Gym", "Restaurant", "Spa"],
                    "room_types": ["Standard", "Deluxe", "Suite"],
                    "description": "Brief description",
                    "best_season": "Winter/Summer/Spring/Fall",
                    "nearby_attractions": ["Attraction 1", "Attraction 2"],
                    "contact_info": {{
                        "phone": "+1234567890",
                        "email": "contact@hotel.com",
                        "website": "https://hotel.com"
                    }},
                    "sustainability_rating": 3.5,
                    "business_facilities": ["Conference Room", "Business Center"],
                    "family_friendly": true,
                    "pet_friendly": false,
                    "accessibility": ["Wheelchair accessible", "Braille signage"]
                }}
            ]
        }}
        
        Please ensure all data is realistic and diverse. Include both luxury and budget options.
        Make sure to vary the ratings, prices, and amenities to create a realistic dataset.
        """
    
    async def collect_hotels_from_gemini(self, location: str) -> List[Dict]:
        """Collect hotel data from Gemini API"""
        try:
            # Get both hotels and villas/residences
            hotel_types = ["hotels", "villas and vacation rentals", "bed and breakfasts"]
            all_hotels = []
            
            for hotel_type in hotel_types:
                prompt = self.generate_hotel_prompt(location, hotel_type)
                
                response = self.model.generate_content(prompt)
                
                # Parse the JSON response
                try:
                    # Extract JSON from the response
                    response_text = response.text
                    
                    # Find JSON content between ```json and ```
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != 0:
                        json_str = response_text[start_idx:end_idx]
                        data = json.loads(json_str)
                        
                        if 'hotels' in data:
                            hotels = data['hotels']
                            
                            # Add metadata
                            for hotel in hotels:
                                hotel['location'] = location
                                hotel['property_type'] = hotel_type
                                hotel['collected_at'] = datetime.now().isoformat()
                                
                                # Ensure numeric fields
                                hotel['star_rating'] = float(hotel.get('star_rating', 3.0))
                                hotel['user_rating'] = float(hotel.get('user_rating', 3.5))
                                
                                # Extract price range numeric values
                                price_text = hotel.get('price_range', '$100-200')
                                hotel['min_price'], hotel['max_price'] = self._extract_price_range(price_text)
                            
                            all_hotels.extend(hotels)
                            logger.info(f"Collected {len(hotels)} {hotel_type} from {location}")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {hotel_type} in {location}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing {hotel_type} data: {str(e)}")
                
                # Rate limiting
                await asyncio.sleep(2)
            
            return all_hotels
            
        except Exception as e:
            logger.error(f"Error collecting hotels from Gemini for {location}: {str(e)}")
            return []
    
    def _extract_price_range(self, price_text: str) -> tuple:
        """Extract min and max price from price range text"""
        try:
            import re
            numbers = re.findall(r'\d+', price_text)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
            elif len(numbers) == 1:
                price = float(numbers[0])
                return price * 0.8, price * 1.2  # Assume 20% range
            else:
                return 100.0, 200.0  # Default range
        except:
            return 100.0, 200.0
    
    def enrich_with_seasonal_data(self, hotels: List[Dict], location: str) -> List[Dict]:
        """Enrich hotel data with seasonal information"""
        # Add seasonal pricing and occupancy patterns
        seasons = ['spring', 'summer', 'fall', 'winter']
        
        for hotel in hotels:
            seasonal_data = {}
            base_min, base_max = hotel['min_price'], hotel['max_price']
            
            for season in seasons:
                # Simulate seasonal pricing variations
                if season == 'summer':  # Peak season
                    multiplier = random.uniform(1.2, 1.5)
                elif season == 'winter':  # Off season (varies by location)
                    multiplier = random.uniform(0.8, 1.1)
                else:  # Shoulder seasons
                    multiplier = random.uniform(0.9, 1.2)
                
                seasonal_data[season] = {
                    'min_price': round(base_min * multiplier, 2),
                    'max_price': round(base_max * multiplier, 2),
                    'occupancy_rate': random.uniform(0.6, 0.95),
                    'avg_rating': hotel['user_rating'] + random.uniform(-0.3, 0.3)
                }
            
            hotel['seasonal_data'] = seasonal_data
        
        return hotels
    
    async def collect_location_data(self, location: str) -> Dict:
        """Collect comprehensive data for a location"""
        logger.info(f"Starting data collection for {location}")
        
        # Get coordinates
        coordinates = self.get_location_coordinates(location)
        
        # Collect hotel data
        hotels = await self.collect_hotels_from_gemini(location)
        
        # Enrich with seasonal data
        hotels = self.enrich_with_seasonal_data(hotels, location)
        
        # Get weather patterns (using free OpenWeatherMap if API key is available)
        weather_data = await self._get_weather_patterns(coordinates) if coordinates else {}
        
        result = {
            'location': location,
            'coordinates': coordinates,
            'hotels': hotels,
            'weather_patterns': weather_data,
            'collection_timestamp': datetime.now().isoformat(),
            'total_properties': len(hotels)
        }
        
        logger.info(f"Collected {len(hotels)} properties for {location}")
        return result
    
    async def _get_weather_patterns(self, coordinates: Dict) -> Dict:
        """Get weather patterns for seasonal analysis using free weather service"""
        try:
            # Use free weather service instead of paid API
            location_str = f"{coordinates.get('latitude', 0)},{coordinates.get('longitude', 0)}"
            weather_data = get_weather_for_hotel_recommendation(location_str)
            return weather_data
        except Exception as e:
            logger.warning(f"Could not fetch weather data: {str(e)}")
            return self._generate_mock_weather_data()
    
    def _generate_mock_weather_data(self) -> Dict:
        """Generate mock weather data for demonstration"""
        return {
            'spring': {'avg_temp': 20, 'rainfall': 60, 'tourism_score': 0.8},
            'summer': {'avg_temp': 28, 'rainfall': 40, 'tourism_score': 0.9},
            'fall': {'avg_temp': 18, 'rainfall': 70, 'tourism_score': 0.7},
            'winter': {'avg_temp': 12, 'rainfall': 80, 'tourism_score': 0.6}
        }
    
    def save_to_json(self, data: Dict, location: str):
        """Save collected data to JSON file"""
        filename = self.data_path / f"{location.replace(' ', '_').lower()}_hotels.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
    
    def save_to_csv(self, hotels: List[Dict], location: str):
        """Save hotel data to CSV for ML training"""
        if not hotels:
            logger.warning(f"No hotels to save for {location}")
            return
        
        df = pd.DataFrame(hotels)
        
        # Flatten nested dictionaries
        if 'contact_info' in df.columns:
            # Convert contact_info column to list of dicts if needed
            contact_data = []
            for contact_info in df['contact_info']:
                if isinstance(contact_info, dict):
                    contact_data.append(contact_info)
                else:
                    contact_data.append({})
            
            if contact_data:
                contact_df = pd.json_normalize(contact_data)
                contact_df.columns = [f'contact_{col}' for col in contact_df.columns]
                df = pd.concat([df.drop('contact_info', axis=1), contact_df], axis=1)
        
        # Convert lists to strings
        list_columns = ['amenities', 'room_types', 'nearby_attractions', 'business_facilities', 'accessibility']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        filename = self.data_path / f"{location.replace(' ', '_').lower()}_hotels.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"CSV data saved to {filename}")
        return filename

    def get_weather_data(self, location: str) -> Dict:
        """Get weather data for a location using free weather service"""
        if self.use_free_weather:
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
        else:
            # Fallback to simple weather data
            return {
                "temperature": 22,
                "condition": "Pleasant",
                "season": "spring",
                "travel_recommendation": "Good time to visit",
                "recommended_activities": ["sightseeing", "dining"],
                "weather_score": 0.7,
                "best_hotel_types": ["city hotel", "boutique hotel"]
            }

# Example usage and testing
async def main():
    """Example usage of the data collector"""
    collector = HotelDataCollector()
    
    # Test locations
    test_locations = ["Paris, France", "Tokyo, Japan", "New York, USA"]
    
    for location in test_locations:
        try:
            data = await collector.collect_location_data(location)
            collector.save_to_json(data, location)
            collector.save_to_csv(data['hotels'], location)
        except Exception as e:
            logger.error(f"Failed to collect data for {location}: {str(e)}")
        
        # Rate limiting between locations
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
