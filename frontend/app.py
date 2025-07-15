import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from typing import Dict, List, Any
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="TravelMind - AI Hotel Recommendations",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.hotel-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background: #f9f9f9;
}
.recommendation-score {
    font-size: 1.2rem;
    font-weight: bold;
    color: #28a745;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🏨 TravelMind</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Hotel Recommendation System</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🎯 Get Recommendations", "🔍 Search Hotels", "📊 System Status", "⚙️ Admin Panel"]
    )
    
    if page == "🎯 Get Recommendations":
        recommendations_page()
    elif page == "🔍 Search Hotels":
        search_hotels_page()
    elif page == "📊 System Status":
        system_status_page()
    elif page == "⚙️ Admin Panel":
        admin_panel_page()

def recommendations_page():
    """Recommendations page"""
    st.header("🎯 Get Personalized Hotel Recommendations")
    
    # Check API health
    if not check_api_health():
        st.error("❌ API is not available. Please start the backend server.")
        return
    
    # User preferences form
    with st.form("preferences_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Travel Details")
            location = st.text_input("Destination", placeholder="e.g., Paris, France")
            
            st.subheader("💰 Budget")
            budget_range = st.slider("Budget per night (USD)", 0, 1000, (100, 300))
            
            st.subheader("⭐ Preferences")
            min_rating = st.slider("Minimum rating", 0.0, 5.0, 3.0, 0.1)
            
        with col2:
            st.subheader("🏨 Property & Travel Type")
            property_type = st.selectbox(
                "Property Type",
                ["Any", "Hotel", "Villa", "Resort", "Apartment", "Bed & Breakfast"]
            )
            
            season = st.selectbox(
                "Preferred Season",
                ["Any", "Spring", "Summer", "Fall", "Winter"]
            )
            
            st.subheader("👥 Travel Style")
            family_travel = st.checkbox("Family Travel")
            business_travel = st.checkbox("Business Travel")
            group_size = st.number_input("Number of travelers", 1, 20, 2)
        
        st.subheader("🛎️ Preferred Amenities")
        amenities_options = [
            "WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", 
            "Parking", "Pet Friendly", "Business Center", "Room Service"
        ]
        amenities = st.multiselect("Select amenities", amenities_options)
        
        recommendation_type = st.selectbox(
            "Recommendation Type",
            ["Hybrid (Recommended)", "Content-Based", "Value-Based", "Luxury", "Family-Friendly"]
        )
        
        top_k = st.slider("Number of recommendations", 5, 20, 10)
        
        submitted = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)
    
    if submitted and location:
        get_recommendations(
            location, budget_range, min_rating, property_type, season,
            family_travel, business_travel, group_size, amenities,
            recommendation_type, top_k
        )

def get_recommendations(location, budget_range, min_rating, property_type, season,
                       family_travel, business_travel, group_size, amenities,
                       recommendation_type, top_k):
    """Get and display recommendations"""
    
    # Map recommendation types
    rec_type_map = {
        "Hybrid (Recommended)": "hybrid",
        "Content-Based": "content", 
        "Value-Based": "value_based",
        "Luxury": "luxury",
        "Family-Friendly": "family"
    }
    
    # Prepare request data
    user_preferences = {
        "location": location,
        "budget_min": budget_range[0],
        "budget_max": budget_range[1],
        "min_rating": min_rating,
        "preferred_amenities": amenities,
        "family_travel": family_travel,
        "business_travel": business_travel,
        "group_size": group_size
    }
    
    if property_type != "Any":
        user_preferences["property_type_preference"] = property_type.lower().replace(" & ", "_")
    
    if season != "Any":
        user_preferences["preferred_season"] = season.lower()
    
    request_data = {
        "user_preferences": user_preferences,
        "recommendation_type": rec_type_map.get(recommendation_type, "hybrid"),
        "top_k": top_k,
        "include_similar": False
    }
    
    # Show loading spinner
    with st.spinner("🤖 AI is analyzing hotels and generating recommendations..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/recommendations",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                display_recommendations(data)
            else:
                error_data = response.json() if response.content else {}
                st.error(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")

def display_recommendations(data):
    """Display recommendation results"""
    recommendations = data.get("recommendations", [])
    total_found = data.get("total_found", 0)
    processing_time = data.get("processing_time_ms", 0)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(recommendations)}</h3>
            <p>Recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_found}</h3>
            <p>Hotels Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_score = np.mean([r["recommendation_score"] for r in recommendations]) if recommendations else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_score:.1f}/5</h3>
            <p>Avg Match Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{processing_time:.0f}ms</h3>
            <p>Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    if not recommendations:
        st.warning("No recommendations found. Try adjusting your criteria.")
        return
    
    # Display recommendations
    st.header("🏆 Recommended Hotels")
    
    for i, rec in enumerate(recommendations):
        hotel = rec["hotel"]
        score = rec["recommendation_score"]
        rank = rec["recommendation_rank"]
        reasoning = rec.get("reasoning", "")
        
        with st.container():
            st.markdown(f"""
            <div class="hotel-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3>#{rank} {hotel["name"]}</h3>
                    <span class="recommendation-score">🎯 {score:.1f}/5.0</span>
                </div>
                <p><strong>📍 Location:</strong> {hotel["location"]}</p>
                <p><strong>⭐ Rating:</strong> {hotel["user_rating"]}/5.0 ({hotel["star_rating"]} stars)</p>
                <p><strong>💰 Price:</strong> ${hotel["min_price"]:.0f} - ${hotel["max_price"]:.0f} per night</p>
                <p><strong>🛎️ Amenities:</strong> {", ".join(hotel.get("amenities", [])[:5])}</p>
                {f'<p><strong>💡 Why recommended:</strong> {reasoning}</p>' if reasoning else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Rating buttons for feedback
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button(f"👍 Like", key=f"like_{i}"):
                    submit_feedback(hotel.get("id", i), 5.0, "Positive feedback")
            with col2:
                if st.button(f"👎 Dislike", key=f"dislike_{i}"):
                    submit_feedback(hotel.get("id", i), 1.0, "Negative feedback")

def search_hotels_page():
    """Hotel search page"""
    st.header("🔍 Search Hotels")
    
    if not check_api_health():
        st.error("❌ API is not available. Please start the backend server.")
        return
    
    # Search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        location = st.text_input("Location", placeholder="e.g., Tokyo, Japan")
    with col2:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
    with col3:
        max_price = st.number_input("Max Price per Night", 0, 2000, 500)
    
    amenities = st.text_input("Amenities (comma-separated)", placeholder="wifi, pool, gym")
    limit = st.slider("Maximum Results", 10, 100, 50)
    
    if st.button("🔍 Search Hotels"):
        search_hotels(location, min_rating, max_price, amenities, limit)

def search_hotels(location, min_rating, max_price, amenities, limit):
    """Search for hotels"""
    params = {"limit": limit}
    
    if location:
        params["location"] = location
    if min_rating > 0:
        params["min_rating"] = min_rating
    if max_price > 0:
        params["max_price"] = max_price
    if amenities:
        params["amenities"] = amenities
    
    try:
        response = requests.get(f"{API_BASE_URL}/hotels", params=params)
        
        if response.status_code == 200:
            data = response.json()
            hotels = data.get("hotels", [])
            
            if hotels:
                st.success(f"Found {len(hotels)} hotels")
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(hotels)
                
                # Select relevant columns
                display_cols = ["name", "location", "star_rating", "user_rating", "min_price", "max_price"]
                if all(col in df.columns for col in display_cols):
                    st.dataframe(df[display_cols], use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
                
                # Show price distribution
                if "min_price" in df.columns and "max_price" in df.columns:
                    fig = px.histogram(df, x="min_price", title="Price Distribution", 
                                     labels={"min_price": "Price per Night (USD)"})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hotels found matching your criteria.")
        else:
            st.error(f"Search failed: {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def system_status_page():
    """System status page"""
    st.header("📊 System Status")
    
    if not check_api_health():
        st.error("❌ API is not available. Please start the backend server.")
        return
    
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        
        if response.status_code == 200:
            data = response.json()
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Hotels", data.get("total_hotels", 0))
            with col2:
                st.metric("Unique Locations", data.get("unique_locations", 0))
            with col3:
                st.metric("API Version", data.get("api_version", "1.0.0"))
            
            # Model status
            st.subheader("🤖 Model Status")
            models = data.get("models_status", [])
            
            if models:
                model_data = []
                for model in models:
                    model_info = {
                        "Model": model.get("model_name", "Unknown"),
                        "Status": "✅ Trained" if model.get("is_trained") else "❌ Not Trained",
                        "Last Trained": model.get("last_trained", "Never")
                    }
                    
                    metrics = model.get("metrics")
                    if metrics:
                        model_info["R² Score"] = f"{metrics.get('r2', 0):.3f}"
                        model_info["MAE"] = f"{metrics.get('mae', 0):.3f}"
                    
                    model_data.append(model_info)
                
                st.table(pd.DataFrame(model_data))
            else:
                st.warning("No model information available")
        else:
            st.error("Failed to fetch system status")
            
    except Exception as e:
        st.error(f"Error fetching status: {str(e)}")

def admin_panel_page():
    """Admin panel for managing the system"""
    st.header("⚙️ Admin Panel")
    
    if not check_api_health():
        st.error("❌ API is not available. Please start the backend server.")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎓 Train Models", "📥 Collect Data", "📈 Database Stats"])
    
    with tab1:
        st.subheader("🎓 Train ML Models")
        st.info("Train recommendation models with data from specified locations")
        
        locations_text = st.text_area(
            "Training Locations (one per line)",
            value="Paris, France\nTokyo, Japan\nNew York, USA\nLondon, UK\nBarcelona, Spain",
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            retrain_existing = st.checkbox("Retrain with existing data")
        with col2:
            hyperparameter_tuning = st.checkbox("Enable hyperparameter tuning", value=True)
        
        if st.button("🚀 Start Training", type="primary"):
            locations = [loc.strip() for loc in locations_text.split('\n') if loc.strip()]
            start_training(locations, retrain_existing, hyperparameter_tuning)
    
    with tab2:
        st.subheader("📥 Collect Hotel Data")
        st.info("Collect fresh hotel data from specified locations using Gemini API")
        
        data_locations = st.text_input(
            "Locations (comma-separated)",
            placeholder="Paris, Tokyo, New York"
        )
        
        force_refresh = st.checkbox("Force refresh existing data")
        
        if st.button("📊 Collect Data"):
            if data_locations:
                locations = [loc.strip() for loc in data_locations.split(',')]
                collect_data(locations, force_refresh)
            else:
                st.error("Please enter at least one location")
    
    with tab3:
        st.subheader("📈 Database Statistics")
        show_database_stats()

def start_training(locations, retrain_existing, hyperparameter_tuning):
    """Start model training"""
    request_data = {
        "locations": locations,
        "retrain_existing": retrain_existing,
        "hyperparameter_tuning": hyperparameter_tuning
    }
    
    try:
        with st.spinner("Starting training process..."):
            response = requests.post(f"{API_BASE_URL}/admin/train", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Training started successfully!")
            st.info(f"Training ID: {data.get('training_id')}")
            st.info(f"Estimated duration: {data.get('estimated_duration_minutes', 'Unknown')} minutes")
        else:
            st.error(f"Training failed: {response.text}")
            
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")

def collect_data(locations, force_refresh):
    """Collect hotel data"""
    request_data = {
        "locations": locations,
        "force_refresh": force_refresh
    }
    
    try:
        with st.spinner("Starting data collection..."):
            response = requests.post(f"{API_BASE_URL}/admin/collect-data", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Data collection started!")
            st.info(f"Collection ID: {data.get('collection_id')}")
        else:
            st.error(f"Data collection failed: {response.text}")
            
    except Exception as e:
        st.error(f"Error starting data collection: {str(e)}")

def show_database_stats():
    """Show database statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/admin/database/stats")
        
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Hotels", stats.get("total_hotels", 0))
                st.metric("User Preferences", stats.get("user_preferences", 0))
            
            with col2:
                st.metric("Unique Locations", stats.get("unique_locations", 0))
                st.metric("Total Recommendations", stats.get("total_recommendations", 0))
            
            with col3:
                st.metric("Total Feedback", stats.get("total_feedback", 0))
        else:
            st.error("Failed to fetch database statistics")
            
    except Exception as e:
        st.error(f"Error fetching stats: {str(e)}")

def submit_feedback(hotel_id, rating, feedback_text):
    """Submit user feedback"""
    feedback_data = {
        "user_id": "streamlit_user",  # In production, this would be actual user ID
        "hotel_id": hotel_id,
        "rating": rating,
        "feedback_text": feedback_text
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/feedback", json=feedback_data)
        
        if response.status_code == 200:
            st.success("Thank you for your feedback!")
        else:
            st.error("Failed to submit feedback")
            
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    main()
