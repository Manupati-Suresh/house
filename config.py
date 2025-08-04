# Configuration file for the House Price Predictor App

# App Configuration
APP_CONFIG = {
    "title": "🏠 Advanced House Price Predictor",
    "page_icon": "🏠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model Configuration
MODEL_CONFIG = {
    "model_file": "linear_model.pkl",
    "features": ["GrLivArea", "BedroomAbvGr", "GarageCars"],
    "feature_names": ["Living Area", "Bedrooms", "Garage Cars"],
    "confidence_threshold": 0.8
}

# UI Configuration
UI_CONFIG = {
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e",
    "success_color": "#2ca02c",
    "warning_color": "#ff7f0e",
    "error_color": "#d62728"
}

# Data Configuration
DATA_CONFIG = {
    "min_area": 500,
    "max_area": 5000,
    "default_area": 1500,
    "area_step": 50,
    "bedroom_options": [1, 2, 3, 4, 5, 6],
    "garage_options": [0, 1, 2, 3, 4]
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "price_range_factor": 0.15,  # ±15% for price range
    "similar_property_tolerance": 0.2,  # ±20% for similar properties
    "confidence_levels": {
        "high": 0.9,
        "medium": 0.8,
        "low": 0.7
    }
}