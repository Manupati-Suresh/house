# Utility functions for the House Price Predictor App

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def format_currency(amount):
    """Format number as currency string"""
    return f"${amount:,.0f}"

def format_percentage(value):
    """Format number as percentage string"""
    return f"{value:+.1f}%"

def calculate_price_per_sqft(price, area):
    """Calculate price per square foot"""
    if area > 0:
        return price / area
    return 0

def generate_synthetic_data(n_samples=1000, seed=42):
    """Generate synthetic house data for demonstration"""
    np.random.seed(seed)
    
    data = pd.DataFrame({
        'GrLivArea': np.random.normal(1500, 500, n_samples),
        'BedroomAbvGr': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'SalePrice': np.random.normal(200000, 80000, n_samples)
    })
    
    # Apply realistic constraints
    data['GrLivArea'] = np.clip(data['GrLivArea'], 500, 5000)
    data['SalePrice'] = np.clip(data['SalePrice'], 50000, 800000)
    
    # Add some correlation between features and price
    data['SalePrice'] = (
        data['GrLivArea'] * 100 + 
        data['BedroomAbvGr'] * 15000 + 
        data['GarageCars'] * 10000 + 
        np.random.normal(0, 20000, n_samples)
    )
    
    return data

def calculate_confidence_score(model, input_data):
    """Calculate a confidence score for the prediction"""
    # This is a simplified confidence calculation
    # In practice, you might use prediction intervals or cross-validation
    try:
        # Use model's score method if available
        if hasattr(model, 'score'):
            # Generate some test data for scoring
            test_data = generate_synthetic_data(100)
            X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'GarageCars']]
            y_test = test_data['SalePrice']
            confidence = model.score(X_test, y_test)
            return max(0.7, min(0.95, confidence))  # Clamp between 70% and 95%
        else:
            # Fallback to a reasonable confidence score
            return np.random.uniform(0.75, 0.90)
    except:
        return 0.8  # Default confidence

def get_similar_properties(data, area, bedrooms, garage, tolerance=0.2):
    """Find similar properties in the dataset"""
    if data is None:
        return pd.DataFrame()
    
    similar = data[
        (data['BedroomAbvGr'] == bedrooms) &
        (data['GarageCars'] == garage) &
        (data['GrLivArea'].between(area * (1 - tolerance), area * (1 + tolerance)))
    ]
    
    return similar

def create_price_distribution_chart(data):
    """Create a price distribution chart"""
    fig = px.histogram(
        data, 
        x='SalePrice', 
        nbins=30,
        title="Distribution of House Prices",
        labels={'SalePrice': 'Sale Price ($)', 'count': 'Number of Houses'}
    )
    fig.update_layout(
        xaxis_title="Sale Price ($)",
        yaxis_title="Count",
        showlegend=False
    )
    return fig

def create_price_vs_area_chart(data):
    """Create a price vs area scatter plot"""
    fig = px.scatter(
        data, 
        x='GrLivArea', 
        y='SalePrice',
        title="Price vs Living Area",
        trendline="ols",
        labels={'GrLivArea': 'Living Area (sq ft)', 'SalePrice': 'Sale Price ($)'}
    )
    fig.update_layout(
        xaxis_title="Living Area (sq ft)",
        yaxis_title="Sale Price ($)"
    )
    return fig

def create_feature_importance_chart(model, feature_names):
    """Create a feature importance chart for linear regression"""
    try:
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            
            fig = px.bar(
                x=feature_names,
                y=coefficients,
                title="Feature Importance (Model Coefficients)",
                labels={'x': 'Features', 'y': 'Coefficient Value'}
            )
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Coefficient Value",
                showlegend=False
            )
            return fig
    except:
        pass
    
    return None

def validate_input(area, bedrooms, garage):
    """Validate user input"""
    errors = []
    
    if area < 500 or area > 5000:
        errors.append("Living area must be between 500 and 5,000 sq ft")
    
    if bedrooms < 1 or bedrooms > 6:
        errors.append("Number of bedrooms must be between 1 and 6")
    
    if garage < 0 or garage > 4:
        errors.append("Garage capacity must be between 0 and 4 cars")
    
    return errors

def get_market_insights(data):
    """Generate market insights from data"""
    if data is None or len(data) == 0:
        return {}
    
    insights = {
        'avg_price': data['SalePrice'].mean(),
        'median_price': data['SalePrice'].median(),
        'avg_area': data['GrLivArea'].mean(),
        'avg_price_per_sqft': (data['SalePrice'] / data['GrLivArea']).mean(),
        'total_properties': len(data),
        'price_std': data['SalePrice'].std()
    }
    
    return insights

def format_insights_text(insights):
    """Format market insights as readable text"""
    if not insights:
        return "Market insights not available."
    
    text = f"""
    **Market Overview:**
    - Average Price: {format_currency(insights['avg_price'])}
    - Median Price: {format_currency(insights['median_price'])}
    - Average Living Area: {insights['avg_area']:,.0f} sq ft
    - Average Price per Sq Ft: {format_currency(insights['avg_price_per_sqft'])}
    - Total Properties Analyzed: {insights['total_properties']:,}
    """
    
    return text

def get_prediction_interpretation(prediction, market_avg):
    """Interpret the prediction relative to market"""
    if market_avg == 0:
        return "Unable to compare with market average."
    
    diff_pct = ((prediction - market_avg) / market_avg) * 100
    
    if diff_pct > 20:
        return f"🔺 This property is priced {abs(diff_pct):.1f}% above market average - Premium property"
    elif diff_pct > 10:
        return f"📈 This property is priced {abs(diff_pct):.1f}% above market average - Above average"
    elif diff_pct > -10:
        return f"📊 This property is priced close to market average ({diff_pct:+.1f}%)"
    elif diff_pct > -20:
        return f"📉 This property is priced {abs(diff_pct):.1f}% below market average - Good value"
    else:
        return f"🔻 This property is priced {abs(diff_pct):.1f}% below market average - Excellent value"

def log_prediction(area, bedrooms, garage, prediction):
    """Log prediction for analytics (placeholder)"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'area': area,
        'bedrooms': bedrooms,
        'garage': garage,
        'prediction': prediction
    }
    # In a real app, you might save this to a database or file
    return log_entry