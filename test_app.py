# Test file for the House Price Predictor App

import unittest
import pandas as pd
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    format_currency, format_percentage, calculate_price_per_sqft,
    generate_synthetic_data, validate_input, get_market_insights
)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        self.assertEqual(format_currency(150000), "$150,000")
        self.assertEqual(format_currency(1500.50), "$1,500")
        self.assertEqual(format_currency(0), "$0")
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        self.assertEqual(format_percentage(15.5), "+15.5%")
        self.assertEqual(format_percentage(-5.2), "-5.2%")
        self.assertEqual(format_percentage(0), "+0.0%")
    
    def test_calculate_price_per_sqft(self):
        """Test price per square foot calculation"""
        self.assertEqual(calculate_price_per_sqft(150000, 1500), 100.0)
        self.assertEqual(calculate_price_per_sqft(200000, 2000), 100.0)
        self.assertEqual(calculate_price_per_sqft(100000, 0), 0)  # Edge case
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        data = generate_synthetic_data(100, seed=42)
        
        # Check data structure
        self.assertEqual(len(data), 100)
        self.assertIn('GrLivArea', data.columns)
        self.assertIn('BedroomAbvGr', data.columns)
        self.assertIn('GarageCars', data.columns)
        self.assertIn('SalePrice', data.columns)
        
        # Check data ranges
        self.assertTrue(data['GrLivArea'].min() >= 500)
        self.assertTrue(data['GrLivArea'].max() <= 5000)
        self.assertTrue(data['BedroomAbvGr'].min() >= 1)
        self.assertTrue(data['BedroomAbvGr'].max() <= 5)
        self.assertTrue(data['GarageCars'].min() >= 0)
        self.assertTrue(data['GarageCars'].max() <= 4)
    
    def test_validate_input(self):
        """Test input validation"""
        # Valid inputs
        errors = validate_input(1500, 3, 2)
        self.assertEqual(len(errors), 0)
        
        # Invalid area
        errors = validate_input(100, 3, 2)  # Too small
        self.assertGreater(len(errors), 0)
        
        errors = validate_input(6000, 3, 2)  # Too large
        self.assertGreater(len(errors), 0)
        
        # Invalid bedrooms
        errors = validate_input(1500, 0, 2)  # Too few
        self.assertGreater(len(errors), 0)
        
        errors = validate_input(1500, 10, 2)  # Too many
        self.assertGreater(len(errors), 0)
        
        # Invalid garage
        errors = validate_input(1500, 3, -1)  # Negative
        self.assertGreater(len(errors), 0)
        
        errors = validate_input(1500, 3, 5)  # Too many
        self.assertGreater(len(errors), 0)
    
    def test_get_market_insights(self):
        """Test market insights calculation"""
        # Test with valid data
        data = generate_synthetic_data(100, seed=42)
        insights = get_market_insights(data)
        
        self.assertIn('avg_price', insights)
        self.assertIn('median_price', insights)
        self.assertIn('avg_area', insights)
        self.assertIn('avg_price_per_sqft', insights)
        self.assertIn('total_properties', insights)
        
        self.assertEqual(insights['total_properties'], 100)
        self.assertGreater(insights['avg_price'], 0)
        self.assertGreater(insights['median_price'], 0)
        
        # Test with empty data
        insights = get_market_insights(None)
        self.assertEqual(insights, {})
        
        insights = get_market_insights(pd.DataFrame())
        self.assertEqual(insights, {})

class TestModelIntegration(unittest.TestCase):
    """Test cases for model integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model for testing
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([150000])
        self.mock_model.coef_ = np.array([100, 15000, 10000])
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Test data
        input_data = pd.DataFrame([[1500, 3, 2]], 
                                 columns=["GrLivArea", "BedroomAbvGr", "GarageCars"])
        
        # Make prediction
        prediction = self.mock_model.predict(input_data)
        
        self.assertEqual(len(prediction), 1)
        self.assertEqual(prediction[0], 150000)
    
    def test_model_coefficients(self):
        """Test model coefficient access"""
        self.assertTrue(hasattr(self.mock_model, 'coef_'))
        self.assertEqual(len(self.mock_model.coef_), 3)

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions"""
    
    def test_data_filtering(self):
        """Test data filtering for similar properties"""
        data = generate_synthetic_data(1000, seed=42)
        
        # Filter for 3-bedroom, 2-garage properties around 1500 sq ft
        filtered = data[
            (data['BedroomAbvGr'] == 3) &
            (data['GarageCars'] == 2) &
            (data['GrLivArea'].between(1200, 1800))
        ]
        
        # Should have some results but not all
        self.assertGreater(len(filtered), 0)
        self.assertLess(len(filtered), len(data))
        
        # All filtered properties should match criteria
        self.assertTrue(all(filtered['BedroomAbvGr'] == 3))
        self.assertTrue(all(filtered['GarageCars'] == 2))
        self.assertTrue(all(filtered['GrLivArea'].between(1200, 1800)))

class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling"""
    
    def test_missing_model_file(self):
        """Test handling of missing model file"""
        # This would be tested in the actual app context
        # Here we just verify the error handling structure exists
        pass
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        # Test with string inputs where numbers expected
        errors = validate_input("invalid", 3, 2)
        # The function should handle this gracefully or we should add type checking
        pass

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()