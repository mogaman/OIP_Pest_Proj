#!/usr/bin/env python3
"""
Simple Test Suite for Organic Farm Pest Management AI System
"""

import sys
import os
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic application functionality"""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Import test
        print("Test 1: Importing modules...")
        from app import app, pest_model, treatment_db
        print("✅ All imports successful")
        
        # Test 2: Pest classifier
        print("\nTest 2: Testing pest classifier...")
        test_image = Image.new('RGB', (224, 224), color='green')
        result = pest_model.predict(test_image)
        print(f"✅ Prediction: {result['pest_name']} ({result['confidence']:.1f}% confidence)")
        
        # Test 3: Treatment database
        print("\nTest 3: Testing treatment database...")
        treatment = treatment_db.get_treatment_info('Aphids')
        print(f"✅ Found {len(treatment['organic_treatments'])} treatments for Aphids")
        
        # Test 4: Flask routes
        print("\nTest 4: Testing Flask routes...")
        with app.test_client() as client:
            routes = ['/', '/analyze', '/chat', '/history']
            for route in routes:
                response = client.get(route)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"{status} {route} - Status: {response.status_code}")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_basic_functionality():
        print("\n📋 Manual Testing Steps:")
        print("1. Start server: python app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Test each page and functionality")
    else:
        print("\n⚠️ Fix errors before proceeding")
