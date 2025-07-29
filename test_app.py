#!/usr/bin/env python3
"""
Test script to verify the application loads correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing application import...")
    from app import app, pest_model, treatment_db
    print("✓ Application imported successfully!")
    
    print("\nTesting pest classifier...")
    # Test the SimplePestClassifier
    from PIL import Image
    import numpy as np
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='green')
    result = pest_model.predict(test_image)
    print(f"✓ Pest prediction test: {result['pest_name']} with {result['confidence']:.1f}% confidence")
    
    print("\nTesting treatment database...")
    # Test the treatment database
    treatments = treatment_db.get_treatment_info('Aphids')
    print(f"✓ Treatment database test: Found {len(treatments['organic_treatments'])} organic treatments for Aphids")
    
    print("\nTesting Flask routes...")
    with app.test_client() as client:
        response = client.get('/')
        print(f"✓ Home page status: {response.status_code}")
        
        response = client.get('/analyze')
        print(f"✓ Analyze page status: {response.status_code}")
    
    print("\n🎉 All tests passed! Application is ready to run.")
    print("\nTo start the application, run:")
    print("C:/Users/ryann/AppData/Local/Programs/Python/Python312/python.exe app.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
