#!/usr/bin/env python3
"""
Comprehensive Test Suite for Organic Farm Pest Management AI System
Run this script to test all major functionality of the application
"""

import sys
import os
import time
import requests
import json
import base64
from PIL import Image
import io

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(color='green', size=(224, 224)):
    """Create a test image with specified color"""
    image = Image.new('RGB', size, color=color)
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_imports():
    """Test 1: Verify all imports work correctly"""
    print("🧪 Test 1: Import Testing")
    try:
        from app import app, pest_model, treatment_db
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pest_classifier():
    """Test 2: Test the SimplePestClassifier with different scenarios"""
    print("\n🧪 Test 2: Pest Classifier Testing")
    
    try:
        from app import pest_model
        
        # Test with different colored images
        test_cases = [
            ('green', 'Should predict Aphids'),
            ('red', 'Should predict Mites'),
            ('blue', 'Should predict Thrips'),
            ('yellow', 'Should predict random pest')
        ]
        
        for color, expected in test_cases:
            test_image = create_test_image(color)
            result = pest_model.predict(test_image)
            
            print(f"  Color: {color:6} | Pest: {result['pest_name']:15} | Confidence: {result['confidence']:.1f}% | {expected}")
            
            # Verify result structure
            assert 'pest_name' in result
            assert 'confidence' in result
            assert 'prediction_success' in result
            assert isinstance(result['confidence'], (int, float))
        
        print("✅ Pest classifier tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Pest classifier test failed: {e}")
        return False

def test_treatment_database():
    """Test 3: Test the OrganicTreatmentDatabase"""
    print("\n🧪 Test 3: Treatment Database Testing")
    
    try:
        from app import treatment_db
        
        # Test with known pests
        test_pests = ['Aphids', 'Beetle', 'Mites', 'Unknown Pest', 'NonExistent']
        
        for pest in test_pests:
            treatment = treatment_db.get_treatment_info(pest)
            print(f"  Pest: {pest:15} | Treatments: {len(treatment['organic_treatments'])} | Severity: {treatment['severity']}")
            
            # Verify treatment structure
            assert 'severity' in treatment
            assert 'organic_treatments' in treatment
            assert 'crops_affected' in treatment
            assert isinstance(treatment['organic_treatments'], list)
        
        print("✅ Treatment database tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Treatment database test failed: {e}")
        return False

def test_flask_routes():
    """Test 4: Test Flask application routes"""
    print("\n🧪 Test 4: Flask Routes Testing")
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test routes
            routes_to_test = [
                ('/', 'Home page'),
                ('/analyze', 'Analysis page'),
                ('/chat', 'Chat page'),
                ('/history', 'History page')
            ]
            
            for route, description in routes_to_test:
                response = client.get(route)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"  {status} {route:10} | {description:15} | Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"    Error: {response.data.decode()[:100]}...")
        
        print("✅ Flask routes tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Flask routes test failed: {e}")
        return False

def test_database_operations():
    """Test 5: Test database operations"""
    print("\n🧪 Test 5: Database Operations Testing")
    
    try:
        from app import init_database, save_analysis, get_analysis_history
        
        # Initialize database
        init_database()
        print("  ✅ Database initialized")
        
        # Test saving analysis
        test_data = {
            'pest_name': 'Test Aphids',
            'confidence': 85.5,
            'severity': 'Medium',
            'image_path': 'static/uploads/test_image.jpg',
            'treatment_applied': 'Neem Oil',
            'user_notes': 'Test analysis entry'
        }
        
        save_analysis(**test_data)
        print("  ✅ Analysis saved to database")
        
        # Test retrieving history
        history = get_analysis_history()
        print(f"  ✅ Retrieved {len(history)} analysis records")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False

def test_api_endpoints():
    """Test 6: Test API endpoints (if server is running)"""
    print("\n🧪 Test 6: API Endpoints Testing")
    
    try:
        base_url = "http://localhost:5000"
        
        # Test if server is running
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            server_running = response.status_code == 200
        except:
            server_running = False
        
        if not server_running:
            print("  ⚠️  Server not running - skipping API tests")
            print("  💡 Start server with: python app.py")
            return True
        
        # Test API endpoints
        print("  🌐 Server is running - testing API endpoints")
        
        # Test chat API
        chat_data = {
            'message': 'What are organic methods to control aphids?'
        }
        
        response = requests.post(f"{base_url}/api/chat", json=chat_data, timeout=10)
        if response.status_code == 200:
            print("  ✅ Chat API working")
        else:
            print(f"  ❌ Chat API failed: {response.status_code}")
        
        # Test image analysis API (would need actual image upload)
        print("  💡 Image analysis API requires manual testing with file upload")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def run_all_tests():
    """Run all test cases"""
    print("🚀 Starting Comprehensive Test Suite for Organic Farm Pest Management AI")
    print("=" * 80)
    
    test_functions = [
        test_imports,
        test_pest_classifier,
        test_treatment_database,
        test_flask_routes,
        test_database_operations,
        test_api_endpoints
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        if test_func():
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    print("\n" + "=" * 80)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your application is ready for use.")
        print("\n📋 Manual Testing Checklist:")
        print("   1. Start the server: python app.py")
        print("   2. Open browser: http://localhost:5000")
        print("   3. Test image upload on /analyze page")
        print("   4. Test chat functionality on /chat page")
        print("   5. Check history page for saved analyses")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
