"""
Utility functions and testing scripts for OrganicGuard AI
"""

import os
import sys
import sqlite3
import requests
import json
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
from datetime import datetime, timedelta
import random

def test_database_connection():
    """Test database connection and initialization"""
    print("Testing database connection...")
    
    try:
        conn = sqlite3.connect('data/pest_analysis.db')
        cursor = conn.cursor()
        
        # Test table creation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pest_name TEXT,
                confidence REAL,
                severity TEXT,
                image_path TEXT,
                treatment_applied TEXT,
                user_notes TEXT
            )
        ''')
        
        # Test insert
        cursor.execute('''
            INSERT INTO analyses (pest_name, confidence, severity, image_path)
            VALUES (?, ?, ?, ?)
        ''', ('Test Pest', 85.5, 'Medium', 'test_image.jpg'))
        
        # Test select
        cursor.execute('SELECT * FROM analyses WHERE pest_name = ?', ('Test Pest',))
        result = cursor.fetchone()
        
        # Clean up test data
        cursor.execute('DELETE FROM analyses WHERE pest_name = ?', ('Test Pest',))
        
        conn.commit()
        conn.close()
        
        if result:
            print("✅ Database connection test passed!")
            return True
        else:
            print("❌ Database connection test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def create_test_images():
    """Create test images for pest classification"""
    print("Creating test images...")
    
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # Define pest types and colors for visual distinction
    pest_types = {
        'Aphids': (144, 238, 144),      # Light green
        'Beetle': (139, 69, 19),        # Brown
        'Whitefly': (255, 255, 255),    # White
        'Thrips': (255, 255, 0),        # Yellow
        'Mites': (255, 0, 0),           # Red
        'Unknown': (128, 128, 128)       # Gray
    }
    
    for pest_name, color in pest_types.items():
        # Create a 224x224 test image
        img = Image.new('RGB', (224, 224), color=(34, 139, 34))  # Forest green background
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes to simulate pest features
        if pest_name == 'Aphids':
            # Small oval shapes
            for i in range(10):
                x = random.randint(20, 180)
                y = random.randint(20, 180)
                draw.ellipse([x, y, x+15, y+10], fill=color)
        
        elif pest_name == 'Beetle':
            # Larger oval shape
            draw.ellipse([80, 90, 140, 130], fill=color)
            draw.ellipse([90, 100, 130, 120], fill=(101, 67, 33))  # Darker brown
        
        elif pest_name == 'Whitefly':
            # Small white dots
            for i in range(15):
                x = random.randint(20, 200)
                y = random.randint(20, 200)
                draw.ellipse([x, y, x+8, y+6], fill=color)
        
        elif pest_name == 'Thrips':
            # Thin elongated shapes
            for i in range(8):
                x = random.randint(20, 180)
                y = random.randint(20, 180)
                draw.ellipse([x, y, x+20, y+5], fill=color)
        
        elif pest_name == 'Mites':
            # Very small red dots
            for i in range(20):
                x = random.randint(20, 200)
                y = random.randint(20, 200)
                draw.ellipse([x, y, x+4, y+4], fill=color)
        
        else:  # Unknown
            # Random shapes
            for i in range(5):
                x = random.randint(20, 180)
                y = random.randint(20, 180)
                draw.rectangle([x, y, x+30, y+20], fill=color)
        
        # Save the image
        img_path = os.path.join(test_dir, f'test_{pest_name.lower()}.jpg')
        img.save(img_path)
    
    print(f"✅ Test images created in {test_dir}/")
    return True

def test_model_loading():
    """Test if the AI model can be loaded"""
    print("Testing model loading...")
    
    try:
        from app import PestClassificationModel
        
        model = PestClassificationModel()
        model.load_or_create_model()
        
        if model.model is not None:
            print("✅ Model loading test passed!")
            print(f"Model input shape: {model.model.input_shape}")
            print(f"Model output classes: {len(model.class_names)}")
            return True
        else:
            print("❌ Model loading test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_image_prediction():
    """Test image prediction functionality"""
    print("Testing image prediction...")
    
    try:
        from app import PestClassificationModel
        
        # Create test image
        test_img = Image.new('RGB', (224, 224), color=(0, 255, 0))
        
        # Convert to base64 for testing
        buffer = io.BytesIO()
        test_img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test prediction
        model = PestClassificationModel()
        model.load_or_create_model()
        
        result = model.predict(img_str)
        
        if 'pest_name' in result:
            print("✅ Image prediction test passed!")
            print(f"Predicted pest: {result['pest_name']}")
            print(f"Confidence: {result['confidence']:.2f}%")
            return True
        else:
            print("❌ Image prediction test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Image prediction error: {e}")
        return False

def test_treatment_database():
    """Test treatment database functionality"""
    print("Testing treatment database...")
    
    try:
        from app import OrganicTreatmentDatabase
        
        db = OrganicTreatmentDatabase()
        
        # Test getting treatment info for known pest
        aphid_treatment = db.get_treatment_info('Aphids')
        
        if aphid_treatment and 'organic_treatments' in aphid_treatment:
            print("✅ Treatment database test passed!")
            print(f"Found {len(aphid_treatment['organic_treatments'])} treatments for Aphids")
            return True
        else:
            print("❌ Treatment database test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Treatment database error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("Testing API endpoints...")
    
    # This test requires the Flask app to be running
    base_url = 'http://localhost:5000'
    
    try:
        # Test home page
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✅ Home page accessible")
        else:
            print(f"❌ Home page returned status {response.status_code}")
            return False
        
        # Test chat API
        chat_data = {'message': 'Hello, how can you help with pest management?'}
        response = requests.post(f'{base_url}/api/chat', 
                               json=chat_data, 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print("✅ Chat API test passed!")
                print(f"Chat response: {result['response'][:100]}...")
            else:
                print("❌ Chat API returned invalid response")
                return False
        else:
            print(f"❌ Chat API returned status {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def generate_sample_data():
    """Generate sample analysis data for testing"""
    print("Generating sample analysis data...")
    
    try:
        conn = sqlite3.connect('data/pest_analysis.db')
        cursor = conn.cursor()
        
        # Sample pest data
        sample_pests = [
            ('Aphids', 87.5, 'Medium', 'test_aphids.jpg'),
            ('Beetle', 92.3, 'High', 'test_beetle.jpg'),
            ('Whitefly', 78.9, 'Medium', 'test_whitefly.jpg'),
            ('Thrips', 85.1, 'Low', 'test_thrips.jpg'),
            ('Mites', 91.7, 'High', 'test_mites.jpg')
        ]
        
        # Insert sample data with realistic timestamps
        for i, (pest, confidence, severity, image) in enumerate(sample_pests):
            timestamp = datetime.now() - timedelta(days=i*2)
            cursor.execute('''
                INSERT INTO analyses (timestamp, pest_name, confidence, severity, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, pest, confidence, severity, image))
        
        conn.commit()
        conn.close()
        
        print("✅ Sample data generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Sample data generation error: {e}")
        return False

def cleanup_test_data():
    """Clean up test data and files"""
    print("Cleaning up test data...")
    
    try:
        # Remove test images
        test_dir = 'test_images'
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print(f"✅ Removed {test_dir}/")
        
        # Clean test database entries (optional)
        response = input("Remove sample database entries? (y/N): ")
        if response.lower() == 'y':
            conn = sqlite3.connect('data/pest_analysis.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM analyses WHERE image_path LIKE "test_%"')
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"✅ Removed {deleted} test database entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running OrganicGuard AI System Tests")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Test Image Creation", create_test_images),
        ("Model Loading", test_model_loading),
        ("Image Prediction", test_image_prediction),
        ("Treatment Database", test_treatment_database),
        ("Sample Data Generation", generate_sample_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # API test (requires running app)
    print("📋 API Endpoints (requires running Flask app):")
    api_result = test_api_endpoints()
    results.append(("API Endpoints", api_result))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OrganicGuard AI is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            run_all_tests()
        elif command == 'create-images':
            create_test_images()
        elif command == 'sample-data':
            generate_sample_data()
        elif command == 'cleanup':
            cleanup_test_data()
        elif command == 'db':
            test_database_connection()
        else:
            print("Available commands:")
            print("  test         - Run all tests")
            print("  create-images - Create test images")
            print("  sample-data  - Generate sample data")
            print("  cleanup      - Clean up test data")
            print("  db          - Test database connection")
    else:
        print("OrganicGuard AI Utility Script")
        print("Usage: python utils.py <command>")
        print("\nAvailable commands:")
        print("  test         - Run all tests")
        print("  create-images - Create test images")
        print("  sample-data  - Generate sample data")
        print("  cleanup      - Clean up test data")
        print("  db          - Test database connection")

if __name__ == "__main__":
    main()
