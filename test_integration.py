"""
Test script to verify model integration with app.py
Run this after training a model to test the integration
"""

import os
import sys
from PIL import Image
import numpy as np

# Add current directory to path so we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_integration():
    """Test that the app can load and use trained models"""
    
    print("🧪 Testing Model Integration")
    print("=" * 50)
    
    # Import the app classes
    try:
        from app import SmartPestClassifier
        print("✅ Successfully imported SmartPestClassifier")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Initialize the classifier
    try:
        classifier = SmartPestClassifier()
        print(f"✅ Classifier initialized")
        print(f"📋 Available classes: {len(classifier.class_names)}")
        print(f"🎯 Class names: {', '.join(classifier.class_names[:5])}...")
        
        if classifier.demo_mode:
            print("🎭 Currently in DEMO mode - no trained model found")
            print("💡 Train a model first using:")
            print("   python train_model_simple.py")
            print("   OR")
            print("   python train_efficientnet.py")
        else:
            print("🤖 AI MODEL LOADED! Real predictions will be used.")
            
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return False
    
    # Test with a dummy image
    try:
        print("\n🖼️ Testing with dummy image...")
        
        # Create a simple test image (green square)
        test_image = Image.new('RGB', (224, 224), color='green')
        
        # Make prediction
        result = classifier.predict(test_image)
        
        print(f"📊 Prediction Result:")
        print(f"   🐛 Pest: {result['pest_name']}")
        print(f"   📈 Confidence: {result['confidence']:.1f}%")
        print(f"   ✅ Success: {result['prediction_success']}")
        print(f"   🔧 Model Type: {result.get('model_type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def check_model_files():
    """Check what model files exist"""
    print("\n📁 Checking for model files...")
    
    model_files = [
        'models/pest_classifier.h5',
        'models/efficientnet_pest_final.h5',
        'models/efficientnet_pest_phase1.h5'
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ Found: {model_file} ({size_mb:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"❌ Missing: {model_file}")
    
    if found_models:
        print(f"\n🎉 Found {len(found_models)} trained model(s)!")
        print("🚀 Your app will use AI predictions!")
    else:
        print("\n⚠️ No trained models found.")
        print("📝 App will run in demo mode with simulated predictions.")
    
    return len(found_models) > 0

def main():
    """Main test function"""
    
    print("🌱 Organic Pest Management - Model Integration Test")
    print("=" * 60)
    
    # Check for trained models
    has_models = check_model_files()
    
    # Test the integration
    success = test_model_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ INTEGRATION TEST PASSED!")
        if has_models:
            print("🤖 Your app is ready to use AI-powered pest identification!")
        else:
            print("🎭 App is in demo mode - train a model for real AI predictions.")
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("🔧 Check the error messages above for troubleshooting.")
    
    print("\n📝 Next steps:")
    if not has_models:
        print("1. Train a model: python train_efficientnet.py")
    print("2. Start the app: python app.py")
    print("3. Open browser: http://localhost:5000")
    print("4. Upload pest images for identification!")

if __name__ == "__main__":
    main()
