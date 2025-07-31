"""
Simple launcher for the Organic Pest Management App
Works with custom CNN or heuristic fallback
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import numpy
        import PIL
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   💡 Install with: pip install flask numpy pillow")
        return False

def check_model_status():
    """Check if custom CNN model exists"""
    model_path = "custom_pest_model.h5"
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ Custom CNN model found ({model_size:.1f}MB)")
        return True
    else:
        print("⚠️ No custom CNN model found")
        print("   🔄 App will use enhanced heuristic analysis")
        print("   💡 Train model with: python custom_cnn_trainer.py")
        return False

def launch_app():
    """Launch the Flask application"""
    print("🚀 Starting Organic Pest Management App...")
    
    try:
        # Import and run the Flask app
        from app import app
        
        print("🌐 Flask app starting...")
        print("   📱 Open http://localhost:5000 in your browser")
        print("   🛑 Press Ctrl+C to stop")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("   💡 Check if app.py exists and is properly configured")

def main():
    """Main launcher function"""
    print("🐛 ORGANIC PEST MANAGEMENT AI SYSTEM")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model status
    check_model_status()
    
    # Launch app
    print("\n" + "=" * 50)
    launch_app()

if __name__ == "__main__":
    main()
