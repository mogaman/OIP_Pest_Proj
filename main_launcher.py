"""
OrganicGuard AI - Main Launcher
Provides options to train models and launch different interfaces
"""

import os
import sys
from datetime import datetime

def main_menu():
    """Display main menu and handle user choices"""
    
    print("=" * 60)
    print("🌱 OrganicGuard AI - Pest Management System")
    print("=" * 60)
    
    # Check model status
    model_exists = os.path.exists('models/pest_classifier.h5')
    if model_exists:
        model_size = os.path.getsize('models/pest_classifier.h5') / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime('models/pest_classifier.h5'))
        print(f"✅ Model Status: Found ({model_size:.1f}MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("❌ Model Status: Not found - Training required")
    
    print("\\n📋 Available Options:")
    print("\\n🎯 Training Options:")
    print("1. Train Enhanced CNN (Recommended - Advanced features)")
    print("2. Train Standard CNN (Original training script)")
    print("3. Check Training Progress")
    
    print("\\n🌐 Interface Options:")
    print("4. Launch Flask Web App (Traditional web interface)")
    print("5. Launch Gradio Interface (Modern UI with real-time predictions)")
    
    print("\\n🔧 Utility Options:")
    print("6. View Model Architecture")
    print("7. Clean Up Checkpoints")
    print("8. Run Tests")
    print("9. Exit")
    
    print("\\n" + "=" * 60)
    
    while True:
        try:
            choice = input("🔍 Enter your choice (1-9): ").strip()
            
            if choice == '1':
                train_enhanced_cnn()
            elif choice == '2':
                train_standard_cnn()
            elif choice == '3':
                check_training_progress()
            elif choice == '4':
                launch_flask_app()
            elif choice == '5':
                launch_gradio_interface()
            elif choice == '6':
                view_model_architecture()
            elif choice == '7':
                cleanup_checkpoints()
            elif choice == '8':
                run_tests()
            elif choice == '9':
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Please enter 1-9.")
                continue
                
            # Ask if user wants to continue
            print("\\n" + "-" * 40)
            cont = input("🔄 Return to main menu? (y/n): ").strip().lower()
            if cont != 'y':
                print("👋 Goodbye!")
                break
            else:
                print("\\n")
                main_menu()
                break
                
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")

def train_enhanced_cnn():
    """Launch enhanced CNN training"""
    print("\\n🚀 Launching Enhanced CNN Training...")
    print("📝 Features: Residual connections, advanced augmentation, checkpointing")
    
    try:
        from enhanced_trainer import train_enhanced_model
        train_enhanced_model()
    except ImportError:
        print("❌ Enhanced trainer not found. Please check enhanced_trainer.py exists.")
    except Exception as e:
        print(f"❌ Training error: {e}")

def train_standard_cnn():
    """Launch standard CNN training"""
    print("\\n🚀 Launching Standard CNN Training...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "train_model.py"], check=True)
    except Exception as e:
        print(f"❌ Training error: {e}")

def check_training_progress():
    """Check training progress"""
    print("\\n📊 Checking Training Progress...")
    
    try:
        from enhanced_trainer import check_training_progress
        check_training_progress()
    except ImportError:
        print("❌ Enhanced trainer not found.")
    except Exception as e:
        print(f"❌ Error: {e}")

def launch_flask_app():
    """Launch Flask web application"""
    print("\\n🌐 Launching Flask Web Application...")
    
    if not os.path.exists('models/pest_classifier.h5'):
        print("❌ Model not found. Please train a model first.")
        return
    
    try:
        import subprocess
        subprocess.run([sys.executable, "app.py"], check=True)
    except Exception as e:
        print(f"❌ Error launching Flask app: {e}")

def launch_gradio_interface():
    """Launch Gradio interface"""
    print("\\n🎨 Launching Gradio Interface...")
    
    if not os.path.exists('models/pest_classifier.h5'):
        print("❌ Model not found. Please train a model first.")
        return
    
    try:
        from gradio_interface import launch_gradio_interface
        launch_gradio_interface()
    except ImportError:
        print("❌ Gradio not installed. Please install requirements:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error launching Gradio interface: {e}")

def view_model_architecture():
    """Show model architecture"""
    print("\\n🧠 Model Architecture Viewer...")
    
    print("Which architecture would you like to view?")
    print("1. Enhanced CNN (with residual connections)")
    print("2. Standard CNN")
    
    choice = input("Enter choice (1-2): ").strip()
    
    try:
        if choice == '1':
            from enhanced_trainer import visualize_enhanced_model_architecture
            visualize_enhanced_model_architecture()
        elif choice == '2':
            from train_model import PestClassifierTrainer
            trainer = PestClassifierTrainer()
            model = trainer.create_model()
            model.summary()
        else:
            print("❌ Invalid choice")
    except Exception as e:
        print(f"❌ Error: {e}")

def cleanup_checkpoints():
    """Clean up training checkpoints"""
    print("\\n🧹 Checkpoint Cleanup...")
    
    try:
        from enhanced_trainer import cleanup_checkpoints
        cleanup_checkpoints()
    except ImportError:
        print("❌ Enhanced trainer not found.")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_tests():
    """Run test suite"""
    print("\\n🧪 Running Tests...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "test_suite.py"], check=True)
    except Exception as e:
        print(f"❌ Error running tests: {e}")

if __name__ == "__main__":
    print("🌱 Starting OrganicGuard AI System...")
    
    # Check if we're in the right directory
    if not os.path.exists('config.py'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    main_menu()
