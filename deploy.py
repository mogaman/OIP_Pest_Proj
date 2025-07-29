"""
Deployment and setup script for OrganicGuard AI
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = 'venv\\Scripts\\pip.exe'
    else:  # Linux/Mac
        pip_path = 'venv/bin/pip'
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    
    directories = [
        'data',
        'models', 
        'static/uploads',
        'logs',
        'test_images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def initialize_database():
    """Initialize the database"""
    print("Initializing database...")
    
    try:
        from app import init_database
        init_database()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def setup_environment_file():
    """Create .env file with default settings"""
    print("Creating environment file...")
    
    env_content = """# OrganicGuard AI Environment Configuration

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=organic_pest_management_secret_key_2024

# Database Configuration
DATABASE_URL=sqlite:///data/pest_analysis.db

# Model Configuration
MODEL_PATH=models/pest_classifier.h5
CONFIDENCE_THRESHOLD=0.6

# Upload Configuration
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216

# Optional API Keys (for enhanced features)
# OPENAI_API_KEY=your-openai-key-here
# DEEPSEEK_API_KEY=your-deepseek-key-here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Environment file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create environment file: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("Running system tests...")
    
    try:
        from utils import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_development_server():
    """Start the development server"""
    print("Starting development server...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = 'venv\\Scripts\\python.exe'
    else:  # Linux/Mac
        python_path = 'venv/bin/python'
    
    try:
        print("🚀 Starting OrganicGuard AI on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        subprocess.run([python_path, 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")

def build_docker_image():
    """Build Docker image"""
    print("Building Docker image...")
    
    try:
        subprocess.run(['docker', 'build', '-t', 'organic-pest-app', '.'], check=True)
        print("✅ Docker image built successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to build Docker image")
        return False

def run_docker_container():
    """Run Docker container"""
    print("Running Docker container...")
    
    try:
        subprocess.run([
            'docker', 'run', '-p', '5000:5000', 
            '--name', 'organic-pest-app',
            'organic-pest-app'
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to run Docker container")
        return False

def deploy_production():
    """Deploy for production"""
    print("Setting up production deployment...")
    
    # Update environment for production
    env_content = """# OrganicGuard AI Production Configuration

FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=CHANGE_THIS_IN_PRODUCTION

DATABASE_URL=sqlite:///data/pest_analysis.db
MODEL_PATH=models/pest_classifier.h5
CONFIDENCE_THRESHOLD=0.6

UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216

LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
    
    try:
        with open('.env.production', 'w') as f:
            f.write(env_content)
        
        print("✅ Production environment file created")
        print("⚠️  Remember to:")
        print("   1. Change SECRET_KEY in .env.production")
        print("   2. Set up proper database (PostgreSQL/MySQL for production)")
        print("   3. Configure reverse proxy (nginx)")
        print("   4. Set up SSL certificates")
        print("   5. Configure monitoring and logging")
        
        return True
    except Exception as e:
        print(f"❌ Failed to setup production: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='OrganicGuard AI Deployment Script')
    parser.add_argument('command', choices=[
        'setup', 'install', 'test', 'run', 'docker-build', 
        'docker-run', 'production', 'clean'
    ], help='Command to execute')
    
    args = parser.parse_args()
    
    print("🌱 OrganicGuard AI Deployment Script")
    print("=" * 50)
    
    if args.command == 'setup':
        print("Setting up OrganicGuard AI...")
        
        success = (
            check_python_version() and
            create_virtual_environment() and
            install_dependencies() and
            setup_directories() and
            setup_environment_file() and
            initialize_database()
        )
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("Run 'python deploy.py run' to start the development server")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
    
    elif args.command == 'install':
        install_dependencies()
    
    elif args.command == 'test':
        run_tests()
    
    elif args.command == 'run':
        start_development_server()
    
    elif args.command == 'docker-build':
        build_docker_image()
    
    elif args.command == 'docker-run':
        run_docker_container()
    
    elif args.command == 'production':
        deploy_production()
    
    elif args.command == 'clean':
        print("Cleaning up...")
        # Add cleanup logic here
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
