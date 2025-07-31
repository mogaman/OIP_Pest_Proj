"""
Configuration settings for OrganicGuard AI Pest Management System
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'organic_pest_management_secret_key_2024'
    DEBUG = True
    
    # Database settings
    DATABASE_PATH = 'data/pest_analysis.db'
    
    # File upload settings
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # Model settings
    MODEL_PATH = 'models/pest_classifier.h5'
    CONFIDENCE_THRESHOLD = 0.6
    IMAGE_SIZE = (224, 224)
    
    # API settings
    API_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/app.log'
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/uploads', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production_secret_key_here'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DATABASE_PATH = 'data/test_pest_analysis.db'
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Pest classification settings
PEST_CLASSES = [
    'ants',
    'bees',
    'beetle', 
    'catterpillar',
    'earthworms',
    'earwig',
    'grasshopper',
    'moth',
    'slug',
    'snail',
    'wasp',
    'weevil'
]

# Treatment effectiveness data
TREATMENT_EFFECTIVENESS = {
    'Insecticidal Soap': {
        'ants': 70,
        'beetle': 75,
        'catterpillar': 65,
        'earwig': 70,
        'slug': 60,
        'snail': 60
    },
    'Neem Oil': {
        'ants': 75,
        'beetle': 85,
        'catterpillar': 80,
        'earwig': 75,
        'moth': 70,
        'weevil': 80
    },
    'Beneficial Insects': {
        'catterpillar': 90,
        'beetle': 85,
        'moth': 80,
        'weevil': 85
    },
    'Diatomaceous Earth': {
        'ants': 80,
        'beetle': 85,
        'catterpillar': 70,
        'earwig': 85,
        'slug': 90,
        'snail': 90,
        'weevil': 75
    },
    'Coffee Grounds': {
        'ants': 75,
        'slug': 80,
        'snail': 80
    },
    'Hand Picking': {
        'catterpillar': 95,
        'beetle': 90,
        'slug': 95,
        'snail': 95
    },
    'Row Covers': {
        'beetle': 90,
        'catterpillar': 85,
        'grasshopper': 95,
        'moth': 80
    }
}

# Cost estimates per acre (in USD)
TREATMENT_COSTS = {
    'Insecticidal Soap': {'min': 10, 'max': 20},
    'Neem Oil': {'min': 20, 'max': 30},
    'Beneficial Insects': {'min': 30, 'max': 80},
    'Diatomaceous Earth': {'min': 15, 'max': 25},
    'Horticultural Oil': {'min': 18, 'max': 28},
    'Hand Picking': {'min': 5, 'max': 15},
    'Sticky Traps': {'min': 10, 'max': 20},
    'Row Covers': {'min': 200, 'max': 400}
}

# Application timeline estimates
TREATMENT_TIMELINES = {
    'Insecticidal Soap': '2-3 days',
    'Neem Oil': '3-5 days',
    'Beneficial Insects': '1-2 weeks',
    'Diatomaceous Earth': '1-2 weeks',
    'Horticultural Oil': '3-7 days',
    'Hand Picking': 'Immediate',
    'Sticky Traps': 'Continuous',
    'Row Covers': 'Season-long'
}

# Severity classifications
SEVERITY_LEVELS = {
    'Low': {
        'color': 'success',
        'description': 'Minor pest presence, monitoring recommended',
        'action': 'Continue monitoring, implement prevention strategies'
    },
    'Medium': {
        'color': 'warning', 
        'description': 'Moderate pest levels, treatment may be needed',
        'action': 'Begin treatment if population increases'
    },
    'High': {
        'color': 'danger',
        'description': 'Significant pest infestation, immediate action required',
        'action': 'Implement treatment immediately to prevent crop damage'
    }
}

# Model training parameters
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': len(PEST_CLASSES),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5
}

# Image preprocessing settings
IMAGE_PREPROCESSING = {
    'target_size': (224, 224),
    'color_mode': 'rgb',
    'rescale': 1.0/255.0,
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}

# Chat response templates
CHAT_TEMPLATES = {
    'greeting': "Hello! I'm OrganicGuard AI, your agricultural expert. How can I help you with organic pest management today?",
    'identification_help': "I can help identify pests from images. Please upload a clear photo of the pest or affected plant area.",
    'treatment_help': "I can recommend organic treatments for various pests. What specific pest are you dealing with?",
    'cost_help': "Organic treatment costs vary by method and application area. What treatment are you considering?",
    'prevention_help': "Prevention is key in organic farming. Would you like tips for specific pests or general IPM strategies?",
    'error': "I apologize, but I'm having trouble understanding your request. Could you please rephrase your question?"
}
