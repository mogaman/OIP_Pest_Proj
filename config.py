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
    MODEL_PATH = 'models/best_pest_model.h5'
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

# LLM Configuration for LM Studio Integration
LLM_CONFIG = {
    'provider': 'lmstudio',  # 'lmstudio' or 'none'
    'lmstudio': {
        'base_url': 'http://localhost:1234/v1',  # LM Studio default API endpoint
        'model': 'local-model',  # LM Studio uses generic model name
        'temperature': 0.3,      # Lower for more consistent responses
        'max_tokens': 400,
        'timeout': 30,
        'api_key': 'llama-2-7b-chat'   # LM Studio API key (matches the model)
    },
    'system_prompt': """You are OrganicGuard AI, a specialized expert ONLY in organic pest management and sustainable farming. 

IMPORTANT CONSTRAINTS:
- ONLY discuss organic/biological pest control methods
- ONLY help with the 12 pest types: ants, bees, beetle, caterpillar, earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil
- NEVER recommend chemical pesticides or synthetic treatments
- If asked about non-pest topics, politely redirect to pest management
- Keep responses practical, concise, and farmer-focused

AVAILABLE TREATMENTS: Insecticidal Soap, Neem Oil, Beneficial Insects, Diatomaceous Earth, Coffee Grounds, Hand Picking, Row Covers, Sticky Traps

REMEMBER: Bees and earthworms are BENEFICIAL - never recommend controlling them!"""
}

# Context keywords for intent detection
CONTEXT_KEYWORDS = {
    'identification': ['identify', 'what is', 'looks like', 'found', 'see', 'picture'],
    'treatment': ['control', 'kill', 'get rid', 'treat', 'eliminate', 'manage'],
    'prevention': ['prevent', 'avoid', 'stop', 'protect', 'keep away'],
    'organic': ['organic', 'natural', 'safe', 'non-toxic', 'chemical-free'],
    'beneficial': ['beneficial', 'good', 'helpful', 'positive', 'useful'],
    'damage': ['damage', 'harm', 'eating', 'destroying', 'ruining']
}

# Enhanced pest knowledge base for LLM context
PEST_KNOWLEDGE_BASE = {
    'ants': {
        'identification': 'Social insects, usually 6 legs, segmented body, often in trails',
        'damage': 'Can farm aphids, disturb plant roots, some species bite',
        'organic_treatments': ['Coffee Grounds', 'Diatomaceous Earth', 'Cinnamon barriers'],
        'prevention': 'Remove food sources, seal entry points, maintain clean areas',
        'beneficial_note': 'Some ants are beneficial predators of other pests'
    },
    'bees': {
        'identification': 'Fuzzy body, important pollinators, various sizes',
        'damage': 'Generally BENEFICIAL - essential for pollination',
        'organic_treatments': ['NEVER CONTROL BEES - relocate hives professionally if needed'],
        'prevention': 'Provide alternative nesting sites away from crops',
        'beneficial_note': 'CRITICAL POLLINATORS - essential for food production'
    },
    'beetle': {
        'identification': 'Hard wing covers, diverse sizes, often metallic coloration',
        'damage': 'Leaf feeding, root damage, some bore into stems',
        'organic_treatments': ['Hand Picking', 'Neem Oil', 'Beneficial Insects', 'Row Covers'],
        'prevention': 'Crop rotation, beneficial habitat, proper sanitation',
        'beneficial_note': 'Many beetles are beneficial predators'
    },
    'catterpillar': {
        'identification': 'Soft-bodied larvae, multiple legs, various colors',
        'damage': 'Leaf feeding, can defoliate plants rapidly',
        'organic_treatments': ['Hand Picking', 'Bt spray', 'Beneficial Insects', 'Row Covers'],
        'prevention': 'Encourage beneficial wasps and birds',
        'beneficial_note': 'Will become moths/butterflies - some are beneficial pollinators'
    },
    'earthworms': {
        'identification': 'Segmented, elongated, soil-dwelling',
        'damage': 'NO DAMAGE - extremely beneficial for soil health',
        'organic_treatments': ['NEVER CONTROL - encourage earthworm populations'],
        'prevention': 'Protect earthworm habitat, avoid soil compaction',
        'beneficial_note': 'Essential for soil aeration and nutrient cycling'
    },
    'earwig': {
        'identification': 'Pincer-like appendages, nocturnal, brown/black',
        'damage': 'Minor leaf feeding, can be both pest and predator',
        'organic_treatments': ['Newspaper traps', 'Diatomaceous Earth', 'Hand removal'],
        'prevention': 'Remove debris, reduce moisture areas',
        'beneficial_note': 'Often beneficial - eats aphids and other soft-bodied pests'
    },
    'grasshopper': {
        'identification': 'Large hind legs for jumping, long antennae',
        'damage': 'Heavy leaf feeding, can devastate crops in swarms',
        'organic_treatments': ['Row Covers', 'Beneficial birds', 'Neem Oil'],
        'prevention': 'Encourage natural predators, remove weedy areas',
        'beneficial_note': 'Important food source for birds and other wildlife'
    },
    'moth': {
        'identification': 'Flying insects, often nocturnal, feathery antennae',
        'damage': 'Adults rarely damage plants, larvae (caterpillars) may feed',
        'organic_treatments': ['Pheromone traps', 'Light traps', 'Beneficial Insects'],
        'prevention': 'Remove crop residue, encourage beneficial predators',
        'beneficial_note': 'Many moths are important nighttime pollinators'
    },
    'slug': {
        'identification': 'Soft-bodied, no shell, leave slime trails',
        'damage': 'Irregular holes in leaves, especially seedlings',
        'organic_treatments': ['Beer traps', 'Copper barriers', 'Hand Picking', 'Diatomaceous Earth'],
        'prevention': 'Reduce moisture, remove hiding places',
        'beneficial_note': 'Some slugs help decompose organic matter'
    },
    'snail': {
        'identification': 'Similar to slugs but with shells',
        'damage': 'Similar to slugs - leaf feeding, especially young plants',
        'organic_treatments': ['Hand Picking', 'Copper barriers', 'Beer traps', 'Crushed eggshells'],
        'prevention': 'Same as slugs - reduce moisture and hiding spots',
        'beneficial_note': 'Some snails are beneficial decomposers'
    },
    'wasp': {
        'identification': 'Narrow waist, less fuzzy than bees, can be colorful',
        'damage': 'Most wasps are beneficial predators',
        'organic_treatments': ['Identify species first - many are beneficial'],
        'prevention': 'Only control if truly problematic, provide alternative habitats',
        'beneficial_note': 'Most wasps are excellent predators of pest insects'
    },
    'weevil': {
        'identification': 'Snout-like projection, hard body, often small',
        'damage': 'Feeding on seeds, fruits, leaves, can damage stored crops',
        'organic_treatments': ['Crop rotation', 'Beneficial Insects', 'Neem Oil', 'Diatomaceous Earth'],
        'prevention': 'Proper storage, field sanitation, beneficial habitat',
        'beneficial_note': 'Some weevils are used as biological control agents'
    }
}
