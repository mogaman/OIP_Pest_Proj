"""
OrganicGuard AI - Simplified Demo Version
A Flask-based web application for organic pest management consultation
"""

import os
import io
import json
import base64
import sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import random
import logging

# Import TensorFlow for model loading
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available - using demo classifier")

# Import config for pest classes
from config import PEST_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'organic_pest_management_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

class PestClassifier:
    """Real pest classifier using trained model"""
    
    def __init__(self):
        self.class_names = PEST_CLASSES
        self.confidence_threshold = 0.6
        self.model = None
        self.model_path = 'models/pest_classifier.h5'
        
        # Try to load trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained model if available"""
        try:
            if TENSORFLOW_AVAILABLE and os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Loaded trained model from {self.model_path}")
            else:
                if not TENSORFLOW_AVAILABLE:
                    logger.warning("⚠️ TensorFlow not available - using demo mode")
                else:
                    logger.warning(f"⚠️ No trained model found at {self.model_path} - using demo mode")
                self.model = None
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict(self, image):
        """Predict pest type from image"""
        try:
            if isinstance(image, str):
                # If image is base64 string
                image_data = base64.b64decode(image.split(',')[1] if ',' in image else image)
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use real model if available
            if self.model is not None:
                return self._predict_with_model(image)
            else:
                return self._predict_demo_mode(image)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'pest_name': 'Error in Classification',
                'confidence': 0,
                'prediction_success': False,
                'error': str(e)
            }
    
    def _predict_with_model(self, image):
        """Real prediction using trained model"""
        try:
            # Preprocess image for model
            img_array = np.array(image.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            pest_name = self.class_names[predicted_class]
            
            # Create all predictions dictionary
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = float(predictions[0][i] * 100)
            
            return {
                'pest_name': pest_name,
                'confidence': confidence * 100,
                'prediction_success': confidence >= self.confidence_threshold,
                'all_predictions': all_predictions,
                'model_used': 'trained_model'
            }
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self._predict_demo_mode(image)
    
    def _predict_demo_mode(self, image):
        """Fallback demo prediction when model not available"""
        logger.info("🔄 Using demo mode - train a model for real predictions")
        
        # Simple color-based classification for demo
        img_array = np.array(image.resize((224, 224)))
        
        # Calculate average color values
        avg_red = np.mean(img_array[:, :, 0])
        avg_green = np.mean(img_array[:, :, 1])
        avg_blue = np.mean(img_array[:, :, 2])
        
        # Simple heuristic classification based on dominant colors
        if avg_green > avg_red and avg_green > avg_blue:
            # Green dominant - likely plant with ants
            pest_index = 0  # ants
            confidence = 0.85 + random.uniform(-0.15, 0.10)
        elif avg_red > avg_green and avg_red > avg_blue:
            # Red dominant - might be beetle
            pest_index = 2  # beetle
            confidence = 0.78 + random.uniform(-0.10, 0.15)
        elif avg_blue > avg_red and avg_blue > avg_green:
            # Blue dominant - unusual, default to moth
            pest_index = 8  # moth
            confidence = 0.72 + random.uniform(-0.12, 0.18)
        else:
            # Mixed colors - random selection
            pest_index = random.choice([3, 6, 9])  # catterpillar, earwig, slug
            confidence = 0.75 + random.uniform(-0.20, 0.20)
        
        # Ensure confidence is within bounds
        confidence = max(0.5, min(0.95, confidence))
        
        pest_name = self.class_names[pest_index]
        
        return {
            'pest_name': pest_name,
            'confidence': confidence * 100,
            'prediction_success': confidence >= self.confidence_threshold,
            'all_predictions': {
                name: random.uniform(5, 95) if name == pest_name else random.uniform(1, 30)
                for name in self.class_names
            },
            'model_used': 'demo_mode'
        }

class OrganicTreatmentDatabase:
    """Database for organic pest treatment recommendations"""
    
    def __init__(self):
        self.treatments = {
            'ants': {
                'severity': 'Low',
                'crops_affected': 'Various crops, gardens',
                'organic_treatments': [
                    {
                        'method': 'Coffee Grounds',
                        'description': 'Spread used coffee grounds around plants. Ants dislike the acidity.',
                        'effectiveness': '75%',
                        'timeline': '1-2 days',
                        'cost': '$5-10/acre',
                        'application': 'Apply dry grounds weekly around affected areas'
                    },
                    {
                        'method': 'Cinnamon Barrier',
                        'description': 'Sprinkle ground cinnamon around plants to deter ants.',
                        'effectiveness': '70%',
                        'timeline': '2-3 days',
                        'cost': '$10-15/acre',
                        'application': 'Reapply after rain or watering'
                    },
                    {
                        'method': 'Diatomaceous Earth',
                        'description': 'Apply food-grade diatomaceous earth around ant trails.',
                        'effectiveness': '80%',
                        'timeline': '3-5 days',
                        'cost': '$15-20/acre',
                        'application': 'Apply in dry weather, reapply as needed'
                    }
                ],
                'prevention': [
                    'Remove food sources and standing water',
                    'Seal entry points around garden beds',
                    'Plant mint or tansy as natural deterrents',
                    'Keep garden clean of fallen fruit',
                    'Use ant-resistant plant varieties'
                ]
            },
            'bees': {
                'severity': 'Beneficial',
                'crops_affected': 'All flowering crops (BENEFICIAL)',
                'organic_treatments': [
                    {
                        'method': 'Protection & Encouragement',
                        'description': '🌟 PROTECT BEES! They are essential pollinators - do not treat as pests.',
                        'effectiveness': '100%',
                        'timeline': 'Ongoing',
                        'cost': '$0/acre',
                        'application': 'Provide bee-friendly flowers and avoid pesticides during bloom'
                    },
                    {
                        'method': 'Bee-Friendly Plants',
                        'description': 'Plant lavender, sunflowers, and native wildflowers to support bees.',
                        'effectiveness': '95%',
                        'timeline': 'Season-long',
                        'cost': '$20-30/acre',
                        'application': 'Plant diverse flowering species for continuous bloom'
                    }
                ],
                'prevention': [
                    'Never apply pesticides during flowering',
                    'Provide clean water sources',
                    'Plant diverse native flowers',
                    'Avoid disturbing natural nesting sites',
                    'Support local beekeepers'
                ]
            },
            'beetle': {
                'severity': 'High',
                'crops_affected': 'Potatoes, beans, cucumbers, squash',
                'organic_treatments': [
                    {
                        'method': 'Hand Picking',
                        'description': 'Remove beetles manually during early morning when sluggish.',
                        'effectiveness': '85%',
                        'timeline': 'Immediate',
                        'cost': '$10-15/acre (labor)',
                        'application': 'Daily inspection and removal during peak activity'
                    },
                    {
                        'method': 'Neem Oil Spray',
                        'description': 'Apply neem oil solution to affected plants and soil.',
                        'effectiveness': '80%',
                        'timeline': '3-7 days',
                        'cost': '$20-25/acre',
                        'application': 'Spray every 7-10 days during beetle season'
                    },
                    {
                        'method': 'Row Covers',
                        'description': 'Use floating row covers during vulnerable plant stages.',
                        'effectiveness': '90%',
                        'timeline': 'Season-long',
                        'cost': '$200-300/acre',
                        'application': 'Install before beetle emergence, remove during pollination'
                    }
                ],
                'prevention': [
                    'Crop rotation every 2-3 years',
                    'Deep cultivation in fall',
                    'Plant trap crops like radishes',
                    'Encourage ground beetles and spiders',
                    'Remove plant debris promptly'
                ]
            },
            'catterpillar': {
                'severity': 'Medium',
                'crops_affected': 'Brassicas, tomatoes, corn, various vegetables',
                'organic_treatments': [
                    {
                        'method': 'Bt Spray (Bacillus thuringiensis)',
                        'description': 'Apply Bt spray targeting caterpillar larvae in evening.',
                        'effectiveness': '90%',
                        'timeline': '3-5 days',
                        'cost': '$25-30/acre',
                        'application': 'Spray when caterpillars are small, reapply every 7-10 days'
                    },
                    {
                        'method': 'Hand Picking',
                        'description': 'Remove caterpillars manually when visible.',
                        'effectiveness': '95%',
                        'timeline': 'Immediate',
                        'cost': '$15-20/acre (labor)',
                        'application': 'Daily inspection, especially undersides of leaves'
                    },
                    {
                        'method': 'Neem Oil',
                        'description': 'Apply neem oil to disrupt caterpillar feeding and growth.',
                        'effectiveness': '75%',
                        'timeline': '5-7 days',
                        'cost': '$20-25/acre',
                        'application': 'Apply every 10-14 days as preventive measure'
                    }
                ],
                'prevention': [
                    'Use pheromone traps for monitoring',
                    'Encourage birds and beneficial wasps',
                    'Plant companion plants like dill and fennel',
                    'Rotate crops annually',
                    'Remove egg masses when found'
                ]
            },
            'earthworms': {
                'severity': 'Beneficial',
                'crops_affected': 'All crops (HIGHLY BENEFICIAL)',
                'organic_treatments': [
                    {
                        'method': 'Protection & Encouragement',
                        'description': '🌟 PROTECT EARTHWORMS! They improve soil health and structure.',
                        'effectiveness': '100%',
                        'timeline': 'Ongoing',
                        'cost': '$0/acre',
                        'application': 'Add organic matter and avoid soil compaction'
                    },
                    {
                        'method': 'Soil Enhancement',
                        'description': 'Add compost and organic matter to encourage earthworm activity.',
                        'effectiveness': '95%',
                        'timeline': 'Season-long',
                        'cost': '$50-75/acre',
                        'application': 'Apply compost 2-3 times per growing season'
                    }
                ],
                'prevention': [
                    'Avoid chemical pesticides and fertilizers',
                    'Maintain soil moisture',
                    'Add organic compost regularly',
                    'Minimize soil tillage',
                    'Keep soil covered with mulch'
                ]
            },
            'earwig': {
                'severity': 'Medium',
                'crops_affected': 'Seedlings, soft fruits, flowers',
                'organic_treatments': [
                    {
                        'method': 'Newspaper Traps',
                        'description': 'Roll up damp newspaper for earwigs to hide in, then dispose.',
                        'effectiveness': '80%',
                        'timeline': '1-2 days',
                        'cost': '$5-10/acre',
                        'application': 'Place traps in evening, collect and dispose in morning'
                    },
                    {
                        'method': 'Diatomaceous Earth',
                        'description': 'Apply food-grade diatomaceous earth around plants.',
                        'effectiveness': '85%',
                        'timeline': '3-5 days',
                        'cost': '$15-20/acre',
                        'application': 'Apply in dry conditions, reapply after rain'
                    },
                    {
                        'method': 'Garden Cleanup',
                        'description': 'Remove garden debris where earwigs hide during day.',
                        'effectiveness': '70%',
                        'timeline': 'Immediate',
                        'cost': '$10/acre (labor)',
                        'application': 'Regular removal of mulch, boards, and plant debris'
                    }
                ],
                'prevention': [
                    'Remove hiding places like boards and debris',
                    'Use copper strips around sensitive plants',
                    'Plant trap crops away from main garden',
                    'Encourage ground beetles and birds',
                    'Keep garden areas well-lit'
                ]
            },
            'grasshopper': {
                'severity': 'High',
                'crops_affected': 'Grains, grasses, vegetables, fruits',
                'organic_treatments': [
                    {
                        'method': 'Row Covers',
                        'description': 'Use floating row covers to protect crops from grasshoppers.',
                        'effectiveness': '95%',
                        'timeline': 'Season-long',
                        'cost': '$200-350/acre',
                        'application': 'Install before grasshopper migration'
                    },
                    {
                        'method': 'Encourage Predators',
                        'description': 'Attract birds and spiders with diverse plantings and habitat.',
                        'effectiveness': '75%',
                        'timeline': '2-4 weeks',
                        'cost': '$30-50/acre',
                        'application': 'Plant native flowers and provide bird nesting sites'
                    },
                    {
                        'method': 'Neem Oil for Young Hoppers',
                        'description': 'Apply neem oil spray when grasshoppers are young and vulnerable.',
                        'effectiveness': '70%',
                        'timeline': '1-2 weeks',
                        'cost': '$20-25/acre',
                        'application': 'Apply early morning when grasshoppers are less active'
                    }
                ],
                'prevention': [
                    'Maintain diverse habitat for natural predators',
                    'Till soil in fall to destroy egg masses',
                    'Use trap crops like wheat or barley',
                    'Keep grass areas mowed short',
                    'Remove weeds that serve as food sources'
                ]
            },
            'moth': {
                'severity': 'Medium',
                'crops_affected': 'Various crops (larvae cause damage)',
                'organic_treatments': [
                    {
                        'method': 'Pheromone Traps',
                        'description': 'Use species-specific pheromone traps to catch adult moths.',
                        'effectiveness': '80%',
                        'timeline': 'Continuous',
                        'cost': '$25-35/acre',
                        'application': 'Install before moth flight period, replace lures monthly'
                    },
                    {
                        'method': 'Light Traps',
                        'description': 'Install light traps away from crops to attract and capture moths.',
                        'effectiveness': '70%',
                        'timeline': 'Nightly',
                        'cost': '$50-75/acre',
                        'application': 'Operate during peak moth activity periods'
                    },
                    {
                        'method': 'Bt Spray for Larvae',
                        'description': 'Apply Bt spray when moth larvae are active.',
                        'effectiveness': '85%',
                        'timeline': '3-7 days',
                        'cost': '$25-30/acre',
                        'application': 'Target young larvae, apply in evening'
                    }
                ],
                'prevention': [
                    'Monitor with pheromone traps',
                    'Remove plant debris and weeds',
                    'Encourage beneficial insects',
                    'Use companion planting',
                    'Practice good crop rotation'
                ]
            },
            'slug': {
                'severity': 'Medium',
                'crops_affected': 'Leafy greens, seedlings, soft fruits',
                'organic_treatments': [
                    {
                        'method': 'Iron Phosphate Bait',
                        'description': 'Use organic iron phosphate slug bait around affected plants.',
                        'effectiveness': '90%',
                        'timeline': '3-7 days',
                        'cost': '$20-30/acre',
                        'application': 'Apply in evening when slugs are active'
                    },
                    {
                        'method': 'Diatomaceous Earth',
                        'description': 'Apply food-grade diatomaceous earth as a barrier.',
                        'effectiveness': '75%',
                        'timeline': '1-3 days',
                        'cost': '$15-20/acre',
                        'application': 'Apply in dry conditions around plants'
                    },
                    {
                        'method': 'Beer Traps',
                        'description': 'Create beer traps to attract and drown slugs.',
                        'effectiveness': '70%',
                        'timeline': '1-2 days',
                        'cost': '$10-15/acre',
                        'application': 'Bury containers level with soil, replace beer regularly'
                    }
                ],
                'prevention': [
                    'Remove hiding places like boards and debris',
                    'Use copper strips as barriers',
                    'Encourage ground beetles and birds',
                    'Reduce moisture around plants',
                    'Hand-pick in evening when active'
                ]
            },
            'snail': {
                'severity': 'Medium',
                'crops_affected': 'Leafy greens, seedlings, fruits',
                'organic_treatments': [
                    {
                        'method': 'Hand Picking',
                        'description': 'Remove snails manually in evening when they are active.',
                        'effectiveness': '95%',
                        'timeline': 'Immediate',
                        'cost': '$10-15/acre (labor)',
                        'application': 'Daily collection during peak activity'
                    },
                    {
                        'method': 'Copper Strips',
                        'description': 'Install copper strips around plants as barriers.',
                        'effectiveness': '85%',
                        'timeline': 'Season-long',
                        'cost': '$100-150/acre',
                        'application': 'Install around bed perimeters and individual plants'
                    },
                    {
                        'method': 'Iron Phosphate Bait',
                        'description': 'Use organic iron phosphate bait safe for pets and wildlife.',
                        'effectiveness': '90%',
                        'timeline': '5-7 days',
                        'cost': '$20-30/acre',
                        'application': 'Apply in evening, reapply after rain'
                    }
                ],
                'prevention': [
                    'Remove hiding places and debris',
                    'Create dry barriers around plants',
                    'Encourage natural predators',
                    'Water plants in morning to reduce evening moisture',
                    'Use raised beds for better drainage'
                ]
            },
            'wasp': {
                'severity': 'Beneficial',
                'crops_affected': 'Various crops (BENEFICIAL PREDATOR)',
                'organic_treatments': [
                    {
                        'method': 'Protection & Encouragement',
                        'description': '🌟 PROTECT BENEFICIAL WASPS! They control many pest insects.',
                        'effectiveness': '100%',
                        'timeline': 'Ongoing',
                        'cost': '$0/acre',
                        'application': 'Avoid pesticides and provide flowering plants'
                    },
                    {
                        'method': 'Habitat Enhancement',
                        'description': 'Plant flowers that provide nectar for beneficial wasps.',
                        'effectiveness': '90%',
                        'timeline': 'Season-long',
                        'cost': '$25-40/acre',
                        'application': 'Plant diverse flowering species'
                    }
                ],
                'prevention': [
                    'Only control if near high-traffic areas',
                    'Provide alternative nesting sites',
                    'Plant flowers for beneficial species',
                    'Avoid broad-spectrum pesticides',
                    'Educate about beneficial vs. pest species'
                ]
            },
            'weevil': {
                'severity': 'High',
                'crops_affected': 'Grains, nuts, stored products, root crops',
                'organic_treatments': [
                    {
                        'method': 'Beneficial Nematodes',
                        'description': 'Apply beneficial nematodes to soil to target weevil larvae.',
                        'effectiveness': '85%',
                        'timeline': '2-4 weeks',
                        'cost': '$40-60/acre',
                        'application': 'Apply to moist soil when temperature is 60-85°F'
                    },
                    {
                        'method': 'Diatomaceous Earth',
                        'description': 'Apply food-grade diatomaceous earth around affected plants.',
                        'effectiveness': '75%',
                        'timeline': '1-2 weeks',
                        'cost': '$15-25/acre',
                        'application': 'Apply in dry conditions, reapply after rain'
                    },
                    {
                        'method': 'Sticky Traps',
                        'description': 'Use yellow sticky traps to monitor and capture adult weevils.',
                        'effectiveness': '60%',
                        'timeline': 'Continuous',
                        'cost': '$15-20/acre',
                        'application': 'Replace traps weekly during peak activity'
                    }
                ],
                'prevention': [
                    'Crop rotation with non-host plants',
                    'Remove plant debris after harvest',
                    'Deep cultivation to expose larvae',
                    'Use resistant plant varieties',
                    'Monitor with pheromone traps'
                ]
            }
        }
    
    def get_treatment_info(self, pest_name):
        """Get treatment information for a specific pest"""
        # Normalize pest name
        for key in self.treatments.keys():
            if key.lower() in pest_name.lower() or pest_name.lower() in key.lower():
                return self.treatments[key]
        
        # Return generic treatment if specific pest not found
        return {
            'severity': 'Medium',
            'crops_affected': 'Various crops may be affected',
            'organic_treatments': [
                {
                    'method': 'General Organic Spray',
                    'description': 'Use a general organic insecticidal soap or neem oil solution.',
                    'effectiveness': '70%',
                    'timeline': '3-7 days',
                    'cost': '$15-25/acre',
                    'application': 'Apply during cooler parts of the day'
                },
                {
                    'method': 'Beneficial Insect Habitat',
                    'description': 'Create habitat for beneficial insects with diverse plantings.',
                    'effectiveness': '60%',
                    'timeline': 'Season-long',
                    'cost': '$20-30/acre',
                    'application': 'Plant flowering plants near crops'
                }
            ],
            'prevention': [
                'Regular monitoring and inspection',
                'Maintain healthy soil with organic matter',
                'Proper crop rotation',
                'Remove infected plant debris',
                'Encourage biodiversity in the farm ecosystem'
            ]
        }

# Initialize global objects
pest_model = PestClassifier()
treatment_db = OrganicTreatmentDatabase()

def init_database():
    """Initialize SQLite database for storing analysis history"""
    conn = sqlite3.connect('data/pest_analysis.db')
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()

def save_analysis(pest_name, confidence, severity, image_path, treatment_applied='', user_notes=''):
    """Save analysis results to database"""
    conn = sqlite3.connect('data/pest_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO analyses (pest_name, confidence, severity, image_path, treatment_applied, user_notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (pest_name, confidence, severity, image_path, treatment_applied, user_notes))
    
    conn.commit()
    conn.close()

def get_recent_analyses(limit=10):
    """Get recent analysis history"""
    conn = sqlite3.connect('data/pest_analysis.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM analyses 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    analyses = cursor.fetchall()
    conn.close()
    
    return analyses

@app.route('/')
def home():
    """Main dashboard page"""
    recent_analyses = get_recent_analyses(5)
    return render_template('index.html', recent_analyses=recent_analyses)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Pest analysis page"""
    if request.method == 'POST':
        try:
            # Handle file upload
            if 'pest_image' not in request.files:
                flash('No image file provided', 'error')
                return redirect(request.url)
            
            file = request.files['pest_image']
            
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Save uploaded file
                filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze the image
                image = Image.open(filepath)
                prediction_result = pest_model.predict(image)
                
                if prediction_result['prediction_success']:
                    # Get treatment recommendations
                    treatment_info = treatment_db.get_treatment_info(prediction_result['pest_name'])
                    
                    # Save analysis to database
                    save_analysis(
                        prediction_result['pest_name'],
                        prediction_result['confidence'],
                        treatment_info['severity'],
                        filepath
                    )
                    
                    return render_template('results.html', 
                                         prediction=prediction_result,
                                         treatment=treatment_info,
                                         image_path=filename)
                else:
                    flash('Could not identify the pest. Please try with a clearer image.', 'warning')
                    return render_template('analyze.html')
            
            else:
                flash('Invalid file type. Please upload JPEG, PNG, or WebP images.', 'error')
                return redirect(request.url)
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            flash(f'An error occurred during analysis: {str(e)}', 'error')
            return render_template('analyze.html')
    
    return render_template('analyze.html')

@app.route('/chat')
def chat():
    """AI chat consultation page"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower()
        
        # Simple rule-based responses for offline capability
        response = generate_chat_response(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Sorry, I encountered an error. Please try again.'}), 500

def generate_chat_response(message):
    """Generate chat responses based on user input"""
    message = message.lower()
    
    if any(word in message for word in ['aphid', 'aphids']):
        return """Aphids are small, soft-bodied insects that feed on plant sap. For organic control:
        
1. **Insecticidal Soap**: Mix 2 tbsp mild liquid soap per gallon of water
2. **Neem Oil**: Apply 2-4 tbsp per gallon every 7-14 days
3. **Beneficial Insects**: Release ladybugs or lacewings
4. **Prevention**: Avoid over-fertilizing with nitrogen, use reflective mulches

Would you like specific application instructions for any of these methods?"""
    
    elif any(word in message for word in ['beetle', 'beetles']):
        return """Beetles can be challenging pests. Organic management strategies:
        
1. **Hand Picking**: Most effective in early morning when beetles are sluggish
2. **Beneficial Nematodes**: Apply to soil for larvae control
3. **Diatomaceous Earth**: Dust around plants (food-grade only)
4. **Row Covers**: Physical barrier during vulnerable stages
5. **Crop Rotation**: Break pest cycles every 2-3 years

Which crop are you protecting from beetles?"""
    
    elif any(word in message for word in ['organic', 'treatment', 'control']):
        return """Key organic pest management principles:
        
1. **Prevention First**: Healthy soil, proper spacing, resistant varieties
2. **Monitoring**: Weekly inspections, sticky traps, threshold-based treatment
3. **Biological Control**: Beneficial insects, parasites, predators
4. **Organic Sprays**: Neem oil, insecticidal soap, horticultural oils
5. **Physical Methods**: Row covers, barriers, hand removal
6. **Cultural Practices**: Crop rotation, companion planting, sanitation

What specific pest or crop situation are you dealing with?"""
    
    elif any(word in message for word in ['cost', 'price', 'expensive']):
        return """Organic pest management costs typically:
        
- **Insecticidal Soap**: $10-20/acre
- **Neem Oil**: $20-30/acre  
- **Beneficial Insects**: $30-80/acre
- **Sticky Traps**: $10-20/acre
- **Row Covers**: $200-400/acre (reusable)

**Cost-Saving Tips**:
- Prevention is always cheaper than treatment
- Make your own insecticidal soap
- Encourage native beneficial insects
- Use integrated approaches for better ROI

Would you like specific budget recommendations for your farm size?"""
    
    elif any(word in message for word in ['when', 'timing', 'time']):
        return """Optimal timing for organic treatments:
        
**Daily Timing**:
- Early morning (6-8 AM) or evening (6-8 PM)
- Avoid midday heat and beneficial insect activity

**Seasonal Timing**:
- Prevention: Before pest emergence
- Treatment: At first sign of damage
- Beneficial releases: When conditions are optimal

**Weather Considerations**:
- No rain expected for 4-6 hours after application
- Light winds (under 10 mph)
- Temperature between 65-85°F

What specific treatment are you planning to apply?"""
    
    elif any(word in message for word in ['help', 'identify', 'what']):
        return """I can help you with:
        
📸 **Upload a photo** for AI-powered pest identification
🌱 **Organic treatment** recommendations
📊 **Cost estimates** and application timing
🔄 **Prevention strategies** and IPM planning
📚 **Crop-specific** pest management advice
📈 **Treatment effectiveness** comparisons

To get started, you can:
- Upload a pest image for identification
- Ask about specific pests or treatments
- Request organic certification-approved methods

What would you like help with today?"""
    
    else:
        return """I'm here to help with organic pest management! I can assist with:
        
- Pest identification from photos
- Organic treatment recommendations  
- Prevention strategies
- Cost estimates and timing
- IPM (Integrated Pest Management) planning

Try asking about specific pests like "How do I control aphids organically?" or upload a photo for AI identification."""

@app.route('/history')
def history():
    """Analysis history page"""
    analyses = get_recent_analyses(50)
    return render_template('history.html', analyses=analyses)

@app.route('/api/analyze_base64', methods=['POST'])
def api_analyze_base64():
    """API endpoint for base64 image analysis"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Analyze the image
        prediction_result = pest_model.predict(image_data)
        
        if prediction_result['prediction_success']:
            treatment_info = treatment_db.get_treatment_info(prediction_result['pest_name'])
            
            return jsonify({
                'success': True,
                'pest_name': prediction_result['pest_name'],
                'confidence': prediction_result['confidence'],
                'treatment': treatment_info
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Could not identify the pest. Please try with a clearer image.'
            })
            
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Initialize the application
    init_database()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
