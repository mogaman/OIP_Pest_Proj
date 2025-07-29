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

class SimplePestClassifier:
    """Simplified pest classifier for demo purposes"""
    
    def __init__(self):
        self.class_names = [
            'Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
            'Mites', 'Sawfly', 'Stem Borer', 'Thrips', 'Whitefly'
        ]
        self.confidence_threshold = 0.6
        
    def predict(self, image):
        """Simple prediction based on image analysis"""
        try:
            if isinstance(image, str):
                # If image is base64 string
                image_data = base64.b64decode(image.split(',')[1] if ',' in image else image)
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simple color-based classification for demo
            img_array = np.array(image.resize((224, 224)))
            
            # Calculate average color values
            avg_red = np.mean(img_array[:, :, 0])
            avg_green = np.mean(img_array[:, :, 1])
            avg_blue = np.mean(img_array[:, :, 2])
            
            # Simple heuristic classification based on dominant colors
            if avg_green > avg_red and avg_green > avg_blue:
                # Green dominant - likely plant with aphids
                pest_index = 0  # Aphids
                confidence = 0.85 + random.uniform(-0.15, 0.10)
            elif avg_red > avg_green and avg_red > avg_blue:
                # Red dominant - might be mites
                pest_index = 5  # Mites
                confidence = 0.78 + random.uniform(-0.10, 0.15)
            elif avg_blue > avg_red and avg_blue > avg_green:
                # Blue dominant - unusual, default to thrips
                pest_index = 8  # Thrips
                confidence = 0.72 + random.uniform(-0.12, 0.18)
            else:
                # Mixed colors - beetle or other
                pest_index = random.choice([2, 4, 6])  # Beetle, Grasshopper, or Sawfly
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
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'pest_name': 'Error in Classification',
                'confidence': 0,
                'prediction_success': False,
                'error': str(e)
            }

class OrganicTreatmentDatabase:
    """Database for organic pest treatment recommendations"""
    
    def __init__(self):
        self.treatments = {
            'Aphids': {
                'severity': 'Medium',
                'crops_affected': 'Vegetables, fruits, herbs, ornamentals',
                'organic_treatments': [
                    {
                        'method': 'Insecticidal Soap Spray',
                        'description': 'Mix 2 tablespoons of mild liquid soap per gallon of water. Spray directly on aphids.',
                        'effectiveness': '85%',
                        'timeline': '2-3 days',
                        'cost': '$10-15/acre',
                        'application': 'Spray early morning or evening every 3-4 days until controlled'
                    },
                    {
                        'method': 'Neem Oil Treatment',
                        'description': 'Apply neem oil solution (2-4 tablespoons per gallon) to affected areas.',
                        'effectiveness': '80%',
                        'timeline': '3-5 days',
                        'cost': '$20-25/acre',
                        'application': 'Apply every 7-14 days as preventive measure'
                    },
                    {
                        'method': 'Beneficial Insects Release',
                        'description': 'Release ladybugs, lacewings, or parasitic wasps for biological control.',
                        'effectiveness': '90%',
                        'timeline': '1-2 weeks',
                        'cost': '$30-50/acre',
                        'application': 'Release when temperatures are 65-85°F'
                    }
                ],
                'prevention': [
                    'Regular monitoring and early detection',
                    'Encourage beneficial insects with diverse plantings',
                    'Avoid over-fertilizing with nitrogen',
                    'Use reflective mulches to deter aphids',
                    'Remove weeds that harbor aphids'
                ]
            },
            'Beetle': {
                'severity': 'High',
                'crops_affected': 'Potatoes, tomatoes, beans, cucumbers',
                'organic_treatments': [
                    {
                        'method': 'Hand Picking',
                        'description': 'Remove beetles manually during early morning when they are sluggish.',
                        'effectiveness': '70%',
                        'timeline': 'Immediate',
                        'cost': '$5-10/acre (labor)',
                        'application': 'Daily inspection and removal during peak activity'
                    },
                    {
                        'method': 'Beneficial Nematodes',
                        'description': 'Apply beneficial nematodes to soil to target beetle larvae.',
                        'effectiveness': '85%',
                        'timeline': '2-3 weeks',
                        'cost': '$40-60/acre',
                        'application': 'Apply to moist soil when temperature is 60-85°F'
                    },
                    {
                        'method': 'Diatomaceous Earth',
                        'description': 'Dust food-grade diatomaceous earth around plants and on foliage.',
                        'effectiveness': '75%',
                        'timeline': '1-2 weeks',
                        'cost': '$15-20/acre',
                        'application': 'Reapply after rain or heavy dew'
                    }
                ],
                'prevention': [
                    'Crop rotation every 2-3 years',
                    'Plant trap crops like radishes',
                    'Use row covers during vulnerable growth stages',
                    'Deep cultivation in fall to expose overwintering beetles',
                    'Encourage ground beetles and spiders'
                ]
            },
            'Whitefly': {
                'severity': 'High',
                'crops_affected': 'Tomatoes, peppers, cabbage, squash',
                'organic_treatments': [
                    {
                        'method': 'Yellow Sticky Traps',
                        'description': 'Place yellow sticky traps around plants to monitor and catch adult whiteflies.',
                        'effectiveness': '60%',
                        'timeline': 'Continuous',
                        'cost': '$10-15/acre',
                        'application': 'Replace traps weekly or when full'
                    },
                    {
                        'method': 'Reflective Mulch',
                        'description': 'Use silver reflective mulch to disorient and repel whiteflies.',
                        'effectiveness': '70%',
                        'timeline': 'Season-long',
                        'cost': '$50-75/acre',
                        'application': 'Install before planting or transplanting'
                    },
                    {
                        'method': 'Parasitic Wasp Release',
                        'description': 'Release Encarsia formosa or Eretmocerus species for biological control.',
                        'effectiveness': '85%',
                        'timeline': '3-4 weeks',
                        'cost': '$60-80/acre',
                        'application': 'Release when whitefly population is detected'
                    }
                ],
                'prevention': [
                    'Remove plant debris and weeds',
                    'Use companion planting with basil or marigolds',
                    'Maintain proper plant spacing for air circulation',
                    'Regular inspection of undersides of leaves',
                    'Avoid excessive nitrogen fertilization'
                ]
            },
            'Thrips': {
                'severity': 'Medium',
                'crops_affected': 'Onions, tomatoes, peppers, flowers',
                'organic_treatments': [
                    {
                        'method': 'Blue Sticky Traps',
                        'description': 'Use blue sticky traps to monitor and capture adult thrips.',
                        'effectiveness': '65%',
                        'timeline': 'Continuous',
                        'cost': '$12-18/acre',
                        'application': 'Place traps at plant height, replace weekly'
                    },
                    {
                        'method': 'Predatory Mites',
                        'description': 'Release predatory mites like Amblyseius species for biological control.',
                        'effectiveness': '80%',
                        'timeline': '2-3 weeks',
                        'cost': '$45-65/acre',
                        'application': 'Release during mild weather conditions'
                    },
                    {
                        'method': 'Spinosad Spray',
                        'description': 'Apply organic spinosad-based insecticide for severe infestations.',
                        'effectiveness': '85%',
                        'timeline': '3-5 days',
                        'cost': '$25-35/acre',
                        'application': 'Spray in evening to protect beneficial insects'
                    }
                ],
                'prevention': [
                    'Remove weeds and plant debris',
                    'Use reflective mulches',
                    'Maintain adequate soil moisture',
                    'Plant resistant varieties when available',
                    'Encourage natural predators like minute pirate bugs'
                ]
            },
            'Mites': {
                'severity': 'High',
                'crops_affected': 'Strawberries, beans, corn, ornamentals',
                'organic_treatments': [
                    {
                        'method': 'Horticultural Oil Spray',
                        'description': 'Apply horticultural oil to smother mites and eggs.',
                        'effectiveness': '75%',
                        'timeline': '3-7 days',
                        'cost': '$18-25/acre',
                        'application': 'Spray during cooler parts of the day'
                    },
                    {
                        'method': 'Predatory Mite Release',
                        'description': 'Release Phytoseiulus persimilis or other predatory mites.',
                        'effectiveness': '90%',
                        'timeline': '2-4 weeks',
                        'cost': '$50-70/acre',
                        'application': 'Release when pest mites are first detected'
                    },
                    {
                        'method': 'Strong Water Spray',
                        'description': 'Use strong water spray to dislodge mites from plants.',
                        'effectiveness': '60%',
                        'timeline': 'Immediate',
                        'cost': '$5/acre (water)',
                        'application': 'Daily spraying for 1-2 weeks'
                    }
                ],
                'prevention': [
                    'Maintain adequate humidity levels',
                    'Avoid water stress in plants',
                    'Regular inspection with magnifying glass',
                    'Remove heavily infested leaves',
                    'Encourage beneficial insects with diverse plantings'
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
pest_model = SimplePestClassifier()
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
