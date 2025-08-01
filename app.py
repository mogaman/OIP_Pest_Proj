"""
Simple Pest AI - Image Analysis & Chat
A streamlined Flask app for pest identification and consultation
"""

import os
import io
import base64
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TensorFlow for real AI model
try:
    import tensorflow as tf
    from config import PEST_CLASSES
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow available - will use trained AI model if available")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("⚠️ TensorFlow not available - using demo mode")

app = Flask(__name__)
app.secret_key = 'simple_pest_ai_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

class SmartPestClassifier:
    """Intelligent pest classifier that uses trained AI model or falls back to demo"""
    
    def __init__(self):
        self.model = None
        self.demo_mode = True
        self.confidence_threshold = 60  # Changed to percentage (0-100) for consistency
        
        logger.info("🚀 Initializing SmartPestClassifier...")
        
        # Try to load trained model first - check multiple possible model files
        model_paths = [
            'models/pest_classifier.h5',                    # From training scripts
            'models/custom_pest_model.h5',                  # From wm_cnn.py
            'models/efficientnet_pest_final.h5',            # From EfficientNet training
            'models/convnext_pest_classifier.h5',           # From ConvNeXt training
            'models/pest_classifier_mobilenetv2.h5'         # Legacy path
        ]
        
        model_loaded = False
        if TF_AVAILABLE:
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        logger.info(f"🤖 Loading trained AI model from {model_path}...")
                        self.model = tf.keras.models.load_model(model_path)
                        self.class_names = PEST_CLASSES  # Use config classes
                        self.demo_mode = False
                        model_loaded = True
                        logger.info("✅ Trained AI model loaded successfully!")
                        logger.info(f"🎯 Model can identify: {', '.join(self.class_names)}")
                        logger.info(f"📁 Using model: {model_path}")
                        break
                    except Exception as e:
                        logger.error(f"❌ Failed to load model {model_path}: {e}")
                        continue
        
        if not model_loaded:
            if not TF_AVAILABLE:
                logger.info("📝 TensorFlow not available - using demo mode")
            else:
                logger.info("📝 No trained model found - using demo mode")
                logger.info("💡 Train a model using the training_scripts to get real predictions")
            self._setup_demo_mode()
    
    def _setup_demo_mode(self):
        """Setup demo mode with fallback pest classes"""
        self.demo_mode = True
        self.class_names = [
            'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
            'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
        ]
        logger.info("🎭 Demo mode active - predictions will be simulated")
    
    def _check_image_quality(self, image):
        """Check if image quality is sufficient for analysis"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Check image size
            if image.size[0] < 100 or image.size[1] < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            # Check brightness
            if len(img_array.shape) == 3:
                brightness = np.mean(img_array)
                if brightness < 30:
                    return False, "Image too dark"
                elif brightness > 240:
                    return False, "Image overexposed"
            
            # Check for blur (simple variance of Laplacian)
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            laplacian_var = np.var(np.gradient(gray))
            if laplacian_var < 100:
                return False, "Image too blurry"
            
            # Check contrast
            if len(img_array.shape) == 3:
                contrast = np.std(img_array)
                if contrast < 20:
                    return False, "Image has low contrast"
            
            # Check color range
            if len(img_array.shape) == 3:
                color_range = np.max(img_array) - np.min(img_array)
                if color_range < 50:
                    return False, "Limited color range in image"
            
            return True, "Image quality acceptable"
            
        except Exception as e:
            logger.error(f"Error checking image quality: {e}")
            return False, f"Error analyzing image: {str(e)}"
        
    def predict(self, image):
        """Predict pest type using trained model or demo logic"""
        try:
            if isinstance(image, str):
                # If image is base64 string
                image_data = base64.b64decode(image.split(',')[1] if ',' in image else image)
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check image quality first
            quality_ok, quality_msg = self._check_image_quality(image)
            if not quality_ok:
                return {
                    'pest_name': 'Image Quality Issue',
                    'confidence': 0,
                    'prediction_success': False,
                    'error': f"Image not clear enough: {quality_msg}"
                }
            
            if not self.demo_mode and self.model is not None:
                # Use trained AI model
                return self._predict_with_ai(image)
            else:
                # Use demo prediction
                return self._predict_demo(image)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'pest_name': 'Error in Classification',
                'confidence': 0,
                'prediction_success': False,
                'error': str(e)
            }
    
    def _predict_with_ai(self, image):
        """Use trained AI model for prediction"""
        try:
            # Determine the expected input size based on model
            # Most training scripts use 224x224, but wm_cnn.py uses 160x160
            input_shape = self.model.input_shape[1:3] if self.model.input_shape else (224, 224)
            logger.info(f"🔍 Model expects input size: {input_shape}")
            
            # Preprocess image for model
            img_array = np.array(image.resize(input_shape))
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            logger.info(f"📊 Input shape: {img_array.shape}")
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            logger.info(f"🔍 Raw predictions shape: {predictions.shape}")
            logger.info(f"🔍 Raw predictions: {predictions[0][:5]}...")  # Show first 5 values
            
            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Ensure class index is valid
            if predicted_class_idx >= len(self.class_names):
                logger.error(f"❌ Invalid class index {predicted_class_idx}, max is {len(self.class_names)-1}")
                return self._predict_demo(image)
            
            pest_name = self.class_names[predicted_class_idx]
            
            # Create detailed predictions for all classes
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(predictions[0]):
                    all_predictions[class_name] = float(predictions[0][i]) * 100
                else:
                    all_predictions[class_name] = 0.0
            
            logger.info(f"🤖 AI Prediction: {pest_name} ({confidence*100:.1f}%)")
            logger.info(f"🎯 Class index: {predicted_class_idx}/{len(self.class_names)}")
            
            return {
                'pest_name': pest_name,
                'confidence': confidence * 100,
                'prediction_success': confidence >= (self.confidence_threshold / 100),  # Convert threshold to 0-1 range
                'all_predictions': all_predictions,
                'model_type': 'AI'
            }
            
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            logger.error(f"Model shape: {self.model.input_shape if self.model else 'No model'}")
            # Fall back to demo mode
            return self._predict_demo(image)
    
    def _predict_demo(self, image):
        """Improved demo prediction with better heuristics"""
        try:
            # Resize image for analysis
            img_array = np.array(image.resize((224, 224)))
            
            # Calculate color statistics
            avg_red = np.mean(img_array[:, :, 0])
            avg_green = np.mean(img_array[:, :, 1])
            avg_blue = np.mean(img_array[:, :, 2])
            
            # Calculate texture features
            gray = np.mean(img_array, axis=2)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # More sophisticated classification based on multiple features
            features = {
                'green_dominant': avg_green > max(avg_red, avg_blue) + 10,
                'brown_tones': (avg_red > avg_blue + 20) and (avg_green > avg_blue + 15),
                'dark_overall': brightness < 100,
                'high_contrast': contrast > 50,
                'red_spots': np.max(img_array[:, :, 0]) - avg_red > 50
            }
            
            # Classification logic based on visual features
            if features['green_dominant'] and features['high_contrast']:
                # Likely on plant, could be leaf-eating pests
                candidates = ['catterpillar', 'slug', 'snail', 'beetle']
                base_confidence = 0.75
            elif features['brown_tones'] and not features['dark_overall']:
                # Brown/earthy colors
                candidates = ['beetle', 'weevil', 'earwig', 'ants']
                base_confidence = 0.70
            elif features['dark_overall']:
                # Dark insects
                candidates = ['beetle', 'earwig', 'ants']
                base_confidence = 0.65
            elif features['red_spots']:
                # Colorful insects
                candidates = ['bees', 'wasp', 'beetle']
                base_confidence = 0.68
            else:
                # Mixed/unclear
                candidates = ['moth', 'grasshopper', 'wasp', 'ants']
                base_confidence = 0.60
            
            # Select most likely candidate
            pest_name = random.choice(candidates)
            
            # Add some variation but keep it reasonable
            confidence = base_confidence + random.uniform(-0.1, 0.15)
            confidence = max(0.5, min(0.9, confidence))  # Keep within bounds
            
            # Create more realistic prediction distribution
            all_predictions = {}
            remaining_confidence = 1.0 - confidence
            
            for name in self.class_names:
                if name == pest_name:
                    all_predictions[name] = confidence * 100
                elif name in candidates:
                    # Other candidates get higher scores
                    all_predictions[name] = random.uniform(0.05, 0.25) * remaining_confidence * 100
                else:
                    # Non-candidates get lower scores
                    all_predictions[name] = random.uniform(0.001, 0.05) * remaining_confidence * 100
            
            # Normalize to ensure they sum to 100%
            total = sum(all_predictions.values())
            for name in all_predictions:
                all_predictions[name] = (all_predictions[name] / total) * 100
            
            logger.info(f"🎭 Demo Prediction: {pest_name} ({confidence*100:.1f}%)")
            logger.info(f"🎨 Image features: {features}")
            
            return {
                'pest_name': pest_name,
                'confidence': confidence * 100,
                'prediction_success': confidence >= (self.confidence_threshold / 100),
                'all_predictions': all_predictions,
                'model_type': 'Demo'
            }
            
        except Exception as e:
            logger.error(f"Demo prediction error: {e}")
            return {
                'pest_name': 'Error in Classification',
                'confidence': 0,
                'prediction_success': False,
                'error': str(e)
            }

class SimpleTreatmentDatabase:
    """Simple treatment recommendations"""
    
    def __init__(self):
        self.treatments = {
            'ants': 'Use coffee grounds or cinnamon around plants. Apply diatomaceous earth.',
            'bees': 'PROTECT! Essential pollinators - do not treat. Encourage with flowers.',
            'beetle': 'Hand pick in early morning. Use neem oil spray or row covers.',
            'catterpillar': 'Hand pick or use Bt (Bacillus thuringiensis) spray.',
            'earthworms': 'BENEFICIAL! Improve soil health - protect them.',
            'earwig': 'Use newspaper traps or diatomaceous earth around plants.',
            'grasshopper': 'Use row covers. Encourage birds with bird houses.',
            'moth': 'Use pheromone traps or light traps in evening.',
            'slug': 'Use iron phosphate bait or copper strips around plants.',
            'snail': 'Hand pick or use organic slug bait. Remove hiding places.',
            'wasp': 'BENEFICIAL predators - usually protect. Remove only if necessary.',
            'weevil': 'Use beneficial nematodes or row covers. Clean up debris.'
        }
    
    def get_treatment(self, pest_name):
        """Get simple treatment recommendation"""
        pest_key = pest_name.lower()
        for key in self.treatments.keys():
            if key in pest_key or pest_key in key:
                return self.treatments[key]
        return 'Monitor regularly. Use organic neem oil spray or insecticidal soap as general treatment.'

# Initialize global objects
pest_model = SmartPestClassifier()
treatment_db = SimpleTreatmentDatabase()

@app.route('/')
def home():
    """Main page with image upload and chat"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image analysis"""
    try:
        # Handle file upload
        if 'pest_image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['pest_image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            image = Image.open(filepath)
            prediction_result = pest_model.predict(image)
            
            if prediction_result['prediction_success']:
                # Get treatment recommendation
                treatment = treatment_db.get_treatment(prediction_result['pest_name'])
                
                return jsonify({
                    'success': True,
                    'pest_name': prediction_result['pest_name'],
                    'confidence': prediction_result['confidence'],
                    'treatment': treatment,
                    'model_type': prediction_result.get('model_type', 'Unknown'),
                    'image_path': filename
                })
            else:
                return jsonify({
                    'success': False,
                    'message': prediction_result.get('error', 'Could not identify the pest. Please try with a clearer image.')
                })
        
        else:
            return jsonify({'error': 'Invalid file type. Please upload JPEG, PNG, or WebP images.'}), 400
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """AI chat endpoint"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_message = data.get('message', '').lower()
            
            # Simple rule-based responses
            response = generate_chat_response(user_message)
            
            return jsonify({'response': response})
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({'error': 'Sorry, I encountered an error. Please try again.'}), 500
    
    # For GET requests, return the main page (chat is integrated)
    return render_template('index.html')

def generate_chat_response(message):
    """Generate simple chat responses"""
    message = message.lower()
    
    if any(word in message for word in ['aphid', 'aphids']):
        return "For aphids: Use insecticidal soap (2 tbsp per gallon water) or neem oil. Release ladybugs for natural control."
    
    elif any(word in message for word in ['beetle', 'beetles']):
        return "For beetles: Hand pick in early morning, use beneficial nematodes for larvae, or apply diatomaceous earth around plants."
    
    elif any(word in message for word in ['slug', 'slugs', 'snail', 'snails']):
        return "For slugs/snails: Use iron phosphate bait, copper strips, or hand pick. Remove hiding places like debris."
    
    elif any(word in message for word in ['caterpillar', 'catterpillar', 'worm']):
        return "For caterpillars: Hand pick or use Bt spray (Bacillus thuringiensis). Check undersides of leaves regularly."
    
    elif any(word in message for word in ['ant', 'ants']):
        return "For ants: Use coffee grounds, cinnamon, or diatomaceous earth around plants. Find and eliminate the source."
    
    elif any(word in message for word in ['organic', 'natural', 'safe']):
        return "Safe organic options: Neem oil, insecticidal soap, beneficial insects, companion planting, and crop rotation."
    
    elif any(word in message for word in ['spray', 'treatment']):
        return "General organic spray: Mix 1-2 tbsp mild soap + 1 tbsp neem oil per quart water. Spray in evening to avoid harming bees."
    
    elif any(word in message for word in ['help', 'identify', 'what']):
        return "I can help identify pests from photos and suggest organic treatments. Upload an image above or ask about specific pests!"
    
    else:
        return "I can help with pest identification and organic treatments. Try uploading a photo or ask about specific pests like aphids, beetles, or slugs."

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
            treatment = treatment_db.get_treatment(prediction_result['pest_name'])
            
            return jsonify({
                'success': True,
                'pest_name': prediction_result['pest_name'],
                'confidence': prediction_result['confidence'],
                'treatment': treatment
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
    # Run the simplified application
    app.run(debug=True, host='0.0.0.0', port=5000)
