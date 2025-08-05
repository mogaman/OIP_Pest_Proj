"""
Organic Farm Pest Management AI - Simple Version
A simplified Flask application for pest identification and treatment recommendations
"""

import os
import io
import base64
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import io
import base64
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify
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

# Import LLM service (LM Studio support)
try:
    from llm_service import OrganicGuardLLM
    LLM_AVAILABLE = True
    organic_guard_llm = OrganicGuardLLM()  # Initialize the LLM service
    logger.info("✅ LM Studio LLM service loaded")
except ImportError as e:
    LLM_AVAILABLE = False
    organic_guard_llm = None  # Set to None if not available
    logger.warning(f"⚠️ LM Studio LLM service not available: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

class SmartPestClassifier:
    """Intelligent pest classifier that uses trained AI model or falls back to demo"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.confidence_threshold = 0.6
        
        # Try to load trained model first
        model_paths = [
            'models/best_pest_model.h5',

        ]
        
        if TF_AVAILABLE:
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        logger.info(f"🤖 Loading trained AI model from {model_path}...")
                        self.model = tf.keras.models.load_model(model_path)
                        self.class_names = PEST_CLASSES  # Use config classes
                        self.model_loaded = True
                        logger.info("✅ Trained AI model loaded successfully!")
                        logger.info(f"🎯 Model can identify: {', '.join(self.class_names)}")
                        break
                    except Exception as e:
                        logger.error(f"❌ Failed to load model from {model_path}: {e}")
                        continue
        
        if not self.model_loaded:
            if not TF_AVAILABLE:
                logger.error("❌ TensorFlow not available - cannot load AI model")
            else:
                logger.error("❌ No trained model found - pest identification unavailable")
            self.class_names = []
    
    
    def predict(self, image):
        """Predict pest type using trained model"""
        try:
            # Check if model is available
            if not self.model_loaded or self.model is None:
                return {
                    'pest_name': 'Model Not Available',
                    'confidence': 0,
                    'prediction_success': False,
                    'error': 'No trained AI model available. Please ensure a valid model file is present.'
                }
            
            if isinstance(image, str):
                # If image is base64 string
                image_data = base64.b64decode(image.split(',')[1] if ',' in image else image)
                image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                logger.info(f"🔄 Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Use trained AI model
            return self._predict_with_ai(image)
                
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
            # Debug: Log original image mode and size
            logger.info(f"🔍 Original image mode: {image.mode}, size: {image.size}")

            # Ensure image is RGB - be more explicit about conversion
            if image.mode != 'RGB':
                logger.info(f"🔄 Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            # Verify conversion worked
            logger.info(f"🔍 After conversion - image mode: {image.mode}")

            # Resize image to match model expectations (224x224 as required by the trained model)
            resized_image = image.resize((224, 224))
            logger.info(f"🔍 Resized image size: {resized_image.size}")

            # Convert to numpy array
            img_array = np.array(resized_image)
            logger.info(f"🔍 Initial numpy array shape: {img_array.shape}")

            # Double-check we have the right number of channels
            if len(img_array.shape) == 2:
                logger.error("❌ Got 2D array (grayscale) - converting to 3-channel")
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[-1] == 1:
                logger.error("❌ Got 3D array with 1 channel - converting to 3-channel")
                img_array = np.repeat(img_array, 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[-1] == 4:
                logger.info("🔄 Got 4-channel image (RGBA) - converting to RGB")
                img_array = img_array[:, :, :3]  # Remove alpha channel

            # Final shape check
            if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
                logger.error(f"❌ Unexpected array shape after conversion: {img_array.shape}")
                raise ValueError(f"Could not convert image to proper RGB format. Shape: {img_array.shape}")

            logger.info(f"✅ Final image array shape: {img_array.shape}")

            # Convert to float32 and normalize to [0, 1] range
            img_array = img_array.astype(np.float32) / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Debug logging
            logger.info(f"🔍 Image shape before prediction: {img_array.shape}")
            logger.info(f"🔍 Image value range: [{img_array.min():.3f}, {img_array.max():.3f}]")

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)

            # Get results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            pest_name = self.class_names[predicted_class_idx]

            logger.info(f"🤖 AI Prediction: {pest_name} ({confidence*100:.1f}%)")

            return {
                'pest_name': pest_name,
                'confidence': confidence * 100,
                'prediction_success': confidence >= self.confidence_threshold,
                'model_type': 'AI'
            }

        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            return {
                'pest_name': 'Error in Classification',
                'confidence': 0,
                'prediction_success': False,
                'error': f'AI prediction failed: {str(e)}'
            }
    
class SimpleTreatmentDatabase:
    """Simple treatment recommendations"""
    
    def __init__(self):
        self.treatments = {
            'ants': 'Use coffee grounds, cinnamon, or diatomaceous earth around plants. Find and eliminate the source.',
            'bees': 'Bees are beneficial! Protect them. If they must be moved, contact a local beekeeper.',
            'beetle': 'Hand pick in early morning, use beneficial nematodes for larvae, or apply diatomaceous earth.',
            'catterpillar': 'Hand pick or use Bt spray (Bacillus thuringiensis). Check undersides of leaves regularly.',
            'earthworms': 'Earthworms are beneficial for soil! No treatment needed.',
            'earwig': 'Use diatomaceous earth or trap with rolled newspaper. Remove hiding places.',
            'grasshopper': 'Use floating row covers, neem oil, or encourage natural predators like birds.',
            'moth': 'Use pheromone traps, Bt spray for larvae, or encourage natural predators.',
            'slug': 'Use iron phosphate bait, copper strips, or hand pick. Remove hiding places like debris.',
            'snail': 'Use iron phosphate bait, copper strips, or hand pick. Remove hiding places like debris.',
            'wasp': 'Most wasps are beneficial predators. If nest removal needed, do it at night.',
            'weevil': 'Use beneficial nematodes, diatomaceous earth, or crop rotation. Remove plant debris.'
        }
    
    def get_treatment(self, pest_name):
        """Get treatment for a specific pest"""
        pest_key = pest_name.lower()
        return self.treatments.get(pest_key, 'General organic spray: Mix 1-2 tbsp mild soap + 1 tbsp neem oil per quart water. Spray in evening.')

# Initialize global objects
pest_model = SmartPestClassifier()
treatment_db = SimpleTreatmentDatabase()

@app.route('/')
def home():
    """Main page with simple interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Simple image analysis endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Analyze the image
        image = Image.open(file)
        prediction_result = pest_model.predict(image)
        
        if prediction_result['prediction_success']:
            treatment = treatment_db.get_treatment(prediction_result['pest_name'])
            
            return jsonify({
                'success': True,
                'pest_name': prediction_result['pest_name'],
                'confidence': round(prediction_result['confidence'], 1),
                'treatment': treatment
            })
        else:
            error_message = prediction_result.get('error', 'Could not identify the pest. Please try with a clearer image.')
            return jsonify({
                'success': False,
                'message': error_message
            })
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'message': f'Analysis failed: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with LM Studio LLM integration"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please provide a message'}), 400
        
        # Use LLM service if available
        if LLM_AVAILABLE and organic_guard_llm is not None:
            response = organic_guard_llm.generate_response(message)
            model_used = 'lmstudio' if hasattr(organic_guard_llm, 'available') and organic_guard_llm.available else 'specialized_fallback'
        else:
            # Fallback to basic response generation
            response = generate_chat_response(message)
            model_used = 'basic_fallback'
        
        return jsonify({
            'response': response,
            'model_used': model_used,
            'project_focused': True,
            'timestamp': datetime.now().isoformat() if 'datetime' in globals() else None
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'response': "I'm experiencing some technical difficulties. Please try asking about pest identification, organic treatments, or prevention methods.",
            'error': True
        }), 500

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

def generate_chat_response_detailed(message):
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
    """Analysis history page - simplified version"""
    # Since we simplified the app without database, return empty history
    analyses = []
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
            treatment = treatment_db.get_treatment(prediction_result['pest_name'])
            
            return jsonify({
                'success': True,
                'pest_name': prediction_result['pest_name'],
                'confidence': prediction_result['confidence'],
                'treatment': treatment
            })
        else:
            error_message = prediction_result.get('error', 'Could not identify the pest. Please try with a clearer image.')
            return jsonify({
                'success': False,
                'message': error_message
            })
            
    except Exception as e:
        logger.error(f"API analysis error: {e}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Run the application with LM Studio LLM integration
    logger.info("🚀 Starting OrganicGuard AI with LM Studio LLM integration")
    
    # Check model status
    if pest_model.model_loaded:
        logger.info("🤖 AI model loaded and ready for pest identification")
    else:
        logger.warning("⚠️ No AI model available - pest identification will show error messages")
    
    if LLM_AVAILABLE and organic_guard_llm is not None and hasattr(organic_guard_llm, 'available') and organic_guard_llm.available:
        logger.info(f"🤖 LM Studio LLM ready and available")
    else:
        logger.info("📝 Running in fallback mode - install LM Studio for advanced AI chat")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
