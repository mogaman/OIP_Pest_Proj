"""
Gradio Interface Module for OrganicGuard AI Pest Classification
Provides an alternative modern interface using Gradio
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from config import Config, PEST_CLASSES

# Pest treatment database - using your project's pest classes
TREATMENTS = {
    'ants': 'Use coffee grounds or cinnamon around plants. Apply diatomaceous earth. Use ant baits with borax (away from children/pets).',
    'bees': '🌟 PROTECT! Essential pollinators - do not treat. Encourage with bee-friendly plants.',
    'beetle': 'Apply neem oil spray. Use row covers during beetle season. Hand-pick adult beetles.',
    'catterpillar': 'Use Bt spray (Bacillus thuringiensis). Hand-pick when visible. Apply neem oil.',
    'earthworms': '🌟 BENEFICIAL! Improve soil health - protect them. Add organic matter to encourage them.',
    'earwig': 'Use newspaper traps. Apply diatomaceous earth around plants. Remove garden debris.',
    'grasshopper': 'Use row covers. Encourage birds and spiders. Apply neem oil for young hoppers.',
    'moth': 'Use pheromone traps. Install light traps away from plants. Apply Bt spray for larvae.',
    'slug': 'Use iron phosphate bait. Apply diatomaceous earth. Create beer traps.',
    'snail': 'Hand-pick in evening. Use copper strips around plants. Apply iron phosphate bait.',
    'wasp': '🌟 BENEFICIAL predators! Usually protect unless near high-traffic areas.',
    'weevil': 'Use beneficial nematodes. Apply diatomaceous earth. Use sticky traps.'
}

class GradioInterface:
    """Gradio interface for pest classification"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_PATH
        self.class_names = PEST_CLASSES
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                print(f"❌ Model not found at {self.model_path}")
                print("💡 Please train a model first using enhanced_trainer.py")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict(self, image):
        """Predict pest type from image"""
        if self.model is None:
            return "❌ Model not loaded. Please train a model first.", ""
        
        if image is None:
            return "Please upload an image", ""
        
        try:
            # Preprocess image - match training size
            img = image.resize(Config.IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            predicted_pest = self.class_names[predicted_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            
            # Format results
            result = f"**🎯 Primary Detection:** {predicted_pest} ({confidence:.1%} confidence)\\n\\n"
            result += "**📊 Top 3 Predictions:**\\n"
            for i, idx in enumerate(top_3_idx, 1):
                pest = self.class_names[idx]
                conf = predictions[0][idx]
                result += f"{i}. {pest}: {conf:.1%}\\n"
            
            # Get treatment advice
            treatment = f"**🌿 Organic Treatment:** {TREATMENTS.get(predicted_pest, 'No treatment info available')}"
            
            return result, treatment
            
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}", ""
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(
            title="🌱 OrganicGuard AI - Pest Classifier",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .pest-header {
                text-align: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            """
        ) as app:
            
            # Header
            gr.HTML("""
            <div class="pest-header">
                <h1>🌱 OrganicGuard AI Pest Classifier</h1>
                <h3>🧠 Enhanced CNN with Advanced Features</h3>
                <p>Upload a pest photo to get identification and organic treatment advice</p>
            </div>
            """)
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Pest Photo", 
                        type="pil",
                        height=300
                    )
                    
                    identify_btn = gr.Button(
                        "🔍 Identify Pest", 
                        variant="primary", 
                        size="lg",
                        scale=1
                    )
                
                with gr.Column(scale=1):
                    result_output = gr.Markdown(
                        label="🔍 Detection Results",
                        value="Upload an image and click 'Identify Pest' to see results"
                    )
                    
                    treatment_output = gr.Markdown(
                        label="🌿 Treatment Advice",
                        value="Treatment recommendations will appear here"
                    )
            
            # Event handlers
            identify_btn.click(
                fn=self.predict,
                inputs=[image_input],
                outputs=[result_output, treatment_output]
            )
            
            # Auto-predict on image upload
            image_input.change(
                fn=self.predict,
                inputs=[image_input],
                outputs=[result_output, treatment_output]
            )
            
            # Information section
            with gr.Accordion("📋 Supported Pest Types & Model Info", open=False):
                gr.Markdown(f"""
                ### 🐛 Supported Pest Types ({len(PEST_CLASSES)} classes)
                {' • '.join(PEST_CLASSES)}
                
                ### 🧠 Model Information
                - **Architecture**: Enhanced CNN with residual connections
                - **Input Size**: {Config.IMAGE_SIZE[0]}×{Config.IMAGE_SIZE[1]} pixels
                - **Features**: Data augmentation, class balancing, label smoothing
                - **Training**: Advanced techniques for better accuracy
                
                ### 🌿 Organic Treatment Focus
                All treatment recommendations use organic, environmentally-friendly methods that are safe for beneficial insects and pollinators.
                """)
            
            # Example images section
            with gr.Accordion("📸 Example Images", open=False):
                gr.Markdown("""
                ### Tips for Best Results:
                - Use clear, well-lit photos
                - Focus on the pest (close-up preferred)
                - Avoid blurry or dark images
                - Multiple angles can help with identification
                """)
        
        return app

def launch_gradio_interface(share=True, port=7860):
    """Launch the Gradio interface"""
    
    print("🚀 Launching OrganicGuard AI Gradio Interface...")
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Model not found at {Config.MODEL_PATH}")
        print("💡 Please train a model first:")
        print("   python enhanced_trainer.py")
        print("   or")
        print("   python train_model.py")
        return
    
    # Create and launch interface
    interface = GradioInterface()
    app = interface.create_interface()
    
    print("🌐 Interface starting...")
    print(f"   📱 Local URL: http://localhost:{port}")
    if share:
        print("   🌍 Public URL will be generated...")
    print("   ⏹️ Press Ctrl+C to stop")
    
    app.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0" if share else "127.0.0.1"
    )

if __name__ == "__main__":
    launch_gradio_interface()
