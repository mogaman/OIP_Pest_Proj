"""
Simple model training script following main.py approach
This script trains a CNN model for pest identification using the same approach as main.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup and configure GPU if available"""
    try:
        # List available GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"🚀 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set the first GPU as the default
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Verify GPU is being used
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0]])
                logger.info("✅ GPU setup successful - TensorFlow will use GPU for training")
                
            return True
            
        else:
            logger.warning("⚠️ No GPU found - training will use CPU (this will be slower)")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU setup failed: {e}")
        logger.info("🔄 Falling back to CPU training")
        return False

class SimplePestClassifier:
    """Simple pest classifier following main.py approach"""
    
    def __init__(self, data_dir='../dataset', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        # Using your existing pest classes from config
        from config import PEST_CLASSES
        self.class_names = PEST_CLASSES
        self.model = None
        
        # Setup GPU if available
        self.gpu_available = setup_gpu()
        if self.gpu_available:
            logger.info("🎯 Training will be accelerated with GPU")
        else:
            logger.info("🐌 Training will use CPU (consider using GPU for faster training)")
        
        # Create models directory
        os.makedirs('../models', exist_ok=True)
    
    def prepare_data(self, batch_size=32):
        """Simple data preparation following main.py approach"""
        
        logger.info("📊 Preparing data...")
        
        # Training data with augmentation (following main.py)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        self.train_ds = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names,
            shuffle=True
        )
        
        self.val_ds = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names,
            shuffle=False
        )
        
        logger.info(f"Training samples: {self.train_ds.samples}")
        logger.info(f"Validation samples: {self.val_ds.samples}")
        
        return self.train_ds, self.val_ds
    
    def build_model(self):
        """Build CNN model following main.py architecture"""
        
        logger.info("🏗️ Building CNN model...")
        
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # First Conv Block (following main.py pattern)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers (following main.py)
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile (simplified like main.py)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"✅ Model built with {self.model.count_params():,} parameters")
        
        # Show architecture
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=20):
        """Train model following main.py approach with GPU optimizations"""
        
        logger.info(f"🚀 Training model for {epochs} epochs...")
        
        # GPU-specific optimizations
        if self.gpu_available:
            logger.info("⚡ Applying GPU optimizations...")
            
            # Enable mixed precision for faster training on modern GPUs
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("✅ Mixed precision enabled (faster training on compatible GPUs)")
            except Exception as e:
                logger.warning(f"⚠️ Mixed precision not available: {e}")
            
            # Set optimal batch size for GPU
            if hasattr(self, 'train_ds'):
                # Prefetch for better GPU utilization
                self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)
                self.val_ds = self.val_ds.prefetch(tf.data.AUTOTUNE)
                logger.info("📊 Dataset optimized for GPU with prefetching")
        
        # Simple callbacks (following main.py pattern)
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '../models/pest_classifier.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Add GPU memory monitoring callback if GPU is available
        if self.gpu_available:
            class GPUMemoryCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    try:
                        # Get GPU memory info
                        gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                        used_mb = gpu_info['current'] // (1024 * 1024)
                        peak_mb = gpu_info['peak'] // (1024 * 1024)
                        logger.info(f"🖥️ GPU Memory - Used: {used_mb}MB, Peak: {peak_mb}MB")
                    except:
                        pass  # Skip if memory info not available
            
            callbacks.append(GPUMemoryCallback())
        
        # Train with device placement
        device_name = '/GPU:0' if self.gpu_available else '/CPU:0'
        logger.info(f"🎯 Training on device: {device_name}")
        
        with tf.device(device_name):
            # Train
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save final model
        self.model.save('../models/pest_classifier.h5')
        logger.info("✅ Model saved as '../models/pest_classifier.h5'")
        
        # Print training summary
        if self.gpu_available:
            logger.info("🚀 Training completed with GPU acceleration!")
        else:
            logger.info("🏁 Training completed on CPU")
        
        return history

def main():
    """Main training function following main.py pattern"""
    
    logger.info("🌱 Starting Organic Pest Classification Model Training")
    logger.info("=" * 60)
    
    # Display system information
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    
    # Check for GPU availability upfront
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"🚀 GPU acceleration available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            logger.info(f"   GPU {i}: {gpu.name}")
    else:
        logger.info("💻 No GPU detected - training will use CPU")
        logger.info("   Consider using a GPU for faster training (NVIDIA GTX/RTX series)")
    
    logger.info("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('../dataset'):
        logger.error("❌ Dataset directory not found!")
        logger.info("Please ensure your dataset is in the '../dataset' directory")
        logger.info("Expected structure:")
        logger.info("../dataset/")
        logger.info("  ├── ants/")
        logger.info("  ├── bees/")
        logger.info("  ├── beetle/")
        logger.info("  ├── catterpillar/")
        logger.info("  ├── earthworms/")
        logger.info("  ├── earwig/")
        logger.info("  ├── grasshopper/")
        logger.info("  ├── moth/")
        logger.info("  ├── slug/")
        logger.info("  ├── snail/")
        logger.info("  ├── wasp/")
        logger.info("  └── weevil/")
        return
    
    # Initialize classifier (this will setup GPU)
    classifier = SimplePestClassifier(data_dir='../dataset')
    
    # Prepare data
    train_ds, val_ds = classifier.prepare_data()
    
    # Build model
    model = classifier.build_model()
    
    # Train model
    history = classifier.train_model(epochs=20)
    
    logger.info("✅ Training completed successfully!")
    logger.info("📁 Model saved to: ../models/pest_classifier.h5")
    
    # Final GPU summary
    if classifier.gpu_available:
        logger.info("🚀 Training used GPU acceleration for optimal performance!")
    else:
        logger.info("💻 Training completed on CPU")
        logger.info("   Tip: Use a dedicated GPU for much faster training in the future")

if __name__ == "__main__":
    main()
