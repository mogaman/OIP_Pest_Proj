"""
EfficientNetV2M-based pest classification training script
Uses pre-trained EfficientNetV2M model with transfer learning for superior performance
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from datetime import datetime

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

class EfficientNetPestClassifier:
    """Advanced pest classifier using pre-trained EfficientNetV2M"""
    
    def __init__(self, data_dir='../dataset', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Using your existing pest classes from config
        try:
            from config import PEST_CLASSES
            self.class_names = PEST_CLASSES
        except ImportError:
            # Fallback to default classes if config not available
            self.class_names = [
                'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
                'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
            ]
            logger.warning("⚠️ Using default pest classes - ensure config.py exists")
        
        self.model = None
        self.base_model = None
        
        # Setup GPU if available
        self.gpu_available = setup_gpu()
        if self.gpu_available:
            logger.info("🎯 Training will be accelerated with GPU")
        else:
            logger.info("🐌 Training will use CPU (consider using GPU for faster training)")
        
        # Create models directory in parent folder
        os.makedirs('../models', exist_ok=True)
        
        logger.info(f"📋 Classes to train: {len(self.class_names)} pest types")
        logger.info(f"🎯 Target classes: {', '.join(self.class_names)}")
    
    def prepare_data(self, batch_size=32):
        """Prepare data with advanced augmentation for transfer learning"""
        
        logger.info("📊 Preparing data with advanced augmentation...")
        
        # Advanced training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation data (only rescaling)
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
        
        logger.info(f"📈 Training samples: {self.train_ds.samples}")
        logger.info(f"📊 Validation samples: {self.val_ds.samples}")
        logger.info(f"🎯 Classes found: {len(self.train_ds.class_indices)}")
        
        # Verify class mapping
        logger.info("🏷️ Class mapping:")
        for class_name, index in self.train_ds.class_indices.items():
            logger.info(f"   {index}: {class_name}")
        
        return self.train_ds, self.val_ds
    
    def build_model(self, fine_tune_layers=50):
        """Build model using pre-trained EfficientNetV2M with transfer learning"""
        
        logger.info("🏗️ Building EfficientNetV2M-based model...")
        logger.info("📦 Downloading pre-trained EfficientNetV2M weights (this may take a moment)...")
        
        # Load pre-trained EfficientNetV2M without top classification layer
        self.base_model = EfficientNetV2M(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        logger.info(f"✅ Loaded EfficientNetV2M with {self.base_model.count_params():,} parameters")
        
        # Freeze base model initially for transfer learning
        self.base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing (EfficientNet expects values in [0, 1])
        x = inputs
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layer for feature learning
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Another dense layer
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile with initial learning rate for transfer learning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        logger.info(f"✅ Model built with {self.model.count_params():,} total parameters")
        logger.info(f"🔒 Frozen parameters: {self.base_model.count_params():,}")
        logger.info(f"🔄 Trainable parameters: {self.model.count_params() - self.base_model.count_params():,}")
        
        # Show model summary
        self.model.summary()
        
        return self.model
    
    def train_phase1(self, epochs=15):
        """Phase 1: Train only the classification head (frozen base model)"""
        
        logger.info("🎯 Phase 1: Training classification head with frozen EfficientNet")
        logger.info(f"⏱️ Training for {epochs} epochs...")
        
        # Callbacks for phase 1
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '../models/efficientnet_pest_phase1.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Add GPU memory monitoring if available
        if self.gpu_available:
            callbacks.append(self._create_gpu_callback())
        
        device_name = '/GPU:0' if self.gpu_available else '/CPU:0'
        logger.info(f"🎯 Training on device: {device_name}")
        
        with tf.device(device_name):
            history_phase1 = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("✅ Phase 1 training completed!")
        return history_phase1
    
    def train_phase2(self, epochs=10, fine_tune_layers=50):
        """Phase 2: Fine-tune top layers of EfficientNet"""
        
        logger.info("🎯 Phase 2: Fine-tuning EfficientNet layers")
        logger.info(f"🔓 Unfreezing top {fine_tune_layers} layers of EfficientNet")
        
        # Unfreeze top layers of the base model
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        trainable_params = sum([tf.keras.utils.count_params(layer.weights) for layer in self.model.layers if layer.trainable])
        logger.info(f"🔄 Now training {trainable_params:,} parameters")
        
        # Callbacks for phase 2
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '../models/efficientnet_pest_final.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Add GPU memory monitoring if available
        if self.gpu_available:
            callbacks.append(self._create_gpu_callback())
        
        device_name = '/GPU:0' if self.gpu_available else '/CPU:0'
        
        with tf.device(device_name):
            history_phase2 = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("✅ Phase 2 fine-tuning completed!")
        return history_phase2
    
    def _create_gpu_callback(self):
        """Create GPU memory monitoring callback"""
        class GPUMemoryCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    used_mb = gpu_info['current'] // (1024 * 1024)
                    peak_mb = gpu_info['peak'] // (1024 * 1024)
                    logger.info(f"🖥️ GPU Memory - Used: {used_mb}MB, Peak: {peak_mb}MB")
                except:
                    pass
        
        return GPUMemoryCallback()
    
    def save_final_model(self):
        """Save the final trained model"""
        final_path = '../models/pest_classifier.h5'
        self.model.save(final_path)
        logger.info(f"💾 Final model saved as '{final_path}'")
        
        # Also save a timestamp version
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'../models/efficientnet_pest_{timestamp}.h5'
        self.model.save(backup_path)
        logger.info(f"📁 Backup saved as '{backup_path}'")
    
    def evaluate_model(self):
        """Evaluate the final model performance"""
        logger.info("📊 Evaluating final model performance...")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_top3 = self.model.evaluate(self.val_ds, verbose=0)
        
        logger.info("=" * 50)
        logger.info("📈 FINAL MODEL PERFORMANCE")
        logger.info("=" * 50)
        logger.info(f"🎯 Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        logger.info(f"🏆 Top-3 Accuracy: {val_top3:.4f} ({val_top3*100:.2f}%)")
        logger.info(f"📉 Validation Loss: {val_loss:.4f}")
        logger.info("=" * 50)
        
        return val_accuracy, val_top3, val_loss

def main():
    """Main training function using EfficientNetV2M"""
    
    logger.info("🌟 EfficientNetV2M Pest Classification Training")
    logger.info("=" * 60)
    
    # Display system information
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"🚀 GPU acceleration available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            logger.info(f"   GPU {i}: {gpu.name}")
    else:
        logger.info("💻 No GPU detected - training will use CPU")
        logger.info("   ⚠️ Warning: EfficientNet training is MUCH faster with GPU!")
    
    logger.info("=" * 60)
    
    # Check dataset
    if not os.path.exists('../dataset'):
        logger.error("❌ Dataset directory not found!")
        logger.info("Please ensure your dataset is in the 'dataset' directory")
        logger.info("Expected structure:")
        logger.info("dataset/")
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
    
    # Initialize EfficientNet classifier
    classifier = EfficientNetPestClassifier(data_dir='../dataset')
    
    # Prepare data
    train_ds, val_ds = classifier.prepare_data(batch_size=16)  # Smaller batch for EfficientNet
    
    # Build model
    model = classifier.build_model()
    
    # Two-phase training approach
    logger.info("🚀 Starting two-phase training approach...")
    
    # Phase 1: Train classification head
    history1 = classifier.train_phase1(epochs=15)
    
    # Phase 2: Fine-tune EfficientNet
    history2 = classifier.train_phase2(epochs=10, fine_tune_layers=50)
    
    # Evaluate final performance
    val_acc, top3_acc, val_loss = classifier.evaluate_model()
    
    # Save final model
    classifier.save_final_model()
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"🏆 Final Accuracy: {val_acc*100:.2f}%")
    logger.info(f"🥉 Top-3 Accuracy: {top3_acc*100:.2f}%")
    
    # Final summary
    if classifier.gpu_available:
        logger.info("🚀 Training used GPU acceleration - EfficientNet performance optimized!")
    else:
        logger.info("💻 Training completed on CPU")
        logger.info("   💡 Tip: Use GPU for much faster EfficientNet training")
    
    logger.info("📁 Model ready for use in your pest classification app!")

if __name__ == "__main__":
    main()
