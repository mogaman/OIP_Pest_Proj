"""
ConvNeXt-based pest classification training script
ConvNeXt often outperforms both EfficientNet and Vision Transformers
while being more efficient than ViT
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup and configure GPU if available"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"🚀 Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            return True
        else:
            logger.warning("⚠️ No GPU found - using CPU")
            return False
    except Exception as e:
        logger.error(f"❌ GPU setup failed: {e}")
        return False

class ConvNeXtPestClassifier:
    """Advanced pest classifier using ConvNeXt architecture"""
    
    def __init__(self, data_dir='dataset', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        
        try:
            from config import PEST_CLASSES
            self.class_names = PEST_CLASSES
        except ImportError:
            self.class_names = [
                'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
                'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
            ]
        
        self.model = None
        self.base_model = None
        self.gpu_available = setup_gpu()
        
        os.makedirs('models', exist_ok=True)
        logger.info(f"🎯 ConvNeXt classifier for {len(self.class_names)} pest classes")
    
    def prepare_data(self, batch_size=32):
        """Prepare data with augmentation optimized for ConvNeXt"""
        
        # ConvNeXt-optimized augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
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
        
        logger.info(f"📈 Training: {self.train_ds.samples}, Validation: {self.val_ds.samples}")
        return self.train_ds, self.val_ds
    
    def build_model(self):
        """Build ConvNeXt-based model"""
        
        logger.info("🏗️ Building ConvNeXt-based model...")
        
        # Load pre-trained ConvNeXt
        self.base_model = ConvNeXtBase(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        logger.info(f"✅ Loaded ConvNeXt with {self.base_model.count_params():,} parameters")
        
        # Freeze initially
        self.base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='gelu')(x)  # GELU works better with ConvNeXt
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile with AdamW (better for ConvNeXt)
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.05),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        logger.info(f"✅ ConvNeXt model ready with {self.model.count_params():,} total parameters")
        return self.model
    
    def train(self, epochs=25):
        """Train ConvNeXt model with progressive unfreezing"""
        
        logger.info("🚀 Training ConvNeXt model...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4),
            keras.callbacks.ModelCheckpoint(
                'models/convnext_pest_classifier.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Phase 1: Train head only
        logger.info("📚 Phase 1: Training classification head...")
        history1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs//2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Unfreeze and fine-tune
        logger.info("🔓 Phase 2: Fine-tuning ConvNeXt layers...")
        self.base_model.trainable = True
        
        # Lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.05),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        history2 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs//2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/pest_classifier.h5')  # Standard name for app.py
        logger.info("✅ ConvNeXt training completed!")
        
        return history1, history2

def main():
    """Main training function"""
    
    logger.info("🌟 ConvNeXt Pest Classification Training")
    logger.info("=" * 50)
    
    if not os.path.exists('dataset'):
        logger.error("❌ Dataset directory not found!")
        return
    
    # Initialize and train
    classifier = ConvNeXtPestClassifier(data_dir='dataset')
    classifier.prepare_data(batch_size=16)
    classifier.build_model()
    classifier.train(epochs=20)
    
    logger.info("🎉 ConvNeXt model ready for your pest app!")

if __name__ == "__main__":
    main()
