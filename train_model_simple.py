"""
Simple model training script following main.py approach
This script trains a CNN model for pest identification using the same approach as main.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePestClassifier:
    """Simple pest classifier following main.py approach"""
    
    def __init__(self, data_dir='dataset', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        # Using your existing pest classes from config
        from config import PEST_CLASSES
        self.class_names = PEST_CLASSES
        self.model = None
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
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
        """Train model following main.py approach"""
        
        logger.info(f"🚀 Training model for {epochs} epochs...")
        
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
                'models/pest_classifier.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/pest_classifier.h5')
        logger.info("✅ Model saved as 'models/pest_classifier.h5'")
        
        return history

def main():
    """Main training function following main.py pattern"""
    
    logger.info("🚀 Simple CNN Training (following main.py approach)")
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
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
    
    # Initialize classifier
    classifier = SimplePestClassifier(data_dir='dataset')
    
    # Prepare data
    train_ds, val_ds = classifier.prepare_data()
    
    # Build model
    model = classifier.build_model()
    
    # Train model
    history = classifier.train_model(epochs=20)
    
    logger.info("✅ Training completed successfully!")
    logger.info("📁 Model saved to: models/pest_classifier.h5")

if __name__ == "__main__":
    main()
