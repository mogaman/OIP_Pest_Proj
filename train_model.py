"""
Model training script for OrganicGuard AI Pest Classification
This script trains a CNN model for pest identification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_CONFIG, IMAGE_PREPROCESSING, PEST_CLASSES
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PestClassifierTrainer:
    """Class for training pest classification model"""
    
    def __init__(self, data_dir='pest_dataset', model_save_path='models/pest_classifier.h5'):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.model = None
        self.history = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
    def create_model(self):
        """Create CNN model architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=MODEL_CONFIG['input_shape']),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(MODEL_CONFIG['num_classes'], activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data_generators(self):
        """Prepare data generators for training and validation"""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=IMAGE_PREPROCESSING['rescale'],
            rotation_range=IMAGE_PREPROCESSING['rotation_range'],
            width_shift_range=IMAGE_PREPROCESSING['width_shift_range'],
            height_shift_range=IMAGE_PREPROCESSING['height_shift_range'],
            horizontal_flip=IMAGE_PREPROCESSING['horizontal_flip'],
            vertical_flip=IMAGE_PREPROCESSING['vertical_flip'],
            zoom_range=IMAGE_PREPROCESSING['zoom_range'],
            fill_mode=IMAGE_PREPROCESSING['fill_mode'],
            validation_split=MODEL_CONFIG['validation_split']
        )
        
        # Validation data (no augmentation)
        validation_datagen = ImageDataGenerator(
            rescale=IMAGE_PREPROCESSING['rescale'],
            validation_split=MODEL_CONFIG['validation_split']
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=IMAGE_PREPROCESSING['target_size'],
            batch_size=MODEL_CONFIG['batch_size'],
            class_mode='categorical',
            subset='training',
            classes=PEST_CLASSES,
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            self.data_dir,
            target_size=IMAGE_PREPROCESSING['target_size'],
            batch_size=MODEL_CONFIG['batch_size'],
            class_mode='categorical',
            subset='validation',
            classes=PEST_CLASSES,
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=MODEL_CONFIG['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger('logs/training_log.csv')
        ]
        
        return callbacks_list
    
    def train_model(self):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} not found!")
            logger.info("Please organize your pest images in the following structure:")
            logger.info(f"{self.data_dir}/")
            for pest_class in PEST_CLASSES:
                logger.info(f"  {pest_class}/")
                logger.info(f"    image1.jpg")
                logger.info(f"    image2.jpg")
                logger.info(f"    ...")
            return None
        
        # Create model
        self.create_model()
        logger.info(f"Model created with {self.model.count_params()} parameters")
        
        # Prepare data
        train_gen, val_gen = self.prepare_data_generators()
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            epochs=MODEL_CONFIG['epochs'],
            validation_data=val_gen,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate_model(self, test_data_dir=None):
        """Evaluate model performance"""
        if self.model is None:
            logger.error("No model to evaluate. Train model first.")
            return None
        
        if test_data_dir and os.path.exists(test_data_dir):
            # Use separate test directory
            test_datagen = ImageDataGenerator(rescale=1.0/255.0)
            test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=IMAGE_PREPROCESSING['target_size'],
                batch_size=1,
                class_mode='categorical',
                classes=PEST_CLASSES,
                shuffle=False
            )
            
            # Evaluate
            test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
                test_generator, verbose=1
            )
            
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
            
            # Generate predictions for classification report
            predictions = self.model.predict(test_generator)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = test_generator.classes
            
            # Classification report
            report = classification_report(
                true_classes, 
                predicted_classes, 
                target_names=PEST_CLASSES
            )
            logger.info(f"Classification Report:\n{report}")
            
            return {
                'test_accuracy': test_accuracy,
                'test_top3_accuracy': test_top3_accuracy,
                'classification_report': report
            }
        else:
            logger.warning("No test data directory provided or found.")
            return None
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.error("No training history to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        if 'top_3_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            axes[1, 0].set_title('Model Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sample_dataset(self):
        """Create a sample dataset structure for demonstration"""
        logger.info("Creating sample dataset structure...")
        
        for pest_class in PEST_CLASSES:
            class_dir = os.path.join(self.data_dir, pest_class)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create a dummy image file for each class
            dummy_image = np.random.randint(0, 255, 
                                          (224, 224, 3), 
                                          dtype=np.uint8)
            
            # Save using PIL
            from PIL import Image
            img = Image.fromarray(dummy_image)
            img.save(os.path.join(class_dir, f'sample_{pest_class.lower()}.jpg'))
        
        logger.info(f"Sample dataset created in {self.data_dir}")
        logger.info("Replace sample images with real pest images for actual training.")

def main():
    """Main training function"""
    # Create trainer instance
    trainer = PestClassifierTrainer()
    
    # Create sample dataset if needed
    if not os.path.exists(trainer.data_dir):
        logger.info("No dataset found. Creating sample dataset structure...")
        trainer.create_sample_dataset()
        logger.info("Please replace sample images with real pest images before training.")
        return
    
    # Train model
    history = trainer.train_model()
    
    if history:
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        trainer.evaluate_model()
        
        logger.info(f"Model saved to {trainer.model_save_path}")
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
