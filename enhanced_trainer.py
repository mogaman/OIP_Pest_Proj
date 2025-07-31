"""
Enhanced CNN Training Module for OrganicGuard AI Pest Classification
Integrates advanced training features with existing project structure
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from datetime import datetime
import json
import shutil
from config import Config, PEST_CLASSES

class EnhancedPestClassifier:
    """Enhanced CNN trainer with advanced features"""
    
    def __init__(self, data_dir='pest_dataset', img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = PEST_CLASSES  # Use classes from config
        self.model = None
        self.class_weights = None
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def prepare_data(self, batch_size=32, validation_split=0.2, random_seed=123):
        """Enhanced data preparation with configurable train/validation split"""
        
        print("📊 Loading dataset with tf.keras.utils.image_dataset_from_directory...")
        print(f"   🔄 Train/Validation split: {int((1-validation_split)*100)}%/{int(validation_split*100)}%")
        print(f"   🎲 Random seed: {random_seed} (for reproducible splits)")
        
        # Use the newer, more reliable data loading method
        try:
            self.train_ds = tf.keras.utils.image_dataset_from_directory(
                self.data_dir,
                validation_split=validation_split,
                subset="training",
                seed=random_seed,
                image_size=self.img_size,
                batch_size=batch_size,
                label_mode='categorical'
            )
            
            self.val_ds = tf.keras.utils.image_dataset_from_directory(
                self.data_dir,
                validation_split=validation_split,
                subset="validation", 
                seed=random_seed,
                image_size=self.img_size,
                batch_size=batch_size,
                label_mode='categorical'
            )
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("💡 Make sure your dataset is in the correct directory structure")
            return None, None
        
        # Calculate class weights for imbalanced dataset
        self._calculate_class_weights()
        
        # Enhanced preprocessing and augmentation
        def preprocess_data(images, labels):
            # Normalize to [0, 1]
            images = tf.cast(images, tf.float32) / 255.0
            return images, labels
        
        def enhanced_augment_data(images, labels):
            """Comprehensive data augmentation for better generalization"""
            # Geometric transformations
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
            
            # Rotation (small angles to preserve pest orientation)
            images = tf.image.rot90(images, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            
            # Color augmentations
            images = tf.image.random_brightness(images, 0.15)
            images = tf.image.random_contrast(images, 0.85, 1.15)
            images = tf.image.random_saturation(images, 0.85, 1.15)
            images = tf.image.random_hue(images, 0.05)
            
            # Zoom and crop (simulate different distances/scales)
            images = tf.image.random_crop(
                tf.image.resize(images, [int(self.img_size[0] * 1.1), int(self.img_size[1] * 1.1)]),
                [tf.shape(images)[0], self.img_size[0], self.img_size[1], 3]
            )
            
            # Add slight noise for robustness
            noise = tf.random.normal(tf.shape(images), mean=0, stddev=0.01)
            images = tf.clip_by_value(images + noise, 0.0, 1.0)
            
            return images, labels
        
        # Apply preprocessing
        self.train_ds = self.train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply enhanced augmentation only to training data
        self.train_ds = self.train_ds.map(enhanced_augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optimize performance with caching and prefetching
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        return self.train_ds, self.val_ds
    
    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalanced dataset"""
        class_counts = {}
        for class_name in self.class_names:
            # Handle different directory naming conventions
            possible_names = [class_name.lower(), class_name.lower() + 's', class_name.lower().rstrip('s')]
            
            for name in possible_names:
                class_dir = os.path.join(self.data_dir, name)
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                    class_counts[class_name] = count
                    break
        
        if not class_counts:
            print("⚠️ Could not calculate class weights - using uniform weights")
            self.class_weights = None
            return
        
        # Calculate balanced weights
        total_samples = sum(class_counts.values())
        n_classes = len(class_counts)
        
        self.class_weights = {}
        for i, class_name in enumerate(self.class_names):
            if class_name in class_counts:
                # Higher weight for underrepresented classes
                self.class_weights[i] = total_samples / (n_classes * class_counts[class_name])
            else:
                self.class_weights[i] = 1.0
        
        print(f"📊 Class distribution: {class_counts}")
        print(f"⚖️ Calculated class weights for balanced training")
    
    def build_enhanced_cnn(self):
        """Build enhanced custom CNN with residual connections and better architecture"""
        
        print("🏗️ Building enhanced custom CNN from scratch...")
        
        # Input layer
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Initial feature extraction
        x = layers.Conv2D(32, (7, 7), strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        # Enhanced Conv Block with residual connections
        def enhanced_conv_block(x, filters, stage, block_id):
            """Enhanced convolutional block with residual connection"""
            shortcut = x
            
            # First conv layer
            x = layers.Conv2D(filters, (3, 3), padding='same', name=f'conv{stage}_{block_id}_1')(x)
            x = layers.BatchNormalization(name=f'bn{stage}_{block_id}_1')(x)
            x = layers.Activation('relu', name=f'relu{stage}_{block_id}_1')(x)
            
            # Second conv layer
            x = layers.Conv2D(filters, (3, 3), padding='same', name=f'conv{stage}_{block_id}_2')(x)
            x = layers.BatchNormalization(name=f'bn{stage}_{block_id}_2')(x)
            
            # Residual connection (if dimensions match)
            if shortcut.shape[-1] == filters:
                x = layers.Add(name=f'add{stage}_{block_id}')([shortcut, x])
            else:
                # Project shortcut to match dimensions
                shortcut = layers.Conv2D(filters, (1, 1), padding='same', 
                                       name=f'shortcut{stage}_{block_id}')(shortcut)
                shortcut = layers.BatchNormalization(name=f'shortcut_bn{stage}_{block_id}')(shortcut)
                x = layers.Add(name=f'add{stage}_{block_id}')([shortcut, x])
            
            x = layers.Activation('relu', name=f'relu{stage}_{block_id}_final')(x)
            return x
        
        def transition_block(x, filters, stage):
            """Transition block with pooling and dropout"""
            x = enhanced_conv_block(x, filters, stage, 'transition')
            x = layers.MaxPooling2D((2, 2), name=f'pool{stage}')(x)
            x = layers.Dropout(0.1, name=f'dropout{stage}')(x)
            return x
        
        # Progressive feature extraction with residual connections
        x = transition_block(x, 64, 1)    # 64 filters
        x = enhanced_conv_block(x, 64, 1, 'extra')
        
        x = transition_block(x, 128, 2)   # 128 filters  
        x = enhanced_conv_block(x, 128, 2, 'extra')
        
        x = transition_block(x, 256, 3)   # 256 filters
        x = enhanced_conv_block(x, 256, 3, 'extra')
        
        x = transition_block(x, 512, 4)   # 512 filters
        x = enhanced_conv_block(x, 512, 4, 'extra')
        
        # Global Average Pooling instead of Flatten (reduces overfitting)
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Enhanced classifier head with better regularization
        x = layers.Dense(512, name='fc1')(x)
        x = layers.BatchNormalization(name='fc1_bn')(x)
        x = layers.Activation('relu', name='fc1_relu')(x)
        x = layers.Dropout(0.3, name='fc1_dropout')(x)
        
        x = layers.Dense(256, name='fc2')(x)
        x = layers.BatchNormalization(name='fc2_bn')(x)
        x = layers.Activation('relu', name='fc2_relu')(x)
        x = layers.Dropout(0.2, name='fc2_dropout')(x)
        
        # Output layer
        outputs = layers.Dense(len(self.class_names), activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='enhanced_pest_classifier')
        
        # Enhanced compilation with label smoothing and better optimizer
        initial_lr = 0.001
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=initial_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        print(f"✅ Enhanced CNN built with {self.model.count_params():,} parameters")
        
        # Print model architecture
        print("\\n📋 Enhanced Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_enhanced(self, epochs=20, resume_from_checkpoint=True):
        """Enhanced training with comprehensive checkpointing and resume capability"""
        
        print(f"🚀 Training enhanced CNN from scratch for {epochs} epochs...")
        print("⏰ Using advanced training strategies for better convergence")
        print("💾 Checkpoint system enabled - safe to cancel and resume!")
        
        # Create checkpoint directory
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define checkpoint paths
        best_model_path = os.path.join(checkpoint_dir, "best_enhanced_model.h5")
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.h5")
        training_state_path = os.path.join(checkpoint_dir, "training_state.json")
        
        # Check for existing checkpoint to resume
        start_epoch = 0
        best_val_acc = 0.0
        
        if resume_from_checkpoint and os.path.exists(training_state_path):
            try:
                with open(training_state_path, 'r') as f:
                    training_state = json.load(f)
                
                if os.path.exists(latest_checkpoint_path):
                    print(f"📥 Found existing checkpoint at epoch {training_state.get('epoch', 0)}")
                    print(f"   🎯 Best validation accuracy so far: {training_state.get('best_val_acc', 0):.3f}")
                    
                    response = input("🔄 Resume from checkpoint? (y/n): ").strip().lower()
                    if response == 'y':
                        self.model = keras.models.load_model(latest_checkpoint_path)
                        start_epoch = training_state.get('epoch', 0)
                        best_val_acc = training_state.get('best_val_acc', 0.0)
                        print(f"✅ Resumed from epoch {start_epoch}")
                    else:
                        print("🆕 Starting fresh training...")
                        
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
                print("🆕 Starting fresh training...")
        
        # Create learning rate schedule
        def cosine_annealing_with_warmup(epoch, lr):
            """Cosine annealing with warmup for better training dynamics"""
            warmup_epochs = 5
            adjusted_epoch = epoch + start_epoch
            
            if adjusted_epoch < warmup_epochs:
                return 0.001 * (adjusted_epoch + 1) / warmup_epochs
            else:
                import math
                cosine_epoch = adjusted_epoch - warmup_epochs
                total_cosine_epochs = epochs - warmup_epochs
                return 0.001 * 0.5 * (1 + math.cos(math.pi * cosine_epoch / total_cosine_epochs))
        
        # Custom callback to save training state
        class TrainingStateCallback(keras.callbacks.Callback):
            def __init__(self, state_path, latest_path):
                self.state_path = state_path
                self.latest_path = latest_path
                self.best_val_acc = best_val_acc
                
            def on_epoch_end(self, epoch, logs=None):
                # Save latest model every epoch
                self.model.save(self.latest_path)
                
                # Update best validation accuracy
                current_val_acc = logs.get('val_accuracy', 0)
                if current_val_acc > self.best_val_acc:
                    self.best_val_acc = current_val_acc
                
                # Save training state
                state = {
                    'epoch': start_epoch + epoch + 1,
                    'total_epochs': epochs,
                    'best_val_acc': float(self.best_val_acc),
                    'current_val_acc': float(current_val_acc),
                    'current_loss': float(logs.get('loss', 0)),
                    'current_val_loss': float(logs.get('val_loss', 0)),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(self.state_path, 'w') as f:
                    json.dump(state, f, indent=2)
                
                print(f"💾 Checkpoint saved - Epoch {start_epoch + epoch + 1}/{epochs}")
        
        # Enhanced callbacks for better training
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                patience=8,
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            
            # Learning rate scheduler
            keras.callbacks.LearningRateScheduler(
                cosine_annealing_with_warmup,
                verbose=1
            ),
            
            # Best model checkpoint
            keras.callbacks.ModelCheckpoint(
                best_model_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            
            # Training state callback
            TrainingStateCallback(training_state_path, latest_checkpoint_path),
            
            # Reduce learning rate on plateau as backup
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=f'./logs/enhanced_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=False
            )
        ]
        
        print("📊 Training with enhanced strategies:")
        print("   • Cosine annealing with warmup")
        print("   • Label smoothing (0.1)")
        print("   • Class balancing weights")
        print("   • Enhanced data augmentation")
        print("   • Residual connections")
        print("   • Global average pooling")
        print("   • 💾 Comprehensive checkpointing system")
        
        # Calculate remaining epochs if resuming
        remaining_epochs = epochs - start_epoch
        
        if remaining_epochs <= 0:
            print(f"✅ Training already complete! ({start_epoch}/{epochs} epochs)")
            if os.path.exists(best_model_path):
                print(f"📥 Loading best model from {best_model_path}")
                self.model = keras.models.load_model(best_model_path)
            return None
        
        print(f"📈 Training epochs {start_epoch + 1} to {epochs} ({remaining_epochs} remaining)")
        
        try:
            # Train with class weights for balanced learning
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=remaining_epochs,
                initial_epoch=start_epoch,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            # Copy best model to final location (compatible with existing app)
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, Config.MODEL_PATH)
                print(f"✅ Enhanced model saved as '{Config.MODEL_PATH}'")
            else:
                # Fallback - save current model
                self.model.save(Config.MODEL_PATH)
                print(f"✅ Enhanced model saved as '{Config.MODEL_PATH}' (current state)")
            
            # Print training summary
            if history and history.history:
                best_val_acc = max(history.history['val_accuracy'])
                best_train_acc = max(history.history['accuracy'])
                final_val_loss = min(history.history['val_loss'])
                
                print(f"\\n📈 Training Summary:")
                print(f"   🎯 Best Validation Accuracy: {best_val_acc:.3f}")
                print(f"   🎯 Best Training Accuracy: {best_train_acc:.3f}")
                print(f"   📉 Final Validation Loss: {final_val_loss:.3f}")
                
        except KeyboardInterrupt:
            print(f"\\n⏹️ Training interrupted by user at epoch {start_epoch}")
            print(f"💾 All checkpoints saved in '{checkpoint_dir}' directory")
            print("🔄 You can resume training later by running the same command")
            
            # Save current state even if interrupted
            if hasattr(self, 'model') and self.model:
                interrupted_path = os.path.join(checkpoint_dir, "interrupted_model.h5")
                self.model.save(interrupted_path)
                print(f"💾 Current model state saved to {interrupted_path}")
            
            return None
        except Exception as e:
            print(f"\\n❌ Training error: {e}")
            print(f"💾 Checkpoints preserved in '{checkpoint_dir}' directory")
            raise
        
        return history

# ============================================================================
# CONVENIENCE FUNCTIONS FOR EASY USAGE
# ============================================================================

def train_enhanced_model(data_dir='pest_dataset', epochs=20):
    """Main function: Train enhanced CNN from scratch with all improvements"""
    
    print("🚀 Enhanced CNN Training for OrganicGuard AI")
    print("📝 Using state-of-the-art techniques for better accuracy!")
    print("🔧 Improvements:")
    print("   • Larger input size (224x224)")
    print("   • Residual connections")
    print("   • Enhanced data augmentation")
    print("   • Class balancing")
    print("   • Label smoothing")
    print("   • Cosine annealing with warmup")
    print("   • Global average pooling")
    
    # Initialize with larger image size
    classifier = EnhancedPestClassifier(data_dir=data_dir, img_size=(224, 224))
    
    # Prepare enhanced data
    print("📊 Preparing enhanced data pipeline...")
    train_ds, val_ds = classifier.prepare_data(batch_size=8)  # Smaller batch for memory efficiency
    
    if train_ds is None:
        print("❌ Failed to prepare data. Please check your dataset directory.")
        return None
    
    # Build enhanced model
    print("🏗️ Building enhanced CNN with residual connections...")
    model = classifier.build_enhanced_cnn()
    
    # Train with enhanced strategies
    history = classifier.train_enhanced(epochs=epochs)
    
    print("✅ Enhanced training complete!")
    print(f"🎯 Model saved to: {Config.MODEL_PATH}")
    print("🚀 You can now run the web application!")
    
    return history

def check_training_progress():
    """Check current training progress and checkpoint status"""
    
    checkpoint_dir = "checkpoints"
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    
    if not os.path.exists(training_state_path):
        print("❌ No training in progress")
        print("💡 Start training with train_enhanced_model()")
        return
    
    try:
        with open(training_state_path, 'r') as f:
            state = json.load(f)
        
        print("📊 Training Progress Status:")
        print(f"   🔄 Epoch: {state.get('epoch', 0)}/{state.get('total_epochs', 20)}")
        print(f"   🎯 Best Validation Accuracy: {state.get('best_val_acc', 0):.3f}")
        print(f"   📈 Current Validation Accuracy: {state.get('current_val_acc', 0):.3f}")
        print(f"   📉 Current Loss: {state.get('current_loss', 0):.4f}")
        print(f"   📉 Current Validation Loss: {state.get('current_val_loss', 0):.4f}")
        print(f"   ⏰ Last Updated: {state.get('timestamp', 'Unknown')}")
        
        # Calculate progress percentage
        progress = (state.get('epoch', 0) / state.get('total_epochs', 20)) * 100
        print(f"   📊 Progress: {progress:.1f}%")
        
        # List available checkpoints
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
            print(f"   💾 Available checkpoints: {len(checkpoint_files)}")
            
    except Exception as e:
        print(f"❌ Error reading training state: {e}")

def visualize_enhanced_model_architecture():
    """Show the enhanced CNN architecture"""
    
    print("🧠 Creating enhanced model to show architecture...")
    classifier = EnhancedPestClassifier(data_dir="pest_dataset", img_size=(224, 224))
    model = classifier.build_enhanced_cnn()
    
    print("\\n" + "="*70)
    print("🏗️ ENHANCED CNN ARCHITECTURE FOR ORGANICGUARD AI")
    print("="*70)
    print("Built from scratch with state-of-the-art improvements!")
    print("✨ Features: Residual connections, Global pooling, Label smoothing")
    print("="*70)
    
    # Show detailed summary
    model.summary()

def cleanup_checkpoints():
    """Clean up training checkpoints"""
    
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print("📁 No checkpoint directory found")
        return
    
    files = os.listdir(checkpoint_dir)
    print(f"🗑️ This will delete {len(files)} checkpoint files:")
    for file in files[:5]:  # Show first 5
        print(f"   • {file}")
    if len(files) > 5:
        print(f"   • ... and {len(files) - 5} more")
    
    confirm = input("\\n⚠️ Are you sure? This cannot be undone! (type 'DELETE' to confirm): ")
    
    if confirm == "DELETE":
        shutil.rmtree(checkpoint_dir)
        print("✅ All checkpoints deleted")
    else:
        print("❌ Cancelled - checkpoints preserved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🌱 OrganicGuard AI - Enhanced Training Module")
    print("="*50)
    
    # Check if model already exists
    if os.path.exists(Config.MODEL_PATH):
        model_size = os.path.getsize(Config.MODEL_PATH) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(Config.MODEL_PATH))
        
        print(f"✅ Found existing model: {Config.MODEL_PATH}")
        print(f"   📊 Size: {model_size:.1f}MB")
        print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        choice = input("\\n🔄 Retrain model? (y/n): ").strip().lower()
        if choice != 'y':
            print("⏭️ Skipping training - using existing model")
            exit()
    
    # Train enhanced model
    print("\\n🚀 Starting enhanced training...")
    train_enhanced_model()
