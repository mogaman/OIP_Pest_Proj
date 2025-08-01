# 🚀 Training Scripts for Organic Pest Classification

This folder contains all the AI model training scripts for your organic pest management system.

## 📁 Scripts Overview

| **Script** | **Model Type** | **Accuracy** | **Training Time** | **Best For** |
|------------|----------------|--------------|-------------------|--------------|
| `train_model_simple.py` | Simple CNN | 70-85% | Fast (~30 min) | Learning/Testing |
| `train_efficientnet.py` | EfficientNetV2M | 85-95% | Medium (~1-2 hours) | **Recommended** |
| `train_convnext.py` | ConvNeXt | 88-96% | Medium (~1-2 hours) | Highest Accuracy |

## 🎯 Recommended Usage

### **For Production Use:**
```bash
cd training_scripts
python train_efficientnet.py
```

### **For Quick Testing:**
```bash
cd training_scripts
python train_model_simple.py
```

### **For Maximum Accuracy:**
```bash
cd training_scripts
python train_convnext.py
```

## 📊 What Each Script Does

### **train_efficientnet.py** ⭐ **RECOMMENDED**
- **Model**: Pre-trained EfficientNetV2M with transfer learning
- **Features**: 
  - Two-phase training (head first, then fine-tuning)
  - GPU acceleration with memory optimization
  - Advanced data augmentation
  - Top-3 accuracy tracking
- **Expected Results**: 85-95% accuracy
- **Best Choice**: Excellent balance of accuracy and training time

### **train_model_simple.py**
- **Model**: Custom CNN from scratch
- **Features**:
  - Simple architecture
  - GPU acceleration
  - Basic data augmentation
- **Expected Results**: 70-85% accuracy
- **Best For**: Understanding the basics, quick tests

### **train_convnext.py**
- **Model**: Pre-trained ConvNeXt (modern CNN)
- **Features**:
  - State-of-the-art architecture
  - GELU activations and LayerNorm
  - AdamW optimizer
- **Expected Results**: 88-96% accuracy
- **Best For**: Maximum accuracy requirements

## 🔧 Requirements

All scripts automatically:
- ✅ Load pest classes from `../config.py`
- ✅ Use dataset from `../dataset/` folder
- ✅ Save models to `../models/` folder
- ✅ Detect and use GPU if available
- ✅ Fall back to CPU if no GPU

## 📂 File Structure Expected

```
organic_farm_pest/
├── dataset/
│   ├── ants/
│   ├── bees/
│   ├── beetle/
│   ├── catterpillar/
│   ├── earthworms/
│   ├── earwig/
│   ├── grasshopper/
│   ├── moth/
│   ├── slug/
│   ├── snail/
│   ├── wasp/
│   └── weevil/
├── models/              # Models saved here
├── training_scripts/    # THIS FOLDER
├── config.py           # Pest class definitions
└── app.py             # Main web application
```

## 🚀 Training Process

1. **Run a training script** from this folder
2. **Models are saved** to `../models/pest_classifier.h5`
3. **Your web app** (`../app.py`) automatically uses the trained model
4. **Test integration** with `../test_integration.py`

## 📈 Model Performance

After training, your models will be saved as:
- `../models/pest_classifier.h5` - Main model (used by app.py)
- `../models/efficientnet_pest_final.h5` - EfficientNet backup
- `../models/efficientnet_pest_TIMESTAMP.h5` - Timestamped backup

## 💡 Tips

### **For Best Results:**
- Use GPU for training (NVIDIA GTX/RTX series)
- Ensure you have at least 100+ images per pest class
- Run EfficientNet training for production use

### **Troubleshooting:**
- If GPU memory errors: Reduce batch size in the script
- If accuracy is low: Check your dataset quality
- If training is slow: Use GPU or try the simple CNN first

## 🔄 Usage Examples

### **Basic Training:**
```bash
cd training_scripts
python train_efficientnet.py
```

### **Check Results:**
```bash
cd ..
python test_integration.py
python app.py
```

### **View Training Logs:**
Training progress is displayed in real-time with emojis and progress indicators.

## 📞 Integration

After training, your model automatically integrates with:
- ✅ `app.py` - Flask web application
- ✅ Real image quality analysis
- ✅ 12 pest class identification
- ✅ Confidence scoring

Your organic pest management AI is ready to help farmers! 🌱🐛
