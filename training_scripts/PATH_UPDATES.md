# Training Scripts Path Updates

## Summary
All training scripts have been updated to work correctly from the `training_scripts/` subfolder.

## Changes Made

### 1. train_model_simple.py
- ✅ Updated dataset path: `'dataset'` → `'../dataset'`
- ✅ Updated models directory: `'models'` → `'../models'`
- ✅ Added sys.path.append for config import
- ✅ Updated ModelCheckpoint callback path
- ✅ Updated final model save path
- ✅ Updated dataset existence check

### 2. train_efficientnet.py
- ✅ Already properly configured with relative paths
- ✅ Uses `../dataset` and `../models/` correctly

### 3. train_convnext.py
- ✅ Updated dataset path: `'dataset'` → `'../dataset'`
- ✅ Updated models directory: `'models/'` → `'../models/'`
- ✅ Added sys.path.append for config import
- ✅ Updated ModelCheckpoint callback path
- ✅ Updated final model save path
- ✅ Updated dataset existence check

## File Structure
```
organic_farm_pest/
├── dataset/           # Training images
├── models/           # Saved model files
└── training_scripts/ # All training scripts (current location)
    ├── train_model_simple.py
    ├── train_efficientnet.py
    ├── train_convnext.py
    ├── train.bat
    └── README.md
```

## Usage
All scripts can now be run from the `training_scripts/` directory and will correctly:
- Find the dataset in `../dataset/`
- Save models to `../models/`
- Import config from the parent directory

## Test Commands
From the `training_scripts/` directory:
```bash
python train_model_simple.py
python train_efficientnet.py
python train_convnext.py
```

Or use the Windows launcher:
```bash
train.bat
```
