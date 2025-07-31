"""
Dataset Preparation Utility for OrganicGuard AI
Helps organize and validate the pest dataset structure
"""

import os
import shutil
from pathlib import Path
from config import PEST_CLASSES

def check_dataset_structure(data_dir='pest_dataset'):
    """Check if dataset is properly structured"""
    
    print(f"🔍 Checking dataset structure in '{data_dir}'...")
    print("=" * 50)
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory '{data_dir}' not found!")
        print("💡 Please create the directory and add your pest images")
        return False
    
    total_images = 0
    missing_classes = []
    class_counts = {}
    
    print("📊 Class Distribution:")
    print(f"{'Class':<15} {'Count':<8} {'Status'}")
    print("-" * 35)
    
    for pest_class in PEST_CLASSES:
        # Check for various naming conventions
        possible_dirs = [
            os.path.join(data_dir, pest_class),
            os.path.join(data_dir, pest_class.lower()),
            os.path.join(data_dir, pest_class.lower() + 's'),
            os.path.join(data_dir, pest_class.lower().rstrip('s'))
        ]
        
        found = False
        for class_dir in possible_dirs:
            if os.path.exists(class_dir):
                # Count images
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
                count = len(image_files)
                
                status = "✅" if count >= 10 else "⚠️ " if count > 0 else "❌"
                print(f"{pest_class:<15} {count:<8} {status}")
                
                class_counts[pest_class] = count
                total_images += count
                found = True
                break
        
        if not found:
            print(f"{pest_class:<15} {'0':<8} ❌")
            missing_classes.append(pest_class)
            class_counts[pest_class] = 0
    
    print("-" * 35)
    print(f"{'TOTAL':<15} {total_images:<8}")
    
    # Analysis
    print("\\n📈 Dataset Analysis:")
    print(f"   • Total Images: {total_images}")
    print(f"   • Classes with Data: {len([c for c in class_counts.values() if c > 0])}/{len(PEST_CLASSES)}")
    
    if missing_classes:
        print(f"   • Missing Classes: {len(missing_classes)}")
        print(f"     {', '.join(missing_classes)}")
    
    # Recommendations
    print("\\n💡 Recommendations:")
    if total_images < 100:
        print("   ⚠️ Very small dataset - consider adding more images")
    elif total_images < 500:
        print("   📝 Small dataset - may need data augmentation")
    else:
        print("   ✅ Good dataset size for training")
    
    # Check class balance
    if class_counts:
        min_count = min([c for c in class_counts.values() if c > 0])
        max_count = max(class_counts.values())
        if max_count > 0:
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 3.0:
                print("   ⚠️ Significant class imbalance detected")
                print("   💡 Consider class balancing during training")
            else:
                print("   ✅ Classes are reasonably balanced")
    
    return total_images > 0

def create_dataset_structure(data_dir='pest_dataset'):
    """Create empty dataset directory structure"""
    
    print(f"🏗️ Creating dataset structure in '{data_dir}'...")
    
    base_dir = Path(data_dir)
    base_dir.mkdir(exist_ok=True)
    
    for pest_class in PEST_CLASSES:
        class_dir = base_dir / pest_class.lower()
        class_dir.mkdir(exist_ok=True)
        
        # Create a README file with instructions
        readme_path = class_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"Add {pest_class} images to this directory\\n")
            f.write(f"Supported formats: JPG, PNG, BMP, GIF, WEBP\\n")
            f.write(f"Recommended: At least 10-50 images per class\\n")
    
    print("✅ Dataset structure created!")
    print(f"📁 Add your images to the subdirectories in '{data_dir}'")

def validate_images(data_dir='pest_dataset'):
    """Validate images in the dataset"""
    
    print(f"🔍 Validating images in '{data_dir}'...")
    
    from PIL import Image
    import numpy as np
    
    total_files = 0
    valid_images = 0
    invalid_files = []
    
    for pest_class in PEST_CLASSES:
        class_dirs = [
            os.path.join(data_dir, pest_class),
            os.path.join(data_dir, pest_class.lower()),
            os.path.join(data_dir, pest_class.lower() + 's'),
            os.path.join(data_dir, pest_class.lower().rstrip('s'))
        ]
        
        for class_dir in class_dirs:
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        file_path = os.path.join(class_dir, filename)
                        total_files += 1
                        
                        try:
                            # Try to open and verify image
                            with Image.open(file_path) as img:
                                img.verify()  # Verify it's a valid image
                            
                            # Try to load as array (additional validation)
                            with Image.open(file_path) as img:
                                img_array = np.array(img)
                                if img_array.size > 0:
                                    valid_images += 1
                                else:
                                    invalid_files.append(file_path)
                                    
                        except Exception as e:
                            invalid_files.append(f"{file_path} - {str(e)}")
                break
    
    print(f"\\n📊 Validation Results:")
    print(f"   • Total Files: {total_files}")
    print(f"   • Valid Images: {valid_images}")
    print(f"   • Invalid Files: {len(invalid_files)}")
    
    if invalid_files:
        print("\\n❌ Invalid Files Found:")
        for invalid_file in invalid_files[:10]:  # Show first 10
            print(f"   • {invalid_file}")
        if len(invalid_files) > 10:
            print(f"   • ... and {len(invalid_files) - 10} more")
    
    return len(invalid_files) == 0

def organize_dataset(source_dir, target_dir='pest_dataset'):
    """Organize images from a source directory into the proper structure"""
    
    print(f"🗂️ Organizing dataset from '{source_dir}' to '{target_dir}'...")
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory '{source_dir}' not found!")
        return
    
    # Create target structure
    create_dataset_structure(target_dir)
    
    # Move/copy files (implement based on your needs)
    print("💡 Manual organization required:")
    print(f"   1. Review files in '{source_dir}'")
    print(f"   2. Sort them into appropriate class folders in '{target_dir}'")
    print(f"   3. Ensure each class has at least 10-20 images")

def main():
    """Main function for dataset utilities"""
    
    print("🌱 OrganicGuard AI - Dataset Preparation Utility")
    print("=" * 50)
    
    while True:
        print("\\n📋 Available Options:")
        print("1. Check Dataset Structure")
        print("2. Create Dataset Structure")
        print("3. Validate Images")
        print("4. Get Dataset Info")
        print("5. Exit")
        
        choice = input("\\n🔍 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            check_dataset_structure()
        elif choice == '2':
            create_dataset_structure()
        elif choice == '3':
            validate_images()
        elif choice == '4':
            print("\\n📋 Expected Dataset Structure:")
            print("pest_dataset/")
            for pest_class in PEST_CLASSES:
                print(f"├── {pest_class.lower()}/")
                print(f"│   ├── image1.jpg")
                print(f"│   ├── image2.png")
                print(f"│   └── ...")
            print("\\n💡 Each class should have at least 10-50 images")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
