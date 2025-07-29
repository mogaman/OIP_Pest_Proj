#!/usr/bin/env python3
"""
Create test images for manual testing of the pest identification system
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_images():
    """Create various test images to simulate different pest scenarios"""
    
    # Create test images directory
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    test_cases = [
        {
            'name': 'green_aphids_test.jpg',
            'color': (50, 150, 50),  # Green (should predict Aphids)
            'description': 'Green dominated image - should predict Aphids'
        },
        {
            'name': 'red_mites_test.jpg', 
            'color': (150, 50, 50),  # Red (should predict Mites)
            'description': 'Red dominated image - should predict Mites'
        },
        {
            'name': 'blue_thrips_test.jpg',
            'color': (50, 50, 150),  # Blue (should predict Thrips)
            'description': 'Blue dominated image - should predict Thrips'
        },
        {
            'name': 'mixed_beetle_test.jpg',
            'color': (100, 100, 80),  # Mixed colors (should predict random)
            'description': 'Mixed colors - should predict Beetle/Grasshopper/Sawfly'
        },
        {
            'name': 'yellow_plant_test.jpg',
            'color': (200, 200, 50),  # Yellow
            'description': 'Yellow plant - will predict random pest'
        }
    ]
    
    print("🖼️  Creating test images for pest identification...")
    
    for case in test_cases:
        # Create base image
        img = Image.new('RGB', (400, 300), color=case['color'])
        draw = ImageDraw.Draw(img)
        
        # Add some texture/patterns to make it more realistic
        for i in range(0, 400, 20):
            for j in range(0, 300, 20):
                # Add slight variations
                variation = (i + j) % 40 - 20
                new_color = tuple(max(0, min(255, c + variation)) for c in case['color'])
                draw.rectangle([i, j, i+10, j+10], fill=new_color)
        
        # Add a label
        try:
            draw.text((10, 10), case['description'], fill='white')
        except:
            pass  # Font might not be available
        
        # Save image
        img.save(os.path.join(test_dir, case['name']))
        print(f"✅ Created: {case['name']} - {case['description']}")
    
    print(f"\n📁 Test images saved in '{test_dir}' directory")
    return test_dir

if __name__ == "__main__":
    create_test_images()
