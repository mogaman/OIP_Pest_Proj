"""
YOLO-based pest detection and classification example
This would be better if you want to LOCATE pests in images, not just classify them
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOPestDetector:
    """
    YOLO-based pest detector that can:
    1. Find WHERE pests are in the image (bounding boxes)
    2. Classify WHAT type of pest each one is
    3. Handle multiple pests in one image
    """
    
    def __init__(self):
        # Load pre-trained YOLO model (would need custom training for pests)
        self.model = YOLO('yolov8n.pt')  # Nano version for speed
        
        # Your pest classes
        self.pest_classes = [
            'ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
            'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil'
        ]
    
    def detect_pests(self, image_path):
        """
        Detect and classify pests in image
        Returns: List of detections with bounding boxes
        """
        results = self.model(image_path)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'pest_type': self.pest_classes[class_id],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],  # Bounding box location
                        'center': [(x1+x2)/2, (y1+y2)/2]  # Center point
                    })
        
        return detections

# Example usage:
# detector = YOLOPestDetector()
# pests_found = detector.detect_pests('farm_image.jpg')
# 
# Result example:
# [
#     {'pest_type': 'aphids', 'confidence': 0.85, 'bbox': [100, 150, 200, 250], 'center': [150, 200]},
#     {'pest_type': 'beetle', 'confidence': 0.72, 'bbox': [300, 100, 400, 180], 'center': [350, 140]}
# ]
