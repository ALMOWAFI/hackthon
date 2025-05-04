"""
Create sample YOLO annotations for math expressions.

This script helps bootstrap the annotation process by:
1. Processing images in the train folder
2. Using OCR to detect text regions
3. Filtering for likely math expressions
4. Creating YOLO format annotation files

Usage:
    python math_analyzer/create_sample_annotations.py

Note: These are rough initial annotations that should be manually reviewed
and refined using a proper annotation tool.
"""
import os
import glob
import cv2
import numpy as np
from pathlib import Path
import pytesseract
import re

def is_math_expression(text):
    """Check if text looks like a math expression."""
    if not text.strip():
        return False
    # Check for math operators or equals sign
    has_operator = any(op in text for op in "+-*/=^√")
    # Check for numbers
    has_number = any(c.isdigit() for c in text)
    # Check for variables commonly used in math
    has_var = re.search(r'[a-zA-Z]\s*[+\-*/=]', text) is not None
    
    return has_operator and (has_number or has_var)

def normalize_coordinates(box, img_width, img_height):
    """Convert absolute coordinates to normalized YOLO format."""
    x1, y1, x2, y2 = box
    # Calculate center point, width and height (normalized)
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def create_annotations(image_path):
    """Process an image and create YOLO format annotations."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # Use Tesseract OCR to find text regions
    custom_config = r'--psm 6'
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    
    boxes = []
    for i, text in enumerate(data['text']):
        # Skip empty text
        if not text.strip():
            continue
        
        # Check if this looks like a math expression
        if is_math_expression(text):
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            # Add a margin to the box (10%)
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)
            
            # Convert to YOLO format
            x_center, y_center, norm_width, norm_height = normalize_coordinates(
                (x1, y1, x2, y2), width, height)
            
            # Class id 0 for "math_expression"
            boxes.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
    
    return boxes

def process_images():
    """Process all images in the train folder and create annotations."""
    # Setup paths
    base_dir = Path(__file__).parent / "data" / "math_expressions"
    train_img_dir = base_dir / "images" / "train"
    train_label_dir = base_dir / "labels" / "train"
    
    if not train_img_dir.exists():
        print(f"Error: Training image directory not found at {train_img_dir}")
        return
    
    # Create labels directory if it doesn't exist
    train_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.jpeg")) + list(train_img_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {train_img_dir}")
        return
    
    # Process each image
    total_boxes = 0
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        annotations = create_annotations(img_path)
        total_boxes += len(annotations)
        
        # Write annotations to file
        label_path = train_label_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            for annotation in annotations:
                f.write(f"{annotation}\n")
    
    print(f"Completed processing {len(image_files)} images with {total_boxes} detected math expressions.")
    print("Note: These are rough initial annotations and should be manually verified and refined.")

if __name__ == "__main__":
    process_images()
