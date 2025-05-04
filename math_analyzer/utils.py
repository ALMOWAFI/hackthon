import os
import json
import base64
import cv2
import numpy as np

def save_results(results, output_dir='output'):
    """
    Save analysis results to files.
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
def image_to_base64(image):
    """
    Convert an OpenCV image to base64 string.
    
    Args:
        image (numpy.ndarray): OpenCV image
        
    Returns:
        str: Base64 encoded image
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')
    
def base64_to_image(base64_string):
    """
    Convert a base64 string to OpenCV image.
    
    Args:
        base64_string (str): Base64 encoded image
        
    Returns:
        numpy.ndarray: OpenCV image
    """
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
def preprocess_image(image):
    """
    Preprocess an image for better OCR results.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return opening
    
def draw_bounding_box(image, box, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on an image.
    
    Args:
        image (numpy.ndarray): Input image
        box (tuple): (x, y, w, h) coordinates
        color (tuple): BGR color
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with bounding box
    """
    x, y, w, h = box
    return cv2.rectangle(image.copy(), (x, y), (x+w, y+h), color, thickness)
    
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (tuple): (x1, y1, w1, h1)
        box2 (tuple): (x2, y2, w2, h2)
        
    Returns:
        float: IoU value
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area 