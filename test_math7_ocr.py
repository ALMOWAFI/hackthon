import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
import re

def preprocess_for_ocr(image):
    """Apply special preprocessing for math notation"""
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding with GAUSSIAN
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 5
    )
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect components
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    # Invert back for OCR
    processed = cv2.bitwise_not(dilated)
    
    # Add padding for better recognition
    padded = cv2.copyMakeBorder(processed, 20, 20, 20, 20, 
                               cv2.BORDER_CONSTANT, value=255)
    
    return padded

def extract_text_with_tesseract(image, config=None):
    """Extract text using Tesseract with optimized settings for math"""
    # Default config optimized for math notation
    if config is None:
        config = r'--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{}^._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'
    
    # Use Tesseract to extract text
    try:
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def normalize_math_expression(text):
    """Normalize mathematical expression for better readability"""
    # Replace common substitutions in OCR output
    replacements = {
        '×': '*',
        '÷': '/',
        '^': '^',
        '²': '^2',
        '³': '^3',
        'α': 'alpha',
        'β': 'beta',
        'π': 'pi'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    # Clean up spaces around operators
    text = re.sub(r'\s*([+\-*/=])\s*', r' \1 ', text)
    
    return text

def segment_image(image, num_segments=4):
    """Segment the image into horizontal sections for better OCR"""
    height, width = image.shape[:2]
    segment_height = height // num_segments
    
    segments = []
    for i in range(num_segments):
        y_start = i * segment_height
        y_end = (i + 1) * segment_height if i < num_segments - 1 else height
        segment = image[y_start:y_end, 0:width]
        segments.append(segment)
        
    return segments

def analyze_math7():
    """Analyze math7.jpeg with optimized OCR for math"""
    image_path = "uploads/math7.jpeg"
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Preprocess the image
    print("Preprocessing image...")
    processed = preprocess_for_ocr(image)
    
    # Segment the image for better OCR
    print("Segmenting image...")
    segments = segment_image(processed)
    
    # Display the original and processed image
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the preprocessing result
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "math7_processed.jpg")
    plt.savefig(output_path)
    print(f"Saved processed image to {output_path}")
    
    # Extract text from each segment
    print("\nExtracting text from segments...")
    all_text = ""
    
    # Create a figure to display segments
    plt.figure(figsize=(12, 12))
    
    for i, segment in enumerate(segments):
        # Try different PSM modes
        psm_modes = [6, 11, 7]  # Single block, single line, single line including spaces
        
        best_text = ""
        best_confidence = 0
        
        for psm in psm_modes:
            config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{{}}^._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'
            segment_text = extract_text_with_tesseract(segment, config)
            
            # Simple confidence heuristic - more math symbols = better
            confidence = sum(c in "+-*/()=^" for c in segment_text)
            
            if confidence > best_confidence or not best_text:
                best_text = segment_text
                best_confidence = confidence
        
        normalized_text = normalize_math_expression(best_text)
        print(f"Segment {i+1}: {normalized_text}")
        all_text += normalized_text + "\n"
        
        # Display the segment
        plt.subplot(len(segments), 1, i+1)
        plt.title(f"Segment {i+1}: {normalized_text}")
        plt.imshow(segment, cmap='gray')
        plt.axis('off')
    
    # Save segments with extracted text
    segments_output_path = os.path.join(output_dir, "math7_segments.jpg")
    plt.tight_layout()
    plt.savefig(segments_output_path)
    print(f"Saved segments with text to {segments_output_path}")
    
    # Print the final extracted text
    print("\nFull extracted text:")
    print(all_text)
    
    # Save the extracted text to a file
    text_output_path = os.path.join(output_dir, "math7_text.txt")
    with open(text_output_path, 'w') as f:
        f.write(all_text)
    print(f"Saved extracted text to {text_output_path}")
    
    # Display the images
    plt.show()
    
    return image, all_text

if __name__ == "__main__":
    analyze_math7()
