import pytesseract
import cv2
import numpy as np
import os
from .config import TESSERACT_CONFIG, IMAGE_SETTINGS

# Set Tesseract path
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

class OCRProcessor:
    def __init__(self):
        self.config = TESSERACT_CONFIG
        
    def preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if needed
        if gray.shape[1] > IMAGE_SETTINGS['preprocess']['resize_width']:
            scale = IMAGE_SETTINGS['preprocess']['resize_width'] / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Apply adaptive thresholding for better handling of handwritten text
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Invert back
        thresh = cv2.bitwise_not(thresh)
        
        return thresh
        
    def extract_text(self, image):
        """
        Extract text from an image using OCR.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            str: Extracted text
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Try math-specific OCR
        try:
            # Get detailed OCR data including confidence scores
            data = pytesseract.image_to_data(processed, 
                                            config=self.config['math_config'],
                                            output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence results
            text_parts = []
            for i, conf in enumerate(data['conf']):
                if conf > 60:  # Only keep high confidence results
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
            
            text = ' '.join(text_parts)
            if text.strip():
                return text
        except Exception as e:
            print(f"Math OCR error: {str(e)}")
            
        # Fall back to regular OCR with different preprocessing
        try:
            # Use regular binary threshold for fallback
            _, binary = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config=self.config['config'])
            return text
        except Exception as e:
            print(f"Regular OCR error: {str(e)}")
            return ""
        
    def extract_math_symbols(self, image):
        """
        Extract mathematical symbols and expressions from an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of detected math expressions
        """
        processed = self.preprocess_image(image)
        
        # Use math-specific configuration
        text = pytesseract.image_to_string(processed,
                                         config=self.config['math_config'])
        
        # Split into individual expressions
        expressions = [expr.strip() for expr in text.split('\n') if expr.strip()]
        
        return expressions 