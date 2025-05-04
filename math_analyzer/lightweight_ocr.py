"""
Lightweight OCR module for math expressions.

This module provides basic OCR capabilities for math expressions without requiring Tesseract.
It's not as accurate as Tesseract, but it can serve as a fallback when Tesseract is not available.

It uses simple image processing techniques to segment characters and match them against templates.
"""

import os
import cv2
import numpy as np
from pathlib import Path


class LightweightOCR:
    """Basic OCR for math expressions without external dependencies."""
    
    def __init__(self):
        """Initialize with built-in templates for common math symbols."""
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in templates for numbers and math operators."""
        # We'll use template sizes of 20x30 pixels
        template_size = (20, 30)
        
        # Create templates for digits 0-9
        for digit in range(10):
            template = np.zeros(template_size, dtype=np.uint8)
            cv2.putText(
                template, str(digit), (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2
            )
            self.templates[str(digit)] = template
        
        # Create templates for basic operators
        operators = ['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}']
        for op in operators:
            template = np.zeros(template_size, dtype=np.uint8)
            cv2.putText(
                template, op, (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2
            )
            self.templates[op] = template
        
        # Add templates for variables
        for var in 'xyz':
            template = np.zeros(template_size, dtype=np.uint8)
            cv2.putText(
                template, var, (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2
            )
            self.templates[var] = template
    
    def extract_text(self, image):
        """
        Extract text from an image using contour detection and template matching.
        
        Args:
            image: OpenCV image (grayscale or BGR)
            
        Returns:
            Extracted text as a string
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort contours from left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Extract characters and match against templates
        characters = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours
            if w < 5 or h < 5:
                continue
                
            # Skip very large contours
            if w > 50 or h > 50:
                continue
            
            # Extract character
            roi = binary[y:y+h, x:x+w]
            
            # Resize to match template size
            roi_resized = cv2.resize(roi, (20, 30))
            
            # Match against templates
            best_match = None
            best_score = 0
            
            for char, template in self.templates.items():
                # Use template matching
                result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                
                if score > best_score and score > 0.5:  # 0.5 is the threshold
                    best_score = score
                    best_match = char
            
            if best_match:
                characters.append((x, best_match))
        
        # Sort by x-coordinate and join
        characters.sort(key=lambda c: c[0])
        return ''.join(char for _, char in characters)
    
    def recognize_expressions(self, image):
        """
        Recognize math expressions in an image and return bounding boxes.
        
        Args:
            image: OpenCV image
            
        Returns:
            List of (x1, y1, x2, y2, text) tuples
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to connect nearby characters
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours for connected components that might be expressions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process potential expression regions
        expressions = []
        min_width = 20
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small regions
            if w < min_width:
                continue
            
            # Skip very thin regions (likely not expressions)
            aspect_ratio = w / h
            if aspect_ratio < 1.0:
                continue
                
            # Extract the region
            roi = gray[y:y+h, x:x+w]
            
            # Try to extract text
            text = self.extract_text(roi)
            
            # Check if it looks like a math expression
            if self._is_math_expression(text):
                # Add some padding
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                
                expressions.append((x1, y1, x2, y2, text))
        
        return expressions
    
    def _is_math_expression(self, text):
        """Check if text looks like a math expression."""
        # Must contain at least one digit
        if not any(c.isdigit() for c in text):
            return False
            
        # Should contain operators or equals sign
        if not any(c in '+-*/=' for c in text):
            return False
            
        # Must be at least 3 characters
        if len(text) < 3:
            return False
            
        return True
