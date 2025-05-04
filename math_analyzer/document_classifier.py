import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

class DocumentClassifier:
    """Classifies documents by type using image features and optional ML models."""
    
    def __init__(self):
        self._force_document_type = None  # For forcing a specific document type
    
    @property
    def force_document_type(self):
        return self._force_document_type
        
    @force_document_type.setter
    def force_document_type(self, value):
        self._force_document_type = value
    """
    Classifies academic documents and determines appropriate processing strategies
    based on layout and content characteristics.
    """
    
    DOCUMENT_TYPES = {
        'MATH_EXAM': 'math_exam',
        'ESSAY': 'essay', 
        'FORM': 'form',
        'STUDENT_RECORD': 'student_record'
    }
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        
        # Load model if provided and exists
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except:
                print(f"Could not load classifier model from {model_path}.")
    
    def extract_features(self, image):
        """
        Extract layout and content features from document image
        
        Args:
            image (numpy.ndarray): Input document image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Get dimensions
        h, w = gray.shape
        
        # Calculate feature: aspect ratio
        aspect_ratio = w / h
        
        # Calculate feature: text density
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_pixel_count = np.sum(thresh == 255)
        text_density = text_pixel_count / (h * w)
        
        # Calculate feature: lines count (horizontal projection)
        h_proj = np.sum(thresh, axis=1)
        threshold = np.mean(h_proj) * 1.5
        line_count = np.sum(np.diff(h_proj > threshold) > 0)
        
        # Calculate feature: column count (vertical projection)
        v_proj = np.sum(thresh, axis=0)
        threshold = np.mean(v_proj) * 1.5
        column_count = np.sum(np.diff(v_proj > threshold) > 0)
        
        # Calculate feature: detect tables
        horizontal = np.copy(thresh)
        vertical = np.copy(thresh)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = np.sum(horizontal > 0)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = np.sum(vertical > 0)
        
        # Detect potential equation regions
        potential_equation = np.zeros_like(gray)
        # Look for dense blocks of text with special characters like =, +, etc.
        # This is a simplified version - in production, use proper equation detection
        equation_regions = self._detect_equation_regions(gray)
        
        # Return feature vector
        return np.array([
            aspect_ratio, 
            text_density, 
            line_count, 
            column_count, 
            horizontal_lines,
            vertical_lines,
            equation_regions
        ])
    
    def _detect_equation_regions(self, gray_image):
        """Helper method to detect potential equation regions"""
        # Use template matching to find common math symbols
        templates = []
        # This would have actual equation symbol templates
        # Example - create equals sign template
        equals_template = np.zeros((30, 30), dtype=np.uint8)
        cv2.line(equals_template, (5, 15), (25, 15), 255, 2)
        cv2.line(equals_template, (5, 20), (25, 20), 255, 2)
        templates.append(equals_template)
        
        # Count matches for each template
        match_count = 0
        for template in templates:
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            matches = np.sum(result > 0.7)  # Threshold can be adjusted
            match_count += matches
            
        return match_count
    
    def classify_document(self, image):
        """
        Classify document type based on extracted features
        
        Args:
            image (numpy.ndarray): Input document image
            
        Returns:
            str: Document type classification
        """
        # Check if a document type is being forced
        if hasattr(self, '_force_document_type') and self._force_document_type:
            print(f"Using forced document type: {self._force_document_type}")
            return self._force_document_type
        
        features = self.extract_features(image)
        
        # If we have a trained model, use it
        if self.model:
            doc_type = self.model.predict([features])[0]
            return doc_type
            
        # Otherwise use rule-based classification
        # These thresholds would need to be calibrated based on your specific documents
        if features[6] > 10:  # High number of equation regions
            return self.DOCUMENT_TYPES['MATH_EXAM']
        elif features[1] > 0.3 and features[4] < 5 and features[5] < 5:  # High text density, few table lines
            return self.DOCUMENT_TYPES['ESSAY']
        elif features[4] > 10 or features[5] > 10:  # Many horizontal or vertical lines (tables)
            return self.DOCUMENT_TYPES['FORM']
        else:
            return self.DOCUMENT_TYPES['STUDENT_RECORD']
    
    def train_model(self, images, labels):
        """
        Train a simple KMeans model to cluster document types
        
        Args:
            images (list): List of document images
            labels (list): Corresponding document type labels
            
        Returns:
            bool: Success status
        """
        if len(images) != len(labels):
            print("Error: Number of images and labels must match.")
            return False
            
        # Extract features from all images
        features = [self.extract_features(img) for img in images]
        
        # Train KMeans model
        self.model = KMeans(n_clusters=len(set(labels)))
        self.model.fit(features)
        
        # Save model if path is provided
        if self.model_path:
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.load(self.model, f)
                return True
            except:
                print(f"Could not save model to {self.model_path}")
                return False
        
        return True
