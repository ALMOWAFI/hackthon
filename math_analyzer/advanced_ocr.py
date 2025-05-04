import cv2
import numpy as np
import pytesseract
import sympy
from sympy.parsing.latex import parse_latex
from PIL import Image
import re
import tensorflow as tf
from sklearn.cluster import DBSCAN

class AdvancedMathOCR:
    """Advanced OCR system specifically designed for mathematical notation"""
    
    def __init__(self):
        self.math_symbols = {
            # Basic operations
            '×': '*', '÷': '/', '±': '\\pm', 
            # Powers and roots
            '²': '^2', '³': '^3', '√': '\\sqrt',
            # Greek letters commonly used in math
            'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'θ': '\\theta',
            'π': '\\pi', 'Σ': '\\sum', '∫': '\\int', '∞': '\\infty',
            # Comparisons
            '≈': '\\approx', '≠': '\\neq', '≤': '\\leq', '≥': '\\geq',
            # Set notation
            '∈': '\\in', '∉': '\\notin', '∪': '\\cup', '∩': '\\cap',
            # Fractions and special functions
            'fraction_marker': '\\frac',
            # Other common symbols
            '…': '...', '⋯': '\\cdots'
        }
        
        # Language settings for multi-language support
        self.lang_options = {
            'eng': 'English',
            'fra': 'French',
            'deu': 'German',
            'spa': 'Spanish'
        }
        
        # Configure Tesseract for math
        self.config = r'--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{}^._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'
        
        # Special patterns for complex mathematical structures
        self.patterns = {
            'fraction': r'\\frac\{(.*?)\}\{(.*?)\}',
            'sqrt': r'\\sqrt\{(.*?)\}',
            'power': r'\^{(.*?)}',
            'subscript': r'_{(.*?)}'
        }
        
    def preprocess_math_image(self, image):
        """Apply specialized preprocessing for math notation"""
        if isinstance(image, str):
            # If image is a path
            img = cv2.imread(image)
        else:
            # If image is already a numpy array
            img = image.copy()
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
        
    def detect_math_structure(self, image):
        """Detect mathematical structures like fractions, powers, etc."""
        # Use connected components analysis to identify potential structures
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find horizontal lines that might be fraction bars
        fraction_bars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Potential fraction bar if it's wide and thin
            if aspect_ratio > 3 and h < 10 and w > 20:
                fraction_bars.append((x, y, w, h))
        
        # Find potential superscripts and subscripts
        script_candidates = []
        avg_height = np.mean([cv2.boundingRect(cnt)[3] for cnt in contours]) if contours else 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # If significantly smaller than average height and not too small
            if 0 < h < avg_height * 0.7 and w > 5:
                script_candidates.append((x, y, w, h))
        
        # Return detected structures
        return {
            'fraction_bars': fraction_bars,
            'potential_scripts': script_candidates
        }
        
    def segment_expression(self, image):
        """Segment a mathematical expression into component parts"""
        # Detect structures
        structures = self.detect_math_structure(image)
        
        # Initialize segments with a default full image
        segments = [('full_expr', image)]
        
        # Process fraction bars to segment numerators and denominators
        for x, y, w, h in structures['fraction_bars']:
            # Define areas above and below bar
            padding = 5
            
            # Get original image regions
            height, width = image.shape[:2]
            
            # Define numerator region (above bar)
            num_y1 = max(0, y - 40)  # Allow space for taller expressions
            num_y2 = y
            if num_y1 < num_y2:
                numerator = image[num_y1:num_y2, max(0, x-10):min(width, x+w+10)]
                if numerator.size > 0:
                    segments.append(('numerator', numerator))
            
            # Define denominator region (below bar)
            denom_y1 = y + h
            denom_y2 = min(height, y + h + 40)  # Allow space for taller expressions
            if denom_y1 < denom_y2:
                denominator = image[denom_y1:denom_y2, max(0, x-10):min(width, x+w+10)]
                if denominator.size > 0:
                    segments.append(('denominator', denominator))
        
        return segments
    
    def extract_text_with_tesseract(self, image, lang='eng'):
        """Extract text using Tesseract with language options"""
        # Convert to PIL image for Tesseract
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(pil_image, lang=lang, config=self.config)
        
        # Extract math expressions differently
        math_text = pytesseract.image_to_string(pil_image, lang=lang, 
                                              config='--psm 6 -c tessedit_char_whitelist="0123456789+-*/()=^"')
                                              
        # Clean up the results
        text = text.strip()
        math_text = math_text.strip()
        
        if text and math_text:
            # If both regular text and math text are found, prefer math_text for equations
            if any(c in math_text for c in "+-*/=^()"):
                return math_text
            return text
        
        # Return whichever is non-empty
        return text if text else math_text
        
    def recognize_math_symbol(self, symbol_img):
        """Use template matching to recognize common math symbols"""
        # Placeholder for neural network-based symbol recognition
        # In a complete implementation, a CNN would be trained to recognize
        # math symbols and return the corresponding LaTeX or Unicode
        
        # For now, we'll use a simple approach with template matching
        # to identify some common symbols
        symbol_dict = {
            "plus": "+", "minus": "-", "div": "÷", "mult": "×",
            "equals": "=", "sqrt": "√", "pi": "π"
        }
        
        # Return a placeholder result 
        # In a real implementation, this would use the actual detection result
        return "+"  # Placeholder
        
    def convert_to_latex(self, detected_text):
        """Convert detected text to LaTeX format"""
        result = detected_text
        
        # Replace common math symbols with LaTeX equivalents
        for symbol, latex in self.math_symbols.items():
            result = result.replace(symbol, latex)
        
        # Handle fractions (x/y) -> \frac{x}{y}
        fraction_pattern = r'(\d+|\([^)]+\))/(\d+|\([^)]+\))'
        
        def replace_fraction(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f"\\frac{{{numerator}}}{{{denominator}}}"
            
        result = re.sub(fraction_pattern, replace_fraction, result)
        
        # Handle powers x^2 -> x^{2} for multi-digit exponents
        power_pattern = r'(\w|\))(\^)(\d\d+)'
        
        def replace_power(match):
            base = match.group(1)
            exp = match.group(3)
            return f"{base}^{{{exp}}}"
            
        result = re.sub(power_pattern, replace_power, result)
        
        # Add LaTeX math delimiters
        if any(latex_cmd in result for latex_cmd in ['\\frac', '\\sqrt', '\\alpha']):
            result = f"${result}$"
        
        return result
        
    def convert_latex_to_sympy(self, latex_expr):
        """Convert LaTeX to SymPy expression for calculation"""
        try:
            # Remove LaTeX math delimiters if present
            latex_expr = latex_expr.replace('$', '')
            
            # Convert to sympy expression
            sympy_expr = parse_latex(latex_expr)
            return sympy_expr
        except Exception as e:
            print(f"Error converting LaTeX to SymPy: {e}")
            return None
            
    def extract_math_from_image(self, image, return_format='text'):
        """
        Main method to extract mathematical expressions from images
        
        Args:
            image: Input image (numpy array or path)
            return_format: The desired output format ('text', 'latex', or 'sympy')
            
        Returns:
            Detected mathematical expression in the requested format
        """
        # Process the image
        processed_img = self.preprocess_math_image(image)
        
        # Segment the expression
        segments = self.segment_expression(processed_img)
        
        # Process each segment
        results = {}
        for segment_type, segment_img in segments:
            text = self.extract_text_with_tesseract(segment_img)
            results[segment_type] = text
        
        # If we found fraction components, construct a fraction
        if 'numerator' in results and 'denominator' in results:
            full_expr = f"{results['numerator']}/{results['denominator']}"
        else:
            full_expr = results.get('full_expr', '')
        
        # Clean and normalize
        full_expr = full_expr.replace('\n', ' ').strip()
        
        # Convert to requested format
        if return_format == 'latex':
            return self.convert_to_latex(full_expr)
        elif return_format == 'sympy':
            latex = self.convert_to_latex(full_expr)
            return self.convert_latex_to_sympy(latex)
        else:  # Default to text
            return full_expr

# Advanced OCR Factory for different types of math documents
class MathOCRFactory:
    """Factory to create specialized OCR processors for different types of math documents"""
    
    @staticmethod
    def create_processor(document_type):
        """Create an appropriate OCR processor based on document type"""
        
        if document_type == 'handwritten':
            return HandwrittenMathOCR()
        elif document_type == 'printed':
            return PrintedMathOCR()
        elif document_type == 'mixed':
            return HybridMathOCR()
        else:
            # Default to advanced OCR
            return AdvancedMathOCR()

# Specialized processors
class HandwrittenMathOCR(AdvancedMathOCR):
    """Specialized OCR for handwritten math"""
    
    def __init__(self):
        super().__init__()
        # Additional preprocessing for handwritten math
        self.config = r'--psm 8 --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{}^._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'
    
    def preprocess_math_image(self, image):
        """Enhanced preprocessing for handwritten math"""
        processed = super().preprocess_math_image(image)
        
        # Additional processing specific to handwriting
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Apply special normalization for handwriting
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
        return processed

class PrintedMathOCR(AdvancedMathOCR):
    """Specialized OCR for printed math"""
    
    def __init__(self):
        super().__init__()
        # Configuration optimized for printed text
        self.config = r'--psm 6 --oem 3 -c tessedit_char_whitelist="0123456789+-*/()=[]{}^._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'

class HybridMathOCR(AdvancedMathOCR):
    """OCR system that combines multiple approaches for mixed documents"""
    
    def __init__(self):
        super().__init__()
        self.handwritten_processor = HandwrittenMathOCR()
        self.printed_processor = PrintedMathOCR()
    
    def extract_math_from_image(self, image, return_format='text'):
        """Try both handwritten and printed approaches and use the best result"""
        # Process with both processors
        handwritten_result = self.handwritten_processor.extract_math_from_image(image, return_format)
        printed_result = self.printed_processor.extract_math_from_image(image, return_format)
        
        # Choose the better result (heuristic: longer and more structured is usually better)
        handwritten_score = len(handwritten_result) + 5 * sum(c in handwritten_result for c in "+-*/=^()")
        printed_score = len(printed_result) + 5 * sum(c in printed_result for c in "+-*/=^()")
        
        if handwritten_score > printed_score:
            return handwritten_result
        else:
            return printed_result
