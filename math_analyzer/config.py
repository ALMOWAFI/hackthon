import os

# OCR Settings
TESSERACT_CONFIG = {
    'lang': 'eng',
    'config': '--psm 6 --oem 3',  # Assume uniform block of text
    'math_config': '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789+-×÷=()[]{}.,/^²³xyzrXYZR '  # Include letters and superscripts
}

# Image Processing
IMAGE_SETTINGS = {
    'preprocess': {
        'resize_width': 2000,
        'threshold': 127,
        'blur_kernel': (3, 3)
    },
    'segmentation': {
        'min_contour_area': 100,  # Reduced for smaller handwritten equations
        'padding': 10,
        'min_line_height': 20,  # Minimum height for a line of text
        'line_spacing': 30   # Expected spacing between lines
    }
}

# Visualization
VISUALIZATION = {
    'colors': {
        'correct': (0, 255, 0),  # Green
        'incorrect': (0, 0, 255),  # Red
        'text': (0, 0, 0),  # Black
        'background': (255, 255, 255)  # White
    },
    'font_scale': 1.0,
    'thickness': 2
}

# Output Settings
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gemini API Settings
GEMINI_API_KEY = 'AIzaSyAh9Oq7kKr3JhKSQdjwNCmUT6WFDudg05k'
GEMINI_MODEL = 'gemini-pro'

# Error Types
ERROR_TYPES = {
    'PROCEDURAL': 'Procedural Error',
    'CONCEPTUAL': 'Conceptual Error',
    'CALCULATION': 'Calculation Error'
} 