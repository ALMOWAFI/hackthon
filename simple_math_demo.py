"""
Simplified Math Feedback System Demo
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math_analyzer.improved_error_localization import MathErrorDetector
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def preprocess_image(image_path):
    """Preprocess math image for better analysis"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Save preprocessed image
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_preprocessed.jpg")
    cv2.imwrite(output_path, binary)
    
    print(f"Saved preprocessed image to {output_path}")
    return image, binary

def analyze_with_local_system(image_path, image):
    """Analyze math problems using local error detection system"""
    print("\n\n-- LOCAL SYSTEM ANALYSIS --")
    
    # Sample data (for demonstration)
    student_work = [
        "x² + y² = r²",
        "1 + 1 = 3",
        "1 - 1 = 4",
        "1/0 = 0",
        "x^2 * x^3 = x^4",
        "(x^2)^2 = x^3"
    ]
    
    correct_solutions = [
        "x² + y² = r²",
        "1 + 1 = 2",
        "1 - 1 = 0",
        "1/0 = undefined",
        "x^2 * x^3 = x^5",
        "(x^2)^2 = x^4"
    ]
    
    print("Student Work:")
    for line in student_work:
        print(line)
    
    print("\nCorrect Solution:")
    for line in correct_solutions:
        print(line)
    
    print("\nDetecting errors...")
    print("")
    
    # Initialize detector
    detector = MathErrorDetector()
    
    # Find errors
    errors = []
    for i, (student, correct) in enumerate(zip(student_work, correct_solutions)):
        if student != correct:
            error_type = detector.classify_error(student, correct)
            errors.append({
                "line": i + 1,
                "equation": student,
                "correction": correct,
                "type": error_type,
                "explanation": detector.explain_error(student, correct, error_type)
            })
    
    # Display errors
    print(f"Found {len(errors)} errors:")
    for i, err in enumerate(errors):
        print(f"Error #{i+1}: Line {err['line']}, \"{err['equation']}\"")
        print(f"  Type: {err['type']}")
        print(f"  Correction: {err['correction']}")
        print(f"  Explanation: {err['explanation']}")
    
    # Mark errors on the image
    result_image = image.copy()
    # In a real implementation, we would locate and mark the errors on the image
    # For this demo, we'll just save the original
    
    # Save marked image
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_local_marked.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"Saved marked image to {output_path}")
    
    return errors

def main(image_path=None):
    """Run the math feedback system demo"""
    print("=== MATH FEEDBACK SYSTEM DEMO ===")
    
    # Default test image if none provided
    if not image_path:
        image_path = "uploads/math7.jpeg"
        if not os.path.exists(image_path):
            image_path = "math_test_sample.jpg"
    
    print(f"Analyzing image: {image_path}")
    
    # Step 1: Preprocess the image
    print("Preprocessing image...")
    result = preprocess_image(image_path)
    if result is None:
        return
    
    image, binary = result
    
    # Step 2: Analyze with local system
    errors = analyze_with_local_system(image_path, image)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    # Use command-line argument as image path if provided
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
