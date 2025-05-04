import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
from math_analyzer.improved_error_localization import MathErrorDetector, ErrorType

# Enhanced OCR preprocessing function for better math recognition
def preprocess_for_math_ocr(image):
    """Apply specialized preprocessing for handwritten math notation"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding to handle varied lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect components of characters
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    
    # Invert back for better visualization
    result = cv2.bitwise_not(dilation)
    
    return result

# Function to generate Gemini-compatible training data
def generate_training_data(image_path, student_work, correct_solution, error_results):
    """Generate training data in a format suitable for Gemini models"""
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # Convert image to base64 (in a real implementation)
    # Here we'll just save preprocessed version for demonstration
    preprocessed = preprocess_for_math_ocr(image)
    preproc_path = os.path.join("results", f"{os.path.splitext(os.path.basename(image_path))[0]}_preproc.jpg")
    cv2.imwrite(preproc_path, preprocessed)
    
    # Create training data structure
    training_example = {
        "image_file": image_path,
        "preprocessed_image": preproc_path,
        "student_work": student_work,
        "correct_solution": correct_solution,
        "detected_errors": []
    }
    
    # Add error data in Gemini-compatible format
    for error in error_results.errors:
        error_data = {
            "text": error.error_text,
            "type": error.error_type,
            "line_number": error.line_number,
            "correction": error.correction,
            "explanation": error.explanation,
            "bounding_box": {
                "top_left_x": error.top_left_x,
                "top_left_y": error.top_left_y,
                "bottom_right_x": error.bottom_right_x,
                "bottom_right_y": error.bottom_right_y
            }
        }
        training_example["detected_errors"].append(error_data)
    
    return training_example

# Main function to test the enhanced math feedback system
def test_enhanced_feedback(image_path, student_work=None, correct_solution=None):
    """Test the enhanced math feedback system with improved OCR and error detection"""
    print(f"Starting analysis of {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # If no student work text is provided, we would use OCR
    # Here we'll use predefined text since OCR integration requires tesseract installation
    if student_work is None:
        # Sample predefined works from the image
        if "math7" in image_path:
            student_work = """x² + y² = r²
1 + 1 = 3
1 - 1 = 4
1/0 = 0
x^2 * x^3 = x^4
(x^2)^2 = x^3"""
        else:
            print("Please provide student work text for this image")
            return
    
    # If no correct solution is provided, provide it
    if correct_solution is None:
        if "math7" in image_path:
            correct_solution = """x² + y² = r²
1 + 1 = 2
1 - 1 = 0
1/0 = undefined
x^2 * x^3 = x^5
(x^2)^2 = x^4"""
        else:
            print("Please provide correct solution for this image")
            return
    
    print("Student Work:")
    print(student_work)
    print("\nCorrect Solution:")
    print(correct_solution)
    
    # Apply preprocessing for better OCR (visualization purpose)
    preprocessed = preprocess_for_math_ocr(image)
    
    # Create error detector and detect errors
    detector = MathErrorDetector()
    print("\nDetecting errors...")
    result = detector.detect_errors(student_work, correct_solution, image)
    
    # Display results
    print(f"\nFound {len(result.errors)} errors:")
    for i, error in enumerate(result.errors):
        print(f"Error #{i+1}: Line {error.line_number}, \"{error.error_text}\"")
        print(f"  Type: {error.error_type}")
        print(f"  Correction: {error.correction}")
        print(f"  Explanation: {error.explanation}")
        print()
    
    # Generate different styles of feedback
    print("\n--- Socratic Style Feedback ---")
    socratic_feedback = generate_pedagogical_feedback(student_work, result.errors, "socratic")
    print(socratic_feedback)
    
    print("\n--- Direct Instruction Style Feedback ---")
    direct_feedback = generate_pedagogical_feedback(student_work, result.errors, "direct")
    print(direct_feedback)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Preprocessed for OCR")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    if result.marked_image is not None:
        plt.title(f"Errors Detected ({len(result.errors)})")
        plt.imshow(cv2.cvtColor(result.marked_image, cv2.COLOR_BGR2RGB))
    else:
        plt.title("No Errors Marked")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Save output
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_image = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.jpg")
    plt.savefig(output_image)
    print(f"\nSaved visual analysis to {output_image}")
    
    # Generate training data for Gemini API
    print("\nGenerating Gemini-compatible training data...")
    training_data = generate_training_data(image_path, student_work, correct_solution, result)
    
    # Save training data
    if training_data:
        training_data_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_training.json")
        with open(training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"Saved Gemini training data to {training_data_path}")
    
    return result

def generate_pedagogical_feedback(student_work, errors, style="direct"):
    """Generate pedagogical feedback based on detected errors"""
    if not errors:
        return "Great job! All your answers are correct."
    
    lines = student_work.strip().split('\n')
    
    # Group errors by line
    errors_by_line = {}
    for error in errors:
        line_idx = error.line_number - 1  # Convert to 0-indexed
        if 0 <= line_idx < len(lines):
            if line_idx not in errors_by_line:
                errors_by_line[line_idx] = []
            errors_by_line[line_idx].append(error)
    
    feedback = []
    
    # Generate feedback for each line with errors
    for line_idx, line_errors in sorted(errors_by_line.items()):
        line = lines[line_idx]
        
        if style == "socratic":
            # Socratic style uses questions to guide the student
            feedback.append(f"Question {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• I see you wrote {error.error_text}. Can you double-check your calculation?")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• What happens to the sign when you move a term across the equals sign?")
                elif "exponent" in error.error_type.lower():
                    feedback.append(f"• Look at how you're handling the exponents in {error.error_text}. What's the rule for exponents when multiplying with the same base?")
                elif "distribution" in error.error_type.lower():
                    feedback.append(f"• When distributing in {error.error_text}, what do you need to do to each term inside the parentheses?")
                elif "factoring" in error.error_type.lower():
                    feedback.append(f"• What factors should you look for when factoring {error.error_text}?")
                elif "variable_substitution" in error.error_type.lower():
                    feedback.append(f"• If you substitute the value into {error.error_text}, what result do you get?")
                else:
                    feedback.append(f"• Take another look at {error.error_text}. What might be wrong here?")
                
            feedback.append("What approach might work better here?")
            
        else:  # Direct instruction style
            # Direct style provides explicit corrections
            feedback.append(f"Problem {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• There's a calculation error with {error.error_text}. The correct value is {error.correction}.")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• When moving terms across the equals sign, you need to change the sign. {error.error_text} should be {error.correction}.")
                elif "exponent" in error.error_type.lower():
                    feedback.append(f"• {error.explanation} The correct expression is {error.correction}.")
                elif "distribution" in error.error_type.lower():
                    feedback.append(f"• {error.explanation} The correct distribution is {error.correction}.")
                elif "factoring" in error.error_type.lower():
                    feedback.append(f"• The factoring in {error.error_text} is incorrect. {error.explanation} The correct factorization is {error.correction}.")
                elif "variable_substitution" in error.error_type.lower():
                    feedback.append(f"• {error.explanation} The correct result is {error.correction}.")
                else:
                    feedback.append(f"• {error.explanation} The correct form is {error.correction}.")
                    
        feedback.append("")  # Add a blank line between problems
    
    # Add overall recommendation
    if style == "socratic":
        feedback.append("What patterns do you notice in these errors? How might you avoid similar mistakes in the future?")
    else:
        feedback.append("Remember to carefully check your calculations and apply mathematical rules correctly, especially with operations like exponents, distribution, and factoring.")
    
    return "\n".join(feedback)

# Main execution
if __name__ == "__main__":
    # Run with math7 image
    image_path = "uploads/math7.jpeg"
    
    # Analyze with enhanced error detection
    test_enhanced_feedback(image_path)
