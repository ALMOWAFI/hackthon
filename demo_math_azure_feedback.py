"""
Math Feedback System Demo with Azure Integration

This script demonstrates the Math Feedback System using Azure Cognitive Services:
1. Image preprocessing and OCR
2. Math error detection 
3. Feedback generation
4. Result visualization

Usage:
    python demo_math_azure_feedback.py [image_path]
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math_analyzer.improved_error_localization import MathErrorDetector
from math_analyzer.azure_integration import AzureMathAnalyzer
import json

def preprocess_image(image_path):
    """Preprocess math image for better analysis"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
        
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect components
    dilation = cv2.dilate(opening, kernel, iterations=1)
    
    # Invert back for visualization
    result = cv2.bitwise_not(dilation)
    
    return {"original": image, "preprocessed": result}

def get_sample_data(image_path):
    """Get sample data for the given image - this simulates OCR"""
    if "math7" in image_path:
        student_work = """x² + y² = r²
1 + 1 = 3
1 - 1 = 4
1/0 = 0
x^2 * x^3 = x^4
(x^2)^2 = x^3"""
        correct_solution = """x² + y² = r²
1 + 1 = 2
1 - 1 = 0
1/0 = undefined
x^2 * x^3 = x^5
(x^2)^2 = x^4"""
    else:
        # Default sample data
        student_work = "1 + 1 = 3"
        correct_solution = "1 + 1 = 2"
        
    return student_work, correct_solution

def analyze_with_local_system(image_path, images):
    """Analyze with local error detection system"""
    print("\n-- LOCAL SYSTEM ANALYSIS --")
    
    # Get sample data
    student_work, correct_solution = get_sample_data(image_path)
    
    print("Student Work:")
    print(student_work)
    print("\nCorrect Solution:")
    print(correct_solution)
    
    # Create error detector
    detector = MathErrorDetector()
    
    # Detect errors
    print("\nDetecting errors...")
    result = detector.detect_errors(student_work, correct_solution, images["original"])
    
    # Print errors found
    print(f"\nFound {len(result.errors)} errors:")
    for i, error in enumerate(result.errors):
        print(f"Error #{i+1}: Line {error.line_number}, \"{error.error_text}\"")
        print(f"  Type: {error.error_type}")
        print(f"  Correction: {error.correction}")
        print(f"  Explanation: {error.explanation}")
        print()
        
    # Save the results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the marked image if available
    if result.marked_image is not None:
        marked_path = os.path.join(output_dir, f"{base_name}_local_marked.jpg")
        cv2.imwrite(marked_path, result.marked_image)
        print(f"Saved marked image to {marked_path}")
        
    # Return the results for comparison
    return result
    
def analyze_with_azure(image_path):
    """Analyze with Azure Cognitive Services"""
    print("\n-- AZURE API ANALYSIS --")
    
    try:
        # Initialize Azure analyzer
        analyzer = AzureMathAnalyzer()
        
        # Analyze the image
        print(f"Analyzing image with Azure: {image_path}")
        result = analyzer.analyze_and_visualize(image_path)
        
        if "error" in result:
            print(f"Azure analysis error: {result['error']}")
            return None
            
        # Print results summary
        errors = result.get("response", {}).get("analysis", {}).get("errors", [])
        print(f"\nAzure Analysis Results: Found {len(errors)} errors:")
        
        for i, error in enumerate(errors):
            print(f"Error #{i+1}: {error.get('text', 'Unknown')}")
            print(f"  Type: {error.get('type', 'Unknown')}")
            print(f"  Correction: {error.get('correction', 'Unknown')}")
            print(f"  Explanation: {error.get('explanation', 'Unknown')}")
            print()
            
        return result
        
    except Exception as e:
        print(f"Azure analysis error: {str(e)}")
        return None

def generate_pedagogical_feedback(errors, style="direct"):
    """Generate feedback with different teaching styles"""
    if not errors:
        return "Great job! All your answers are correct."
        
    feedback = []
    
    # Process each error
    for i, error in enumerate(errors):
        error_text = error.get('text', '') if isinstance(error, dict) else error.error_text
        error_type = error.get('type', '') if isinstance(error, dict) else error.error_type
        correction = error.get('correction', '') if isinstance(error, dict) else error.correction
        explanation = error.get('explanation', '') if isinstance(error, dict) else error.explanation
        
        if style == "socratic":
            # Socratic style uses questions
            feedback.append(f"Question about \"{error_text}\":")
            
            if "arithmetic" in error_type.lower():
                feedback.append(f"• I see you wrote {error_text}. Can you double-check your calculation?")
            elif "exponent" in error_type.lower():
                feedback.append(f"• With expressions like {error_text}, what's the rule for combining exponents?")
            elif "division" in error_type.lower():
                feedback.append(f"• Think about {error_text}. What happens when we divide by zero?")
            else:
                feedback.append(f"• Take another look at {error_text}. What might need correction here?")
                
            feedback.append("What approach might work better here?")
            
        else:  # Direct instruction style
            # Direct style provides explicit corrections
            feedback.append(f"Problem with \"{error_text}\":")
            feedback.append(f"• {explanation}")
            feedback.append(f"• The correct form is: {correction}")
            
        feedback.append("")  # Add a blank line between problems
    
    # Add overall recommendation
    if style == "socratic":
        feedback.append("What patterns do you notice in these errors? How might you avoid similar mistakes in the future?")
    else:
        feedback.append("Remember to carefully check your calculations and apply mathematical rules correctly.")
    
    return "\n".join(feedback)

def main(image_path="uploads/math7.jpeg"):
    """Main function to run the Azure-integrated demo"""
    print(f"=== MATH FEEDBACK SYSTEM WITH AZURE INTEGRATION ===")
    print(f"Analyzing image: {image_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Preprocess the image
    print("Preprocessing image...")
    images = preprocess_image(image_path)
    if images is None:
        return
        
    # Save preprocessed image
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preproc_path = os.path.join(output_dir, f"{base_name}_preprocessed.jpg")
    cv2.imwrite(preproc_path, images["preprocessed"])
    print(f"Saved preprocessed image to {preproc_path}")
    
    # Analyze with local system
    local_result = analyze_with_local_system(image_path, images)
    
    # Analyze with Azure API
    azure_result = analyze_with_azure(image_path)
    
    # Generate feedback for local results
    if local_result and len(local_result.errors) > 0:
        print("\n-- PEDAGOGICAL FEEDBACK (LOCAL ANALYSIS) --")
        
        print("\nSocratic Style Feedback:")
        socratic = generate_pedagogical_feedback(local_result.errors, "socratic")
        print(socratic)
        
        print("\nDirect Instruction Style Feedback:")
        direct = generate_pedagogical_feedback(local_result.errors, "direct")
        print(direct)
        
    # Generate feedback for Azure results
    if azure_result and "response" in azure_result:
        azure_errors = azure_result["response"]["analysis"]["errors"]
        if azure_errors:
            print("\n-- PEDAGOGICAL FEEDBACK (AZURE ANALYSIS) --")
            
            print("\nSocratic Style Feedback:")
            azure_socratic = generate_pedagogical_feedback(azure_errors, "socratic")
            print(azure_socratic)
            
            print("\nDirect Instruction Style Feedback:")
            azure_direct = generate_pedagogical_feedback(azure_errors, "direct")
            print(azure_direct)
    
    # Display comparison if both analyses are available
    if local_result and azure_result and "marked_image_path" in azure_result:
        print("\n-- SYSTEM COMPARISON --")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(images["original"], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"Local Analysis ({len(local_result.errors)} errors)")
        plt.imshow(cv2.cvtColor(local_result.marked_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        azure_marked = cv2.imread(azure_result["marked_image_path"])
        azure_error_count = len(azure_result["response"]["analysis"]["errors"])
        plt.title(f"Azure Analysis ({azure_error_count} errors)")
        plt.imshow(cv2.cvtColor(azure_marked, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Save the comparison
        comparison_path = os.path.join(output_dir, f"{base_name}_system_comparison.jpg")
        plt.savefig(comparison_path)
        print(f"Saved system comparison to {comparison_path}")
        
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    # Use command-line argument as image path if provided
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
