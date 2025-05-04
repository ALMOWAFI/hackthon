import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import from the math_analyzer module
sys.path.append('.')
from math_analyzer.improved_error_localization import MathErrorDetector

def analyze_math7_for_errors():
    """Analyze math7.jpeg using built-in Math Feedback System components"""
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
    
    # Since we know OCR is challenging, we'll manually provide the text
    # This simulates what a teacher might do when OCR fails
    print("Using manual transcription since OCR for handwritten math is challenging...")
    
    # Based on visual inspection of math7.jpeg
    math7_text = """x² + y² = r²
1 + 1 = 3
1 - 1 = 4
1/0 = 0"""
    
    # Provide the correct solutions
    correct_solutions = """x² + y² = r²
1 + 1 = 2
1 - 1 = 0
1/0 = undefined"""
    
    print(f"Extracted text (manually provided):\n{math7_text}\n")
    print(f"Correct solutions:\n{correct_solutions}\n")
    
    # Create error detector
    detector = MathErrorDetector()
    
    # Detect errors
    print("Detecting errors...")
    result = detector.detect_errors(math7_text, correct_solutions, image)
    
    # Print errors found
    print(f"\nFound {len(result.errors)} errors:")
    for i, error in enumerate(result.errors):
        print(f"Error #{i+1}: Line {error.line_number}, \"{error.error_text}\"")
        print(f"  Type: {error.error_type}")
        print(f"  Correction: {error.correction}")
        print(f"  Explanation: {error.explanation}")
        print()
    
    # Generate feedback using different pedagogical styles
    print("\nGenerating teacher-like feedback with different pedagogical styles:")
    
    # Socratic style (question-based)
    socratic_feedback = generate_feedback(math7_text, result.errors, style="socratic")
    print("\n--- Socratic Style Feedback ---")
    print(socratic_feedback)
    
    # Direct instruction style
    direct_feedback = generate_feedback(math7_text, result.errors, style="direct")
    print("\n--- Direct Instruction Style Feedback ---")
    print(direct_feedback)
    
    # Mark errors on the image
    if result.marked_image is not None:
        # Convert from BGR to RGB for matplotlib
        marked_rgb = cv2.cvtColor(result.marked_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Marked Image ({len(result.errors)} errors)")
        plt.imshow(marked_rgb)
        plt.axis('off')
        
        # Save the output
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, "math7_marked_with_errors.jpg")
        plt.savefig(output_path)
        print(f"\nSaved marked image to {output_path}")
        
        # Show the result
        plt.tight_layout()
        plt.show()
    
    return result

def generate_feedback(student_work, errors, style="direct"):
    """
    Generate pedagogical feedback based on detected errors
    
    Args:
        student_work: Student's work as text
        errors: List of ErrorLocation objects
        style: Pedagogical style ("socratic" or "direct")
    
    Returns:
        Feedback text in the specified style
    """
    if not errors:
        return "Great job! All your answers are correct."
    
    lines = student_work.strip().split('\n')
    
    # Group errors by line
    errors_by_line = {}
    for error in errors:
        line_idx = error.line_number - 1  # Convert to 0-indexed
        if line_idx not in errors_by_line:
            errors_by_line[line_idx] = []
        errors_by_line[line_idx].append(error)
    
    feedback = []
    
    # Generate feedback for each line with errors
    for line_idx, line_errors in sorted(errors_by_line.items()):
        # Skip if line_idx is out of range
        if line_idx < 0 or line_idx >= len(lines):
            continue
            
        line = lines[line_idx]
        
        if style == "socratic":
            # Socratic style uses questions to guide the student
            feedback.append(f"Question {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• I see you wrote {error.error_text}. Can you double-check your calculation?")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• What happens to the sign when you move a term to the other side of the equation?")
                elif "fraction" in error.error_type.lower() or "division" in error.explanation.lower():
                    feedback.append(f"• What happens when we divide by zero? Is there a defined result?")
                else:
                    feedback.append(f"• Take another look at {error.error_text}. What might be wrong here?")
                
            feedback.append("What would be a better approach to solve this?")
            
        else:  # Direct instruction style
            # Direct style provides explicit corrections
            feedback.append(f"Problem {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• There's a calculation error with {error.error_text}. The correct value is {error.correction}.")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• When moving {error.error_text} to the other side of the equation, you need to change the sign. It should be {error.correction}.")
                elif "fraction" in error.error_type.lower() or "division" in error.explanation.lower():
                    feedback.append(f"• Division by zero ({error.error_text}) is undefined, not {error.error_text}.")
                else:
                    feedback.append(f"• {error.explanation} The correct form is {error.correction}.")
                    
        feedback.append("")  # Add a blank line between problems
    
    # Add overall recommendation
    if style == "socratic":
        feedback.append("Think about what these errors have in common. How can you avoid similar mistakes in the future?")
    else:
        feedback.append("Remember to carefully check your calculations and pay attention to the rules of algebra, especially when it comes to signs and division by zero.")
    
    return "\n".join(feedback)

if __name__ == "__main__":
    analyze_math7_for_errors()
