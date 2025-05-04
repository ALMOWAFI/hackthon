import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from math_analyzer.advanced_ocr import HybridMathOCR
from math_analyzer.improved_error_localization import MathErrorDetector

def test_error_localization(image_path, correct_solution=None, doc_type="mixed"):
    """
    Test error localization on a given image
    
    Args:
        image_path: Path to the image containing student work
        correct_solution: The correct solution (if known)
        doc_type: Document type for OCR processing ("handwritten", "printed", or "mixed")
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Create OCR system with specified document type
    if doc_type == "handwritten":
        from math_analyzer.advanced_ocr import HandwrittenMathOCR
        ocr = HandwrittenMathOCR()
    elif doc_type == "printed":
        from math_analyzer.advanced_ocr import PrintedMathOCR
        ocr = PrintedMathOCR()
    else:
        from math_analyzer.advanced_ocr import HybridMathOCR
        ocr = HybridMathOCR()
    
    # Extract text from image
    print(f"Extracting text from image using {doc_type} OCR processor...")
    text = ocr.extract_math_from_image(image)
    print(f"Extracted text:\n{text}\n")
    
    # If no correct solution provided, use a placeholder
    if correct_solution is None:
        # Try to infer the correct solution based on the problem
        if "3x + 2 = 8" in text:
            correct_solution = "3x + 2 = 8\n3x = 8 - 2\n3x = 6\nx = 2"
        elif "2(x+3) = 10" in text:
            correct_solution = "2(x+3) = 10\n2x + 6 = 10\n2x = 4\nx = 2"
        elif "x + 5 = 10" in text:
            correct_solution = "x + 5 = 10\nx = 10 - 5\nx = 5"
        elif "x^2 - 4 = 0" in text:
            correct_solution = "x^2 - 4 = 0\n(x+2)(x-2) = 0\nx = -2 or x = 2"
        else:
            # Generic placeholder based on the text extracted
            correct_solution = "Placeholder solution"
            print("Warning: No correct solution provided. Using a placeholder.")
    
    # Create error detector
    detector = MathErrorDetector()
    
    # Detect errors
    print("Detecting errors...")
    result = detector.detect_errors(text, correct_solution, image)
    
    # Print errors found
    print(f"Found {len(result.errors)} errors:")
    for i, error in enumerate(result.errors):
        print(f"Error #{i+1}: Line {error.line_number}, \"{error.error_text}\"")
        print(f"  Type: {error.error_type}")
        print(f"  Correction: {error.correction}")
        print(f"  Explanation: {error.explanation}")
        print(f"  Position: [{error.top_left_x:.2f}, {error.top_left_y:.2f}, {error.bottom_right_x:.2f}, {error.bottom_right_y:.2f}]")
        print()
    
    # Display results
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
            
        # Extract filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_filename}_marked_{doc_type}.jpg")
        
        plt.savefig(output_path)
        print(f"Saved marked image to {output_path}")
        
        # Show the result
        plt.tight_layout()
        plt.show()
    else:
        print("No marked image available.")
    
    return result

if __name__ == "__main__":
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "uploads/math7.jpeg"
    
    # Determine if a document type is specified
    doc_type = "mixed"
    if len(sys.argv) > 2:
        doc_type = sys.argv[2]
    
    # Set correct solution if known (placeholder for now)
    correct_solution = None  # Let the script try to infer it
    
    # Run the test with the specified document type
    print(f"Running error detection on {image_path} using {doc_type} OCR...")
    test_error_localization(image_path, correct_solution, doc_type)
    
    # If no errors were found with the default processor, try the others
    if doc_type == "mixed":
        print("\nTrying handwritten OCR processor...")
        test_error_localization(image_path, correct_solution, "handwritten")
        
        print("\nTrying printed OCR processor...")
        test_error_localization(image_path, correct_solution, "printed")
