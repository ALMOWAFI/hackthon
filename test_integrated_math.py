"""
Integrated test script for the Math Feedback System
This uses the complete MathRecognitionSystem to process images with handwritten math
"""

import os
import cv2
import json
import numpy as np
import sys
from math_analyzer.ocr_integration import MathRecognitionSystem
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
import matplotlib.pyplot as plt
import math

# Fix encoding issues for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def create_test_image(width=1200, height=1000, background_color=(255, 255, 255), create_sample_image=True):
    """
    Create a test image with math problems
    
    Args:
        width: Image width
        height: Image height
        background_color: Background color as (R,G,B)
        create_sample_image: Whether to draw sample math expressions
        
    Returns:
        Image with math problems
    """
    # Create blank image
    image = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
    
    if create_sample_image:
        # Draw some math problems on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        color = (0, 0, 0)  # Black
        
        # Correct problems
        cv2.putText(image, "2 + 2 = 4", (100, 100), font, font_scale, color, thickness)
        cv2.putText(image, "5 - 3 = 2", (100, 200), font, font_scale, color, thickness)
        cv2.putText(image, "3 x 4 = 12", (100, 300), font, font_scale, color, thickness)
        cv2.putText(image, "10 / 2 = 5", (100, 400), font, font_scale, color, thickness)
        cv2.putText(image, "3² + 4² = 5²", (100, 500), font, font_scale, color, thickness)
        cv2.putText(image, "5² + 12² = 13²", (100, 600), font, font_scale, color, thickness)
        
        # Incorrect problems
        cv2.putText(image, "2 + 3 = 6", (600, 100), font, font_scale, color, thickness)
        cv2.putText(image, "7 - 2 = 4", (600, 200), font, font_scale, color, thickness)
        cv2.putText(image, "6 x 3 = 24", (600, 300), font, font_scale, color, thickness)
        cv2.putText(image, "8 / 2 = 3", (600, 400), font, font_scale, color, thickness)
        cv2.putText(image, "6² + 8² = 10²", (600, 500), font, font_scale, color, thickness)
        cv2.putText(image, "x² + y² = z²", (600, 600), font, font_scale, color, thickness)
        
        # Add title
        cv2.putText(image, "Math Homework Test", (width//2 - 200, 50), font, 1.5, (0, 0, 255), 3)
    
    return image

def test_ocr_only(image_path):
    """
    Test the OCR component directly
    
    Args:
        image_path: Path to image file
    
    Returns:
        OCR results
    """
    # Load OCR system
    ocr = HandwrittenMathOCR()
    
    # Recognize expression
    print("Testing specialized OCR:")
    result = ocr.recognize_expression(image_path)
    expression = ocr.combine_symbols(result)
    print(f"Recognized expression: {expression}")
    
    # Get structure type
    structure_type, structure_data = ocr._classify_equation_structure(expression)
    print(f"Structure type: {structure_type}")
    
    # Check if it's an equation
    is_equation = '=' in expression
    print(f"Is equation: {is_equation}")
    
    # Check validity if it's an equation
    is_valid = None
    if is_equation:
        is_valid, validation_message = ocr.validate_equation(expression)
        print(f"Is valid: {is_valid}")
        if not is_valid:
            print(f"Validation message: {validation_message}")
    else:
        print(f"Is valid: {is_valid}")
    
    # Generate visualization
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization image
    image = cv2.imread(image_path)
    vis_image = ocr.create_visualization(image, result)
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_ocr_direct.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"OCR visualization saved to: {vis_path}")
    
    return result

def test_integrated(input_image=None, create_sample_image=False, visualize=True):
    """
    Test the integrated OCR and math analysis system
    
    Args:
        input_image: Path to input image or None to use test image
        create_sample_image: Whether to create a sample test image
        visualize: Whether to show visualizations
    """
    print("\n--- Testing Integrated Math Recognition System ---")
    
    # Initialize the system
    system = MathRecognitionSystem()
    
    # Use provided image, create test image, or use default test image
    if input_image is None and create_sample_image:
        # Create test image with math problems
        image = create_test_image(create_sample_image=True)
        
        # Save the test image
        test_image_path = "math_test_sample.jpg"
        cv2.imwrite(test_image_path, image)
        print(f"Created test image: {test_image_path}")
        
        input_image = test_image_path
    elif input_image is None:
        # Use default test image if exists
        default_test_paths = [
            "math_test_sample.jpg",
            "test_images/math_homework.jpg",
            "test_images/math_sample.jpg"
        ]
        
        for path in default_test_paths:
            if os.path.exists(path):
                input_image = path
                print(f"Using default test image: {input_image}")
                break
        
        if input_image is None:
            print("No test image found. Creating one...")
            image = create_test_image(create_sample_image=True)
            test_image_path = "math_test_sample.jpg"
            cv2.imwrite(test_image_path, image)
            print(f"Created test image: {test_image_path}")
            input_image = test_image_path
    
    # Process the image
    print(f"Processing image: {input_image}")
    results = system.process_homework_image(input_image, create_visualization=visualize)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Print summary of results
    print(f"\nFound {len(results['problems'])} math problems")
    print(f"Detected {results['error_count']} errors")
    
    # Display problems and analysis
    print("\n--- Problems Detected ---")
    for i, problem in enumerate(results['problems']):
        print(f"\nProblem #{problem['id']}:")
        print(f"  Text: {problem['text']}")
        print(f"  Source: {problem.get('source', 'unknown')}")
        print(f"  Confidence: {problem.get('confidence', 0):.2f}")
        
        # Print additional problem info if available
        if "operation" in problem:
            print(f"  Operation: {problem['operation']}")
        
        if "math_concept" in problem:
            print(f"  Math Concept: {problem['math_concept']}")
            
        if "pythagorean_form" in problem and problem["pythagorean_form"]:
            print(f"  Pythagorean Theorem Form Detected")
        
        if "is_valid" in problem:
            valid_str = "Valid" if problem["is_valid"] else "Invalid"
            print(f"  Equation Validity: {valid_str}")
    
    # Print analysis results
    print("\n--- Analysis Results ---")
    for analysis in results['analysis']:
        if analysis.get('is_correct') is False:
            print(f"\nError in Problem #{analysis['id']}:")
            print(f"  Text: {analysis['text']}")
            print(f"  Error Type: {analysis.get('error_type', 'unknown')}")
            print(f"  Expected: {analysis.get('expected_result')}")
            
            # Add specialized feedback for Pythagorean theorem
            if analysis.get('error_type') == 'pythagorean_theorem_error':
                print("  This is an error in applying the Pythagorean theorem (a² + b² = c²)")
                if 'parsed' in analysis and all(k in analysis['parsed'] for k in ['side_a', 'side_b', 'side_c']):
                    a = analysis['parsed']['side_a']
                    b = analysis['parsed']['side_b']
                    c = analysis['parsed']['side_c']
                    print(f"  {a}² + {b}² = {a**2 + b**2}, but {c}² = {c**2}")
                    expected_c = round(math.sqrt(a**2 + b**2), 2)
                    print(f"  The correct value for c should be approximately {expected_c}")
    
    # Show visualization if available and requested
    if visualize and "visualization" in results:
        # Display the visualization
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(results["visualization"], cv2.COLOR_BGR2RGB))
        plt.title("Math Analysis Results")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save the visualization
        vis_path = "math_analysis_results.jpg"
        cv2.imwrite(vis_path, results["visualization"])
        print(f"\nSaved visualization to: {vis_path}")
    
    print("\n--- Test Complete ---")

def test_hybrid_ocr(input_image=None, create_sample_image=False):
    """
    Test the hybrid OCR approach (specialized math OCR + Azure OCR)
    
    Args:
        input_image: Path to input image or None to use test image
        create_sample_image: Whether to create a sample test image
    """
    print("\n--- Testing Hybrid OCR Approach ---")
    
    # Initialize the system
    system = MathRecognitionSystem()
    
    # Use provided image, create test image, or use default test image
    if input_image is None and create_sample_image:
        # Create test image with math problems
        image = create_test_image(create_sample_image=True)
        
        # Save the test image
        test_image_path = "math_test_sample.jpg"
        cv2.imwrite(test_image_path, image)
        print(f"Created test image: {test_image_path}")
        
        input_image = test_image_path
    elif input_image is None:
        # Use default test image if exists
        if os.path.exists("math_test_sample.jpg"):
            input_image = "math_test_sample.jpg"
            print(f"Using default test image: {input_image}")
        else:
            print("No test image found. Creating one...")
            image = create_test_image(create_sample_image=True)
            test_image_path = "math_test_sample.jpg"
            cv2.imwrite(test_image_path, image)
            print(f"Created test image: {test_image_path}")
            input_image = test_image_path
    
    # Load the image
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            print(f"Error: Image file not found: {input_image}")
            return
        image = cv2.imread(input_image)
    else:
        # Assume it's already an image
        image = input_image
    
    if image is None:
        print("Failed to load image")
        return
    
    try:
        # Test specialized OCR
        print("\nTesting Specialized OCR...")
        specialized_results = system._process_with_specialized_ocr(image)
        specialized_problems = specialized_results.get("problems", [])
        print(f"Specialized OCR detected {len(specialized_problems)} problems")
        
        for i, prob in enumerate(specialized_problems):
            print(f"  Problem #{i+1}: {prob.get('text', '')}")
            print(f"    Confidence: {prob.get('confidence', 0):.2f}")
        
        # Test Azure OCR
        print("\nTesting Azure OCR...")
        azure_results = system._process_with_azure_ocr(input_image)
        azure_problems = azure_results.get("problems", [])
        print(f"Azure OCR detected {len(azure_problems)} problems")
        
        for i, prob in enumerate(azure_problems):
            print(f"  Problem #{i+1}: {prob.get('text', '')}")
            print(f"    Confidence: {prob.get('confidence', 0):.2f}")
            if "operation" in prob:
                print(f"    Operation: {prob['operation']}")
            if "math_concept" in prob:
                print(f"    Math Concept: {prob['math_concept']}")
        
        # Test hybrid approach
        print("\nTesting Hybrid OCR Approach...")
        merged_problems = system._merge_problem_results(specialized_problems, azure_problems)
        print(f"Hybrid OCR approach detected {len(merged_problems)} problems")
        
        for i, prob in enumerate(merged_problems):
            print(f"  Problem #{i+1}: {prob.get('text', '')}")
            print(f"    Source: {prob.get('source', 'unknown')}")
            print(f"    Confidence: {prob.get('confidence', 0):.2f}")
            if "operation" in prob:
                print(f"    Operation: {prob['operation']}")
            if "math_concept" in prob:
                print(f"    Math Concept: {prob['math_concept']}")
    
    except Exception as e:
        print(f"Error during hybrid OCR test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Hybrid OCR Test Complete ---")

if __name__ == "__main__":
    # Create a command-line argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Test the Math Recognition System")
    parser.add_argument("-i", "--image", help="Path to input image", default=None)
    parser.add_argument("-c", "--create", help="Create sample test image", action="store_true")
    parser.add_argument("-v", "--visualize", help="Show visualizations", action="store_true")
    parser.add_argument("-o", "--ocr-only", help="Test OCR only", action="store_true")
    parser.add_argument("-y", "--hybrid", help="Test hybrid OCR approach", action="store_true")
    
    args = parser.parse_args()
    
    # Run the selected test
    if args.ocr_only:
        test_ocr_only(args.image, args.create)
    elif args.hybrid:
        test_hybrid_ocr(args.image, args.create)
    else:
        test_integrated(args.image, args.create, args.visualize)
