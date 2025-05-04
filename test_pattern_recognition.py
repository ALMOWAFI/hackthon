import cv2
import numpy as np
import matplotlib.pyplot as plt
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
from math_analyzer.ocr_integration import MathRecognitionSystem

def create_test_image_with_patterns():
    """
    Create a test image with known math patterns that would typically 
    generate redundant question marks
    
    Returns:
        Image with math patterns
    """
    # Create blank image
    height, width = 600, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Define commonly misrecognized equations
    equations = [
        "6 + 8 = 14",        # Simple addition
        "6² + 8² = 10²",     # Pythagorean theorem (6-8-10 triple)
        "3² + 4² = 5²",      # Pythagorean theorem (3-4-5 triple)
        "15 - 7 = 8",        # Subtraction 
        "5 × 6 = 30",        # Multiplication
        "21 ÷ 3 = 7"         # Division
    ]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 0, 0)  # Black
    thickness = 2
    
    # Add title
    cv2.putText(image, "Testing Pattern Recognition", (150, 50), font, 1.5, (0, 0, 255), 3)
    
    # Draw equations
    y_pos = 120
    for i, eq in enumerate(equations):
        cv2.putText(image, eq, (200, y_pos), font, font_scale, color, thickness)
        y_pos += 70
    
    # Save the image
    output_path = "test_patterns.jpg"
    cv2.imwrite(output_path, image)
    print(f"Created test image at: {output_path}")
    
    return image, output_path

def test_pattern_recognition():
    """
    Test the pattern recognition capabilities for handling uncertain characters
    """
    print("\n=== Testing Pattern Recognition for Uncertain Characters ===\n")
    
    # Create test image
    _, test_image_path = create_test_image_with_patterns()
    
    # Initialize OCR module
    ocr = HandwrittenMathOCR()
    
    # Initialize the full system
    system = MathRecognitionSystem()
    
    # Test cases with simulated uncertain recognition
    test_cases = [
        "6?? + 8?? = 10??",    # Should recognize as 6 + 8 = 14 or 6² + 8² = 10²
        "3?? + 4?? = 5??",     # Should recognize as 3² + 4² = 5²
        "15?? - 7?? = 8??",    # Should recognize as 15 - 7 = 8
        "5?? × 6?? = 30??",    # Should recognize as 5 × 6 = 30
        "21?? ÷ 3?? = 7??"     # Should recognize as 21 ÷ 3 = 7
    ]
    
    # Test direct pattern recognition
    print("Testing direct pattern recognition:")
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case #{i+1}: {test_case}")
        
        # Apply pattern recognition
        cleaned = ocr.clean_recognized_text(test_case)
        print(f"  Cleaned result: {cleaned}")
        
        # Check if a specific common pattern was matched
        pattern_match = ocr._match_common_equation_patterns(test_case)
        if pattern_match:
            print(f"  Pattern matched: {pattern_match}")
        else:
            print("  No specific pattern matched")
    
    # Test with the full system on the test image
    print("\n\nTesting with full system on test image:")
    # Process the image with the system
    results = system.process_homework_image(test_image_path)
    
    # Display the results
    if "problems" in results:
        print(f"\nDetected {len(results['problems'])} problems:")
        for i, problem in enumerate(results["problems"]):
            print(f"\n  Problem #{i+1}: {problem.get('text', '')}")
            print(f"  Source: {problem.get('source', 'unknown')}")
            print(f"  Confidence: {problem.get('confidence', 0):.2f}")
            
            # Print any improvement metrics if available
            if "improvement_level" in problem:
                print(f"  Improvement level: {problem['improvement_level']:.2f}")
            if "raw_text" in problem and problem["raw_text"] != problem["text"]:
                print(f"  Original text: {problem['raw_text']}")
                print(f"  Improved text: {problem['text']}")
    
    # Visualize the results
    if "visualization" in results:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(results["visualization"], cv2.COLOR_BGR2RGB))
        plt.title("Pattern Recognition Results")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Save visualization
        cv2.imwrite("pattern_recognition_results.jpg", results["visualization"])
        print("\nVisualization saved to: pattern_recognition_results.jpg")

def test_specific_pattern(pattern="6?? + 8?? = 10??"):
    """
    Test a specific pattern with detailed diagnostics
    
    Args:
        pattern: The pattern to test
    """
    print(f"\n=== Testing Specific Pattern: '{pattern}' ===\n")
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Step 1: Clean the text
    cleaned = ocr.clean_recognized_text(pattern)
    print(f"1. Clean recognized text: '{cleaned}'")
    
    # Step 2: Try specific pattern matching
    pattern_match = ocr._match_common_equation_patterns(pattern)
    print(f"2. Pattern matching result: '{pattern_match}'")
    
    # Step 3: Try digit prediction if no pattern match
    if not pattern_match and '=' in pattern:
        parts = pattern.split('=')
        left = parts[0].strip()
        right = parts[1].strip()
        
        left_prediction = ocr.predict_missing_digits(left)
        right_prediction = ocr.predict_missing_digits(right)
        
        print(f"3. Left side prediction: '{left}' -> '{left_prediction}'")
        print(f"   Right side prediction: '{right}' -> '{right_prediction}'")
        
        equation = f"{left_prediction} = {right_prediction}"
        print(f"   Combined equation: '{equation}'")
    
    # Step 4: Try equation completion
    completed = ocr.complete_uncertain_equation(pattern)
    print(f"4. Equation completion: '{completed}'")
    
    # Overall result
    final_result = completed if completed != pattern else (pattern_match if pattern_match else cleaned)
    print(f"\nFinal result: '{pattern}' -> '{final_result}'")

def test_specific_pythagorean_patterns():
    """
    Test specifically for Pythagorean theorem patterns with question marks
    """
    print("\n=== Testing Pythagorean Theorem Pattern Recognition ===\n")
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Test cases with different variations of the 6-8-10 Pythagorean triple
    test_cases = [
        "6?? + 8?? = 10??",          # Typical uncertain format 
        "6? + 8? = 10?",            # Single question marks
        "6?? + 8?? = 10??",           # Double question marks
        "6² + 8² = 10?",            # Mix of squared and question mark
        "6? + 8? = ?0",             # Different question mark positions
        "?6 + ?8 = 10",             # Question marks at start
        "6²? + 8²? = 10²?",         # Extra question marks
        "6 ? + 8 ? = 10 ?",         # Spaces between digits and question marks
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case #{i+1}: '{test_case}'")
        
        # Apply pattern recognition
        cleaned = ocr.clean_recognized_text(test_case)
        print(f"  Cleaned result: '{cleaned}'")
        
        # Check direct pattern match
        pattern_match = ocr._match_common_equation_patterns(test_case)
        if pattern_match:
            print(f"  Pattern matched: '{pattern_match}'")
        else:
            print("  No specific pattern matched")

if __name__ == "__main__":
    # Test specific Pythagorean theorem patterns first
    test_specific_pythagorean_patterns()
    
    # Test the specific pattern in the image
    test_specific_pattern("6?? + 8?? = 10??")
    
    # Then test general pattern recognition
    # Uncomment to run the full test
    # test_pattern_recognition()
