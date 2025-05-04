import cv2
import numpy as np
import matplotlib.pyplot as plt
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR

def create_test_image_with_formulas(output_path="math_formulas_test.jpg"):
    """
    Create a test image with various mathematical formulas
    """
    # Create blank image
    height, width = 1000, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Define formulas to test
    formulas = [
        # Pythagorean theorem
        "6² + 8² = 10²",
        "3² + 4² = 5²",
        
        # Arithmetic operations
        "24 + 18 = 42",
        "35 - 17 = 18",
        "7 × 8 = 56",
        "56 ÷ 8 = 7",
        
        # Linear equations
        "2x + 3 = 7",
        "x - 5 = 10",
        
        # Quadratic equations
        "x² + 5x + 6 = 0",
        "x² - 4 = 0",
        
        # Algebraic identities
        "(a + b)² = a² + 2ab + b²",
        "a² - b² = (a+b)(a-b)",
        
        # Calculus formulas
        "d/dx(x³) = 3x²",
        "∫x² dx = x³/3 + C",
        
        # Trigonometric identities
        "sin²(x) + cos²(x) = 1",
        "tan(x) = sin(x)/cos(x)"
    ]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (0, 0, 0)  # Black
    thickness = 2
    
    # Add title
    cv2.putText(image, "Math Formula Pattern Recognition Test", (100, 50), font, 1.5, (0, 0, 255), 3)
    
    # Draw formulas
    y_pos = 100
    for i, formula in enumerate(formulas):
        cv2.putText(image, formula, (150, y_pos), font, font_scale, color, thickness)
        y_pos += 50
        
        if i % 4 == 3:  # Add extra space every 4 formulas
            y_pos += 30
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Created test image at: {output_path}")
    
    return image, output_path

def test_math_pattern_recognition():
    """
    Test the expanded pattern recognition for various math formulas
    """
    print("\n=== Testing Math Formula Pattern Recognition ===\n")
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Test cases with question marks simulating uncertainty
    test_cases = [
        # Pythagorean theorem
        "6?? + 8?? = 10??",
        "3?? + 4?? = 5??",
        
        # Arithmetic operations
        "24? + 18? = 42?",
        "35? - 17? = 18?",
        "7? × 8? = 56?",
        "56? ÷ 8? = 7?",
        
        # Linear equations
        "2?x + 3? = 7?",
        "x? - 5? = 10?",
        
        # Quadratic equations
        "x?? + 5?x + 6? = 0",
        "x?? - 4? = 0",
        
        # Algebraic identities
        "(a? + b?)² = a?² + 2?ab + b?²",
        "a?² - b?² = (a?+b?)(a?-b?)",
        
        # Calculus formulas
        "d?/dx?(x?³) = 3?x²",
        "∫x?² dx? = x?³/3 + C?",
        
        # Trigonometric identities
        "sin?²(x) + cos?²(x) = 1?",
        "tan?(x) = sin?(x)/cos?(x)"
    ]
    
    # Test each pattern
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case #{i+1}: '{test_case}'")
        
        # Apply pattern recognition
        cleaned = ocr.clean_recognized_text(test_case)
        print(f"  Cleaned result: '{cleaned}'")
        
        # Direct pattern matching
        pattern_match = ocr._match_common_equation_patterns(test_case)
        if pattern_match:
            print(f"  Pattern matched: '{pattern_match}'")
        else:
            print("  No specific pattern matched")

def test_uncertain_character_scenarios():
    """
    Test different scenarios of uncertain characters in common math problems
    """
    print("\n=== Testing Different Uncertain Character Scenarios ===\n")
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Different variations of uncertainty in the same equation
    variations = [
        # Pythagorean theorem with different question mark placements
        "6² + 8² = ?0²",
        "6² + ?8² = 10²",
        "?6² + 8² = 10²",
        "6? + 8? = 10?",
        "6²? + 8²? = 10²?",
        
        # Linear equation with different uncertainty patterns
        "2x + ? = 7",
        "?x + 3 = 7",
        "2x + 3 = ?",
        "2?x + 3? = 7?",
        
        # Quadratic equation with different uncertainty patterns
        "x² + ?x + 6 = 0",
        "x² + 5x + ? = 0",
        "?² + 5x + 6 = 0",
        "x?? + 5?x + 6? = 0?",
        
        # Mixed operations with various question mark placements
        "sin²(?) + cos²(x) = 1",
        "sin²(x) + cos²(?) = ?",
        "(a + ?)² = a² + 2a? + ?²"
    ]
    
    # Test each variation
    for i, variation in enumerate(variations):
        print(f"\nVariation #{i+1}: '{variation}'")
        
        # Apply pattern recognition
        cleaned = ocr.clean_recognized_text(variation)
        print(f"  Cleaned result: '{cleaned}'")
        
        # Direct pattern matching
        pattern_match = ocr._match_common_equation_patterns(variation)
        if pattern_match:
            print(f"  Pattern matched: '{pattern_match}'")
        else:
            print("  No specific pattern matched")

def test_basic_pattern_recognition():
    """
    Test the pattern recognition for basic math formulas with ASCII-only symbols
    """
    print("\n=== Testing Basic Math Formula Pattern Recognition ===\n")
    
    # Initialize OCR
    ocr = HandwrittenMathOCR()
    
    # Test cases using ASCII-compatible characters
    test_cases = [
        # Pythagorean theorem
        "6?? + 8?? = 10??",
        "3?? + 4?? = 5??",
        
        # Arithmetic operations
        "24? + 18? = 42?",
        "35? - 17? = 18?",
        "7? * 8? = 56?",
        "56? / 8? = 7?",
        
        # Linear equations
        "2?x + 3? = 7?",
        "x? - 5? = 10?",
        
        # Quadratic equations
        "x?? + 5?x + 6? = 0",
        "x?? - 4? = 0",
        
        # Simple algebraic patterns
        "a?^2 - b?^2 = (a?+b?)(a?-b?)",
        "(a? + b?)^2 = a?^2 + 2?ab + b?^2"
    ]
    
    # Test each pattern
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case #{i+1}: '{test_case}'")
        
        # Apply pattern recognition
        cleaned = ocr.clean_recognized_text(test_case)
        print(f"  Cleaned result: '{cleaned}'")
        
        # Direct pattern matching
        pattern_match = ocr._match_common_equation_patterns(test_case)
        if pattern_match:
            print(f"  Pattern matched: '{pattern_match}'")
        else:
            print("  No specific pattern matched")

if __name__ == "__main__":
    # Generate test image with math formulas
    create_test_image_with_formulas()
    
    # Test basic pattern recognition (ASCII-compatible)
    test_basic_pattern_recognition()
    
    # Uncomment to run the full tests
    # test_math_pattern_recognition()
    # test_uncertain_character_scenarios()
