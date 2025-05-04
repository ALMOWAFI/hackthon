#!/usr/bin/env python3
"""
Test script for enhanced equation localization
Tests the improved bounding box localization on math homework images
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from math_analyzer.advanced_localization import AdvancedMathLocalization

def test_equation_localization(image_path):
    """
    Test the enhanced equation localization on a math homework image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to output visualization
    """
    print(f"Testing enhanced equation localization on: {image_path}")
    
    # Initialize the advanced localization system
    localizer = AdvancedMathLocalization()
    
    # Process the image with debug visualizations enabled
    results = localizer.process_image(image_path, debug=True)
    
    # Print detected equations
    print(f"\nDetected {len(results['equations'])} equations")
    for i, equation in enumerate(results['equations']):
        x, y, w, h = equation['bounding_box']
        refined = equation.get('refined', False)
        print(f"  Equation #{i+1}: at ({x}, {y}) size {w}x{h}, {'refined' if refined else 'standard'} box")
        
        if 'confidence' in equation:
            print(f"    Confidence: {equation['confidence']:.2f}")
            
        # Print operation info if available
        if 'operation' in equation:
            operands = equation.get('operands', [])
            op_type = equation.get('operation', 'unknown')
            result = equation.get('result', '?')
            expected = equation.get('expected_result', '?')
            
            operand_str = ', '.join([str(op) for op in operands])
            print(f"    Operation: {op_type}, Operands: [{operand_str}]")
            print(f"    Result: {result}, Expected: {expected}")
            print(f"    Correct: {result == expected}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "localization_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and display the debug images
    if hasattr(localizer, 'debug_images') and localizer.debug_images:
        # Get the equation visualization image
        if 'equations' in localizer.debug_images:
            equation_vis = localizer.debug_images['equations']
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_equation_localization.jpg")
            cv2.imwrite(output_path, equation_vis)
            
            print(f"\nSaved equation localization visualization to: {output_path}")
            return output_path
    
    return None

def display_results(original_path, result_path):
    """
    Display the original image and localization results side by side
    
    Args:
        original_path: Path to original image
        result_path: Path to result visualization
    """
    if not os.path.exists(result_path):
        print(f"Error: Result image not found: {result_path}")
        return
        
    # Read images
    original = cv2.imread(original_path)
    result = cv2.imread(result_path)
    
    if original is None or result is None:
        print("Error: Could not read images")
        return
    
    # Convert from BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # Create figure with two subplots
    plt.figure(figsize=(14, 7))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Result image
    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb)
    plt.title('Enhanced Equation Localization')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    comparison_path = os.path.join(os.path.dirname(result_path), 
                                  f"{os.path.basename(original_path).split('.')[0]}_comparison.jpg")
    plt.savefig(comparison_path)
    print(f"Saved comparison visualization to: {comparison_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Test on math_homework.jpg
    image_path = "math_homework.jpg"
    
    # Run the test
    result_path = test_equation_localization(image_path)
    
    if result_path:
        # Display the results
        display_results(image_path, result_path)
    else:
        print("Error: No result visualization was generated")
