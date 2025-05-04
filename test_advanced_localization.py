"""
Test script for Advanced Math Localization System
Specifically focusing on handling messy handwriting and generating targeted practice sheets
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.abspath('.'))

# Import our modules
from math_analyzer.advanced_localization import AdvancedMathLocalization
from math_analyzer.azure_integration import AzureMathAnalyzer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test Advanced Math Localization system with handwritten images"
    )
    
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the handwritten math homework image")
    
    parser.add_argument("--output_dir", type=str, default="results/advanced",
                        help="Directory to save localization results and practice sheets")
    
    parser.add_argument("--debug", action="store_true",
                        help="Generate debug visualizations")
    
    parser.add_argument("--compare_azure", action="store_true",
                        help="Compare results with Azure OCR")
    
    parser.add_argument("--student_name", type=str, default="Student",
                        help="Student name for personalized practice sheet")
    
    return parser.parse_args()

def compare_with_azure(image_path, output_dir):
    """Run Azure OCR for comparison"""
    print("\n=== Running Azure OCR for comparison ===")
    
    # Initialize Azure analyzer
    azure = AzureMathAnalyzer()
    
    # Process the image
    print(f"Processing image with Azure OCR: {image_path}")
    results = azure.analyze_and_visualize(image_path, output_dir=output_dir)
    
    if "error" in results:
        print(f"Azure error: {results['error']}")
        return None
    
    print(f"Azure OCR results saved to: {output_dir}")
    return results

def visualize_localization_results(image_path, analysis, output_dir):
    """Create visualization of localization results"""
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Add a title
    title = "Advanced Math Localization Results"
    cv2.putText(vis_image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, (0, 0, 255), 2)
    
    # Draw detected equations with relevant information
    for eq in analysis.get("equations", []):
        # Get bounding box
        bbox = eq.get("bounding_box")
        if not bbox:
            continue
            
        x, y, w, h = bbox
        
        # Determine color based on correctness
        color = (0, 255, 0) if eq.get("is_correct", False) else (0, 0, 255)
        
        # Draw rectangle around equation
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Add equation information
        eq_id = eq.get("id", 0)
        operation = eq.get("operation", "unknown")
        operands = eq.get("operands", [])
        student_result = eq.get("student_result", "?")
        correct_result = eq.get("correct_result", "?")
        
        # Format equation text
        if len(operands) >= 2:
            if operation == "addition":
                eq_text = f"{operands[0]} + {operands[1]} = {student_result}"
                correct_text = f"Should be: {correct_result}"
            elif operation == "subtraction":
                eq_text = f"{operands[0]} - {operands[1]} = {student_result}"
                correct_text = f"Should be: {correct_result}"
            else:
                eq_text = f"Equation #{eq_id}"
                correct_text = f"Detected result: {student_result}"
        else:
            eq_text = f"Equation #{eq_id}"
            correct_text = f"Detected result: {student_result}"
        
        # Add text above bounding box
        cv2.putText(vis_image, eq_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # If incorrect, add correct result
        if not eq.get("is_correct", False):
            cv2.putText(vis_image, correct_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add summary information
    correct_count = sum(1 for eq in analysis.get("equations", []) if eq.get("is_correct", False))
    total_count = len(analysis.get("equations", []))
    
    summary_text = f"Correct: {correct_count}/{total_count}"
    cv2.putText(vis_image, summary_text, (20, image.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"localization_results.jpg")
    cv2.imwrite(output_path, vis_image)
    
    print(f"Visualization saved to: {output_path}")
    return output_path

def create_focused_practice_sheet(analysis, student_name, output_dir):
    """Create a focused practice sheet based on specific errors"""
    print("\n=== Creating Focused Practice Sheet ===")
    
    # Initialize localization system (just for using its practice sheet generator)
    localizer = AdvancedMathLocalization()
    
    # Generate practice sheet
    output_path = os.path.join(output_dir, f"practice_sheet_{student_name.replace(' ', '_')}.jpg")
    sheet_path = localizer.generate_practice_sheet(analysis, output_path)
    
    print(f"Practice sheet saved to: {sheet_path}")
    return sheet_path

def main():
    """Main function to test advanced localization"""
    args = parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n===== Advanced Math Localization System =====")
    print(f"Processing image: {args.image}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize the advanced localization system
    localizer = AdvancedMathLocalization()
    
    # Process the image
    print("\n=== Running Advanced Localization ===")
    analysis = localizer.process_image(args.image, debug=args.debug)
    
    # Print analysis results
    print("\n=== Analysis Results ===")
    print(f"Detected structure:")
    print(f"  Line count: {analysis['structure']['line_count']}")
    print(f"  Equation count: {analysis['structure']['equation_count']}")
    print(f"  Symbol count: {analysis['structure']['symbol_count']}")
    
    print("\nDetected equations:")
    for eq in analysis.get("equations", []):
        eq_id = eq.get("id", 0)
        operation = eq.get("operation", "unknown")
        operands = eq.get("operands", [])
        student_result = eq.get("student_result", "?")
        correct_result = eq.get("correct_result", "?")
        is_correct = eq.get("is_correct", False)
        
        status = "+" if is_correct else "-"
        
        print(f"  {status} Equation #{eq_id}: ", end="")
        
        if len(operands) >= 2:
            if operation == "addition":
                print(f"{operands[0]} + {operands[1]} = {student_result}", end="")
            elif operation == "subtraction":
                print(f"{operands[0]} - {operands[1]} = {student_result}", end="")
            else:
                print(f"Operation: {operation}, Result: {student_result}", end="")
        else:
            print(f"Operation: {operation}, Result: {student_result}", end="")
        
        if not is_correct:
            print(f" (should be {correct_result})")
        else:
            print(" (correct)")
    
    # Compare with Azure if requested
    if args.compare_azure:
        azure_results = compare_with_azure(args.image, args.output_dir)
    
    # Create visualization
    vis_path = visualize_localization_results(args.image, analysis, args.output_dir)
    
    # Generate practice sheet
    practice_path = create_focused_practice_sheet(analysis, args.student_name, args.output_dir)
    
    print("\n===== Testing complete! =====")
    print(f"Results saved to: {args.output_dir}")
    print(f"  Visualization: {os.path.basename(vis_path)}")
    print(f"  Practice sheet: {os.path.basename(practice_path)}")
    
    # Show summary of accuracy
    correct_count = sum(1 for eq in analysis.get("equations", []) if eq.get("is_correct", False))
    total_count = len(analysis.get("equations", []))
    
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\nStudent accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)")
    
if __name__ == "__main__":
    main()
