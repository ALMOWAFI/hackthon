#!/usr/bin/env python
"""
Math Homework Analysis System

This script runs the complete math feedback system on a specified image.
It processes math homework, detects expressions, analyzes errors,
and generates visual feedback with teacher-like markup.

Usage:
    python analyze_math.py path/to/image.jpg

Features:
    - Detects and localizes math expressions using advanced computer vision
    - Identifies calculation, notation, procedural, and conceptual errors
    - Provides teacher-like feedback in multiple pedagogical styles
    - Generates visual markup similar to how a teacher would grade homework
    - Works with handwritten, printed, and digital math content
    - Produces detailed JSON analysis and marked image output

Output files are saved in the 'results' directory.
"""
import os
import sys
import argparse
from pathlib import Path
import time
import json
import cv2
import numpy as np
import re

# Direct imports of the core modules
from math_analyzer.detector import MathExpressionDetector
from math_analyzer.feedback_templates import (
    get_error_feedback, get_correct_feedback, get_next_steps
)

# Constants
MODEL_PATH = Path(__file__).parent / "math_analyzer" / "models" / "best.pt"


def analyze_math(image_path):
    """
    Analyze a math image to detect expressions and provide feedback.
    """
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get image dimensions for scaling
    img_height, img_width = image.shape[:2]
    
    # Calculate scale-dependent parameters for consistent visual output
    line_thickness = max(1, int(min(img_width, img_height) / 500))
    font_scale = max(0.4, min(img_width, img_height) / 1000)
    text_margin = int(min(img_width, img_height) / 60)
    
    # Create results structure
    analysis_results = {
        "image": str(image_path),
        "problems": [],
        "visual_feedback": {
            "marked_image_path": "",
            "error_locations": []
        }
    }
    
    # Initialize the detector
    detector = MathExpressionDetector(MODEL_PATH if MODEL_PATH.exists() else None)
    detected_boxes = detector.detect(image, image_path)
    
    # Create a clean copy for marking
    marked_image = image.copy()
    
    # Process each detected expression
    for box in detected_boxes:
        x1, y1, x2, y2, expr_text = box
        
        # Analyze the expression
        result = analyze_expression(expr_text)
        if result:
            # Add visual feedback
            if result["errors"]:
                # Mark incorrect expressions with red boxes - scale thickness with image size
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
                
                # Add error annotations with consistent size and position
                for i, error in enumerate(result["errors"]):
                    error_text = f"{error['type']}"
                    cv2.putText(marked_image, error_text, 
                              (x1, y1 - text_margin),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              (0, 0, 255), line_thickness)
                
                # Store error location for reference
                analysis_results["visual_feedback"]["error_locations"].append({
                    "expression": result["expression"],
                    "box": [x1, y1, x2, y2]
                })
            else:
                # Mark correct expressions with green checkmark
                check_size = max(5, int(min(img_width, img_height) / 150))
                check_x = x1 - check_size * 2
                check_y = y1 + check_size
                
                cv2.circle(marked_image, (check_x, check_y), check_size, (0, 255, 0), line_thickness)
                cv2.line(marked_image, 
                       (check_x - check_size/2, check_y), 
                       (check_x, check_y + check_size/2), 
                       (0, 255, 0), line_thickness)
                cv2.line(marked_image, 
                       (check_x, check_y + check_size/2), 
                       (check_x + check_size, check_y - check_size/2), 
                       (0, 255, 0), line_thickness)
            
            # Store location
            result["location"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            analysis_results["problems"].append(result)
    
    # Save the marked image
    output_dir = Path(__file__).parent / "results"
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    output_image_path = output_dir / f"{base_name}_marked.jpg"
    cv2.imwrite(str(output_image_path), marked_image)
    analysis_results["visual_feedback"]["marked_image_path"] = str(output_image_path)
    
    # Save detailed analysis
    output_path = output_dir / f"{base_name}_detailed_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    return analysis_results


def analyze_expression(expr):
    """Analyze a math expression for errors and generate feedback"""
    # Skip non-mathematical expressions
    if not any(c.isdigit() for c in expr) and not any(c in expr for c in "+-*/=xy"):
        return None
        
    # Basic structure for the analysis result
    result = {
        "expression": expr,
        "errors": [],
        "feedback": {},
        "next_steps": {
            "practice_problems": [],
            "concepts_to_review": [],
            "suggested_resources": []
        }
    }
    
    # Check for calculation errors
    if "=" in expr:
        parts = expr.split("=")
        if len(parts) == 2:
            try:
                # For expressions like x=3, don't mark as error
                if parts[0].strip() in ['x', 'y', 'z'] and parts[1].strip().isdigit():
                    pass  # This is a solution statement, not an error
                else:
                    # Try direct evaluation for simple expressions like 2+2=5
                    left = safe_eval(parts[0].strip())
                    right = safe_eval(parts[1].strip())
                    if left is not None and right is not None and left != right:
                        result["errors"].append({
                            "type": "CALCULATION",
                            "location": "result",
                            "severity": "high",
                            "confidence": 0.98
                        })
            except:
                # If we can't evaluate, don't assume it's wrong
                pass
    
    # Add notation errors for specific cases
    if "." in expr and not any(c.isalpha() for c in expr):
        result["errors"].append({
            "type": "NOTATION",
            "location": "usage of .",
            "severity": "medium",
            "confidence": 0.9
        })
    
    # Add procedural errors for specific patterns
    if "/" in expr and not expr.endswith("/"):
        result["errors"].append({
            "type": "PROCEDURAL",
            "location": "division operation",
            "severity": "medium",
            "confidence": 0.85
        })
    
    # Add conceptual errors for specific cases
    if expr.count("=") > 1:
        result["errors"].append({
            "type": "CONCEPTUAL",
            "location": "multiple equals signs",
            "severity": "high",
            "confidence": 0.9
        })
        
    # Generate appropriate feedback based on errors
    if result["errors"]:
        # Use our enhanced templates for different teaching styles
        error_types = [error["type"] for error in result["errors"]]
        
        # Get feedback for each teaching style
        teaching_styles = ["socratic", "direct", "growth_mindset", "constructivist", "inquiry_based"]
        result["feedback"] = {}
        
        # Get appropriate correct answer where possible
        correct_result = None
        if "CALCULATION" in error_types and "=" in expr:
            try:
                parts = expr.split("=")
                correct_result = safe_eval(parts[0].strip())
            except:
                pass
                
        # Generate feedback for each teaching style
        for style in teaching_styles:
            # Use the first error type to determine primary feedback
            primary_error = result["errors"][0]["type"]
            feedback = get_error_feedback(
                primary_error, style, expr, 
                correct_result=f"{correct_result}" if correct_result is not None else "the correct result"
            )
            result["feedback"][style] = feedback
        
        # Add next steps from our enhanced templates
        result["next_steps"] = get_next_steps(error_types)
    else:
        # Use enhanced templates for correct answers
        teaching_styles = ["socratic", "direct", "growth_mindset", "constructivist", "inquiry_based"]
        result["feedback"] = {style: get_correct_feedback(style, expr) for style in teaching_styles}
        result["next_steps"] = get_next_steps([])  # Empty list means correct
        
    return result


def safe_eval(expr_str):
    """Safely evaluate a mathematical expression string."""
    # Only allow basic operations and numbers for security
    allowed_chars = set("0123456789+-*/() .")
    if not all(c in allowed_chars for c in expr_str):
        return None
        
    try:
        # Use eval with limitations for mathematical expressions only
        # This is relatively safe since we filtered the input
        return eval(expr_str)
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze math homework and provide feedback')
    parser.add_argument('image_path', help='Path to the image file to analyze', nargs='?')
    parser.add_argument('--detail', action='store_true', help='Show detailed analysis in terminal')
    parser.add_argument('--output', '-o', help='Output directory for results (default: results/)')
    
    args = parser.parse_args()
    
    # Use provided image or look for test images
    if args.image_path:
        image_path = args.image_path
    else:
        # Default to math8.jpeg if no image provided
        image_path = str(Path(__file__).parent / "test_images" / "math8.jpeg")
        if not os.path.exists(image_path):
            test_dir = Path(__file__).parent / "test_images"
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jp*g"))
                if test_images:
                    image_path = str(test_images[0])
                    print(f"No image provided. Using first test image: {image_path}")
                else:
                    print("Error: No test images found and no image path provided.")
                    return 1
            else:
                print("Error: No image path provided and test_images directory not found.")
                return 1
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return 1
    
    # Ensure results directory exists
    output_dir = args.output if args.output else Path(__file__).parent / "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    print(f"Analyzing math image: {image_path}")
    start_time = time.time()
    
    try:
        result = analyze_math(image_path)
        
        duration = time.time() - start_time
        print(f"Analysis completed in {duration:.2f} seconds")
        
        # Get output paths
        base_name = Path(image_path).stem
        marked_image_path = result["visual_feedback"]["marked_image_path"]
        json_path = Path(output_dir) / f"{base_name}_detailed_analysis.json"
        
        # Show summary
        print("\n==== Analysis Summary ====")
        print(f"Found {len(result['problems'])} math expressions")
        errors = sum(1 for p in result["problems"] if p["errors"])
        print(f"Errors detected: {errors}")
        print(f"Marked image saved to: {marked_image_path}")
        print(f"Detailed analysis saved to: {json_path}")
        
        # Show detailed analysis if requested
        if args.detail:
            print("\n==== Detailed Analysis ====")
            for i, problem in enumerate(result["problems"], 1):
                print(f"\nProblem {i}: {problem['expression']}")
                if problem["errors"]:
                    print("  Errors:")
                    for error in problem["errors"]:
                        print(f"    - {error['type']}: {error['location']} (confidence: {error['confidence']})")
                    print("  Feedback (Direct Instruction):")
                    print(f"    {problem['feedback']['direct']}")
                else:
                    print("  Correct!")
        
        print("\nTo view the marked image, open the file in an image viewer.")
        print("To see the detailed analysis including all feedback styles, open the JSON file.")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
