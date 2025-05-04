#!/usr/bin/env python
"""
Math Homework Feedback System

A comprehensive system for analyzing math homework, providing teacher-like
feedback, and marking errors with professional annotations.

Features:
- Multi-tiered detection system with Azure OCR for superior handwritten math recognition
- Multiple pedagogical feedback styles (Socratic, Direct Instruction, etc.)
- Professional teacher-like annotations for both correct and incorrect work
- Works with any image size, orientation, or paper type
- Detailed error analysis with personalized next steps

Usage:
    python run_math_analyzer.py [options] image_path

Options:
    --azure         Force use of Azure OCR (if configured)
    --detail        Show detailed analysis in output
    --help          Show this help message

Examples:
    python run_math_analyzer.py test_images/math8.jpeg
    python run_math_analyzer.py --azure --detail homework.jpg
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np

# Import our modules
from math_analyzer.detector import MathExpressionDetector
from math_analyzer.feedback_templates import get_error_feedback, get_correct_feedback, get_next_steps
from math_analyzer.azure_ocr import AzureOCR


def setup_azure():
    """Check Azure configuration and provide setup instructions if needed."""
    azure = AzureOCR()
    if not azure.is_available():
        print("\n===== Azure OCR Configuration =====")
        print("Azure OCR is not configured but provides the best results for handwritten math.")
        print(azure.get_setup_instructions())
        print("For now, falling back to other detection methods.")
        print("=====================================\n")
    return azure.is_available()


def analyze_math_image(image_path, force_azure=False):
    """
    Analyze a math image to detect expressions and provide feedback.
    
    Args:
        image_path: Path to the image file
        force_azure: If True, only use Azure OCR and fail if not configured
        
    Returns:
        Analysis results as a dictionary
    """
    # Check if Azure is forced but not available
    if force_azure:
        azure = AzureOCR()
        if not azure.is_available():
            print("Error: Azure OCR requested but not configured")
            print(azure.get_setup_instructions())
            sys.exit(1)
    
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
    
    # Initialize the detector with appropriate settings
    detector = MathExpressionDetector()
    
    # Detect expressions
    detected_boxes = detector.detect(image, image_path)
    
    # Create a clean copy for marking
    marked_image = image.copy()
    
    # Process each detected expression
    for box in detected_boxes:
        x1, y1, x2, y2, expr_text = box
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Skip empty or unknown expressions that couldn't be read
        if expr_text in ["<expr>", "<expr-group>", "<unknown>"]:
            # For now we'll skip these, but you could extract the region and try
            # to recognize the text with a more specialized method
            continue
        
        # Analyze the expression
        result = analyze_expression(expr_text)
        if result:
            # Add visual feedback
            if result["errors"]:
                # Mark incorrect expressions with red boxes - scale thickness with image size
                cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
                
                # Add 'X' mark for incorrect answers - use safer positioning
                x_size = max(10, int(min(img_width, img_height) / 100))
                x_pos = max(0, x1 - x_size)
                y_pos = max(0, y1 - x_size)
                
                # Draw X mark with integer coordinates and safe positions
                cv2.line(marked_image, 
                       (x_pos, y_pos), 
                       (x_pos + x_size, y_pos + x_size), 
                       (0, 0, 255), line_thickness)
                cv2.line(marked_image, 
                       (x_pos, y_pos + x_size), 
                       (x_pos + x_size, y_pos), 
                       (0, 0, 255), line_thickness)
                
                # Add error annotations with consistent size and position
                for i, error in enumerate(result["errors"]):
                    error_text = f"{error['type']}"
                    text_y = max(text_margin + 10, y1 - text_margin)
                    cv2.putText(marked_image, error_text, 
                              (x1, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              (0, 0, 255), line_thickness)
                
                # Store error location for reference
                analysis_results["visual_feedback"]["error_locations"].append({
                    "expression": result["expression"],
                    "box": [x1, y1, x2, y2]
                })
            else:
                # Mark correct expressions with green checkmark - use safer positioning
                check_size = max(5, int(min(img_width, img_height) / 150))
                check_x = max(check_size * 2, x1 - check_size * 2)
                check_y = max(check_size, y1 + check_size)
                
                # Draw circle with integer coordinates
                cv2.circle(marked_image, (check_x, check_y), check_size, (0, 255, 0), line_thickness)
                
                # Draw checkmark with integer coordinates and safe positions
                pt1 = (max(0, check_x - check_size//2), check_y)
                pt2 = (check_x, max(0, check_y + check_size//2))
                pt3 = (min(img_width-1, check_x + check_size), max(0, check_y - check_size//2))
                
                cv2.line(marked_image, pt1, pt2, (0, 255, 0), line_thickness)
                cv2.line(marked_image, pt2, pt3, (0, 255, 0), line_thickness)
            
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
        
    # Special case for division operations detected by line detection
    if expr == "division_operation":
        # We couldn't read the exact expression but detected a division operation
        result = {
            "expression": "Division operation (possible division by zero)",
            "errors": [{
                "type": "CONCEPTUAL",
                "location": "division operation",
                "severity": "critical",
                "confidence": 0.95
            }],
            "feedback": {},
            "next_steps": {
                "practice_problems": [],
                "concepts_to_review": [],
                "suggested_resources": []
            }
        }
        
        # Generate appropriate feedback for division by zero
        teaching_styles = ["socratic", "direct", "growth_mindset", "constructivist", "inquiry_based"]
        for style in teaching_styles:
            if style == "socratic":
                result["feedback"][style] = "What happens when we divide a number by zero? Why is this operation different from other division problems?"
            elif style == "direct":
                result["feedback"][style] = "Division by zero is undefined in mathematics. This is a critical conceptual error to address."
            elif style == "growth_mindset":
                result["feedback"][style] = "Understanding why division by zero is undefined is an important milestone in developing mathematical reasoning."
            elif style == "constructivist":
                result["feedback"][style] = "Consider what division means: sharing into equal groups. How would you share something into zero groups?"
            else:  # inquiry_based
                result["feedback"][style] = "What would happen if we tried dividing a number by smaller and smaller values? What pattern emerges as we approach zero?"
        
        # Add next steps for division by zero
        result["next_steps"] = {
            "practice_problems": [
                "Division with small denominators",
                "Identifying undefined operations"
            ],
            "concepts_to_review": [
                "Domain of functions",
                "Limits",
                "Undefined operations"
            ],
            "suggested_resources": [
                "Visual representations of division",
                "Understanding domain restrictions"
            ]
        }
        
        return result
    
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
    
    # Check for division by zero
    if "/" in expr:
        parts = expr.split("/")
        if len(parts) >= 2 and parts[1].strip().startswith("0"):
            result["errors"].append({
                "type": "CONCEPTUAL",
                "location": "division by zero",
                "severity": "critical",
                "confidence": 0.98
            })
            
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
    """Main entry point for the math analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze math homework and provide teacher-like feedback'
    )
    parser.add_argument(
        'image_path', 
        help='Path to the image file to analyze', 
        nargs='?'
    )
    parser.add_argument(
        '--azure', 
        action='store_true',
        help='Force use of Azure OCR for best handwritten math recognition'
    )
    parser.add_argument(
        '--detail', 
        action='store_true', 
        help='Show detailed analysis in terminal'
    )
    
    args = parser.parse_args()
    
    # Display welcome message
    print("\n===== Math Homework Feedback System =====")
    print("Analyzing handwritten math and providing teacher-like feedback")
    print("===========================================\n")
    
    # Check Azure configuration
    if args.azure:
        print("Azure OCR requested for optimal handwritten math recognition")
        has_azure = setup_azure()
        if not has_azure:
            print("Error: Azure OCR not configured but requested. Exiting.")
            return 1
    else:
        # Just check and inform, don't force
        setup_azure()
    
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
    output_dir = Path(__file__).parent / "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    print(f"Analyzing math image: {image_path}")
    start_time = time.time()
    
    try:
        result = analyze_math_image(image_path, force_azure=args.azure)
        
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
