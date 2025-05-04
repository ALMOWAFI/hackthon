"""
Unified Math Feedback System Test

This script tests the entire pipeline from image input to math analysis and feedback generation,
with proper handling of handwritten math notation.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from math_analyzer.ocr_integration import MathRecognitionSystem
from math_analyzer.advanced_math_analyzer import AdvancedMathAnalyzer

def test_complete_pipeline(image_path, use_azure=True, use_specialized=True, visualize=True):
    """Test the complete math feedback pipeline"""
    print(f"\n===== TESTING COMPLETE MATH FEEDBACK PIPELINE ON {image_path} =====")
    
    # Step 1: Initialize the math recognition system
    print("Initializing Math Recognition System...")
    math_system = MathRecognitionSystem()
    
    # Step 2: Process the image to extract math problems
    print(f"\nProcessing image: {image_path}")
    feedback = math_system.process_homework_image(image_path)
    
    # Step 3: Display the results
    print("\n===== EXTRACTED PROBLEMS AND FEEDBACK =====")
    
    # Check if we have recognized problems
    problems = feedback.get('problems', [])
    print(f"Detected {len(problems)} math problems")
    
    # Display problems and their analysis
    for i, problem in enumerate(problems):
        print(f"\nProblem #{i+1}: {problem.get('text', 'Unknown')}")
        
        # If the problem has analysis
        if 'analysis' in problem:
            analysis = problem['analysis']
            print(f"  Problem type: {analysis.get('type', 'Unknown')}")
            
            # Show errors
            for error in analysis.get('errors', []):
                print(f"  Error: {error.get('type', 'Unknown')} - {error.get('description', 'No description')}")
            
            # Show feedback
            for fb in analysis.get('feedback', []):
                content = fb.get('content', 'No content')
                print(f"  Feedback ({fb.get('type', 'Unknown')}): {content[:100]}..." if len(content) > 100 else content)
    
    # Display pedagogical feedback if available
    if 'pedagogical_feedback' in feedback and feedback['pedagogical_feedback']:
        print("\n===== PEDAGOGICAL FEEDBACK =====")
        for i, feedback_item in enumerate(feedback['pedagogical_feedback']):
            print(f"\nFeedback Item #{i+1}:")
            
            # Socratic approach
            if 'socratic_approach' in feedback_item:
                print("  Socratic questions to guide thinking:")
                for j, question in enumerate(feedback_item['socratic_approach']):
                    print(f"    {j+1}. {question}")
            
            # Direct instruction
            if 'direct_instruction' in feedback_item:
                print("\n  Direct instruction:")
                for j, instruction in enumerate(feedback_item['direct_instruction']):
                    print(f"    {j+1}. {instruction}")
    
    # Save the results to a JSON file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(image_path).stem}_feedback.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feedback, f, indent=2, ensure_ascii=False)
    
    print(f"\nFeedback saved to: {output_path}")
    
    # If visualization requested, display the image with annotations
    if visualize and 'visualization_path' in feedback:
        print(f"\nVisualization saved to: {feedback['visualization_path']}")
    
    return feedback

def test_direct_expression_analysis(expressions):
    """Test the AdvancedMathAnalyzer directly with expressions"""
    print("\n===== TESTING DIRECT EXPRESSION ANALYSIS =====")
    
    analyzer = AdvancedMathAnalyzer()
    results = {}
    
    for expr in expressions:
        print(f"\nAnalyzing expression: '{expr}'")
        
        # Normalize and analyze
        normalized = analyzer.normalize_expression(expr)
        print(f"Normalized expression: '{normalized}'")
        
        # Run the analysis
        analysis = analyzer.analyze_expression(expr)
        
        # Print the results
        print(f"Expression type: {analysis.get('type', 'Unknown')}")
        
        # Print errors
        for error in analysis.get('errors', []):
            print(f"Error: {error.get('type', 'Unknown')} - {error.get('description', 'No description')}")
        
        # Print feedback
        for feedback in analysis.get('feedback', []):
            content = feedback.get('content', 'No content')
            print(f"Feedback ({feedback.get('type', 'Unknown')}): {content[:100]}..." if len(content) > 100 else content)
        
        # Store results
        results[expr] = analysis
    
    return results

def run_unified_test():
    """Run a unified test of the entire Math Feedback System"""
    print("===== UNIFIED MATH FEEDBACK SYSTEM TEST =====")
    
    # Available test images
    test_images = []
    
    # Look for test images in the main directory
    main_dir = os.getcwd()
    for img_name in ["math_test_sample.jpg", "math_formulas_test.jpg", "math_homework.jpg"]:
        img_path = os.path.join(main_dir, img_name)
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    # Also look in the test_images directory
    test_img_dir = os.path.join(main_dir, "test_images")
    if os.path.exists(test_img_dir):
        for file in os.listdir(test_img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_img_dir, file))
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test expressions for direct analysis
    test_expressions = [
        "1 + 1 = 2",  # Simple correct equation
        "2 + 2 = 5",  # Simple incorrect equation
        "x + 5 = 10",  # Linear equation
        "x^2 + 3x + 2 = 0",  # Quadratic equation
        "a^2 + b^2 = c^2",  # Pythagorean theorem
        "3^2 + 4^2 = 5^2",  # Numerical pythagorean example
    ]
    
    # Part 1: Test direct expression analysis
    expr_results = test_direct_expression_analysis(test_expressions)
    
    # Part 2: Test the complete pipeline with images
    image_results = {}
    for image_path in test_images:
        try:
            print(f"\nTesting with image: {os.path.basename(image_path)}")
            result = test_complete_pipeline(image_path)
            image_results[os.path.basename(image_path)] = result
        except Exception as e:
            print(f"Error testing image {os.path.basename(image_path)}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save unified test results
    unified_results = {
        "expression_analysis": {expr: str(result) for expr, result in expr_results.items()},
        "image_analysis": {img: "Success" for img in image_results}
    }
    
    with open(os.path.join("results", "unified_test_results.json"), 'w', encoding='utf-8') as f:
        json.dump(unified_results, f, indent=2, ensure_ascii=False)
    
    print("\n===== UNIFIED TEST COMPLETE =====")
    print(f"Results saved to {os.path.join('results', 'unified_test_results.json')}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_unified_test()
