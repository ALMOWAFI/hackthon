"""
Quick test script for the Math Feedback System
This simplified script tests the system with Azure OCR on a homework image
"""

import os
import sys
import json
import cv2
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.abspath('.'))

# Import our modules
from math_analyzer.azure_integration import AzureMathAnalyzer
from math_analyzer.analysis import MathAnalyzer

def save_test_image(source_file, target_dir="test_images"):
    """Save a copy of the test image to our test directory"""
    os.makedirs(target_dir, exist_ok=True)
    
    # Get the base filename
    base_name = os.path.basename(source_file)
    target_file = os.path.join(target_dir, base_name)
    
    # Copy the file
    import shutil
    shutil.copy2(source_file, target_file)
    
    print(f"Saved test image to: {target_file}")
    return target_file

def test_azure_ocr(image_path):
    """Test the Azure OCR integration on a homework image"""
    print(f"\n===== Testing Azure OCR on: {image_path} =====")
    
    # Initialize the analyzer
    try:
        azure = AzureMathAnalyzer()
    except Exception as e:
        print(f"Error initializing Azure analyzer: {str(e)}")
        return None
    
    # Process the image
    print("Analyzing image with Azure OCR...")
    results = azure.analyze_and_visualize(image_path)
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        return None
    
    # Print results summary
    print(f"Analysis complete!")
    print(f"Found {results.get('error_count', 0)} errors in the image")
    
    # Show the response structure
    print("\n===== Azure Response Structure =====")
    for key in results.get("response", {}).keys():
        print(f"- {key}")
    
    # Show analysis results
    analysis = results.get("response", {}).get("analysis", {})
    
    print("\n===== Problems =====")
    for problem in analysis.get("problems", []):
        print(f"Problem: {problem.get('text', '')}")
        
    print("\n===== Errors =====")
    for error in analysis.get("errors", []):
        print(f"Error #{error.get('id', '')}: {error.get('text', '')}")
        print(f"  Correction: {error.get('correction', '')}")
        print(f"  Explanation: {error.get('explanation', '')}")
    
    print(f"\nDetailed results saved to: {results.get('marked_image_path', '')}")
    
    return results

def analyze_problems(image_path):
    """Use the MathAnalyzer to analyze problems in the image"""
    print(f"\n===== Analyzing Math Problems in: {image_path} =====")
    
    # Initialize the analyzer
    analyzer = MathAnalyzer()
    
    # Hard-coded problems from the image for testing
    problems = [
        "2 + 2 = 5",
        "1 + 2 = 5"
    ]
    
    # Analyze each problem
    results = []
    for problem in problems:
        print(f"\nAnalyzing problem: {problem}")
        analysis = analyzer.analyze_question(problem)
        
        if analysis:
            print(f"Question: {analysis.get('question', '')}")
            print(f"Student answer: {analysis.get('student_answer', '')}")
            print(f"Correct answer: {analysis.get('correct_answer', '')}")
            print(f"Score: {analysis.get('score', 0)}")
            
            for error in analysis.get('errors', []):
                print(f"  Error: {error.get('description', '')}")
            
            results.append(analysis)
    
    return results

def main():
    """Main function to test the system"""
    # Ask for the image path
    print("===== Math Feedback System Quick Test =====")
    
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to the homework image: ")
    
    # Validate the image path
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Step 1: Save a copy of the test image
    test_image = save_test_image(image_path)
    
    # Step 2: Test Azure OCR
    azure_results = test_azure_ocr(test_image)
    
    # Step 3: Analyze math problems
    analysis_results = analyze_problems(test_image)
    
    # Step 4: Save combined results
    output_file = "results/combined_analysis.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    combined_results = {
        "azure_results": azure_results,
        "analysis_results": analysis_results
    }
    
    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to: {output_file}")
    print("\n===== Testing complete! =====")

if __name__ == "__main__":
    main()
