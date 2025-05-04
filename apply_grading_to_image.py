#!/usr/bin/env python3
"""
Apply Paper Grading to Math Homework Image

This script applies OCR to extract text from a math homework image,
then uses the paper grading system to provide teacher-like feedback.
"""

import os
import cv2
import numpy as np
import json
import pytesseract
from PIL import Image
from math_analyzer import MathHomeworkAnalyzer
from simple_paper_grading_demo import SimplePaperGrader

def main():
    print("\nAPPLYING PAPER GRADING TO MATH HOMEWORK IMAGE\n")
    
    # Path to the image
    image_path = "math_homework.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Load the image
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Step 2: Extract text using OCR
    print("\nAttempting to extract text using OCR...")
    
    # Convert image to grayscale for better OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply some preprocessing to improve OCR
    # Adaptive thresholding can help with varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Save preprocessed image
    cv2.imwrite("results/preprocessed_math_homework.jpg", thresh)
    print("Saved preprocessed image to results/preprocessed_math_homework.jpg")
    
    # Convert OpenCV image to PIL format for pytesseract
    pil_img = Image.fromarray(thresh)
    
    # Apply OCR
    try:
        extracted_text = pytesseract.image_to_string(pil_img)
        print("\nExtracted text:")
        print("-" * 50)
        print(extracted_text)
        print("-" * 50)
        
        # Save extracted text
        with open("results/extracted_text.txt", "w") as f:
            f.write(extracted_text)
        
        # Check if meaningful text was extracted
        if len(extracted_text.strip()) < 20:
            print("\nWarning: Very little text extracted from the image.")
            should_use_sample = True
        else:
            should_use_sample = False
    except Exception as e:
        print(f"Error during OCR: {str(e)}")
        should_use_sample = True
    
    # Step 3: Apply paper grading
    print("\nApplying paper grading system...")
    
    # Initialize the grader
    grader = SimplePaperGrader(subject_area="STEM", education_level="high_school")
    
    # If OCR failed or returned little text, use a sample math text
    if should_use_sample:
        print("\nUsing sample math text for demonstration purposes since OCR yielded limited results.")
        
        # Sample math homework text
        graded_text = """
Mathematics Homework Assignment

Question 1: Solve the quadratic equation x^2 - 5x + 6 = 0
Answer: I'll use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
Where a=1, b=-5, c=6
x = (5 ± √(25 - 24)) / 2
x = (5 ± √1) / 2
x = (5 ± 1) / 2
x = 3 or x = 2

Question 2: Find the derivative of f(x) = 3x^2 - 2x + 1
Answer: Using the power rule: f'(x) = 6x - 2

Question 3: If a triangle has sides of length 3, 4, and 5, what is its area?
Answer: This is a right triangle (by the Pythagorean theorem).
Area = (1/2) × base × height = (1/2) × 3 × 4 = 6 square units

Question 4: Simplify the expression: (3x^2y)(2xy^3)
Answer: (3x^2y)(2xy^3) = 3 × 2 × x^2 × x × y × y^3 = 6x^3y^4
"""
    else:
        graded_text = extracted_text
    
    # Grade the text
    results = grader.grade_paper(graded_text)
    
    # Save grading results
    with open("results/math_homework_grading.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Display results
    markup = results["markup"]
    assessment = results["assessment"]
    summary = results["summary"]
    
    print("\n" + "="*80)
    print("PAPER GRADING RESULTS FOR MATH HOMEWORK")
    print("="*80)
    
    print(f"\nAnnotations: {len(markup['annotations'])}")
    
    # Display assessment
    print(f"\nFinal Grade: {assessment['letter_grade']} ({assessment['percentage']:.1f}%)")
    print(f"Points: {assessment['total_points']}/{assessment['total_possible']}")
    
    print("\nScores by criterion:")
    for criterion in assessment["criteria_scores"]:
        print(f"  {criterion['criterion_name']}: {criterion['score']}/{criterion['max_points']}")
        print(f"    {criterion['justification']}")
    
    print("\nOverall Assessment:")
    print(summary["overall_assessment"])
    
    print("\nAreas for Improvement:")
    for i, area in enumerate(summary["improvement_areas"]):
        print(f"{i+1}. {area['area']}")
        print(f"   {area['justification']}")
    
    print("\nNext Steps:")
    for i, step in enumerate(summary["next_steps"]):
        print(f"{i+1}. {step['focus_area']}:")
        for suggestion in step["suggestions"]:
            print(f"   • {suggestion}")
    
    print("\nClosing Comment:")
    print(summary["closing_comment"])
    
    # Step 4: Also try with MathHomeworkAnalyzer for comparison
    print("\n" + "="*80)
    print("COMPARISON WITH MATH ANALYZER")
    print("="*80)
    
    print("\nAttempting to analyze with MathHomeworkAnalyzer...")
    try:
        # Initialize the math analyzer
        analyzer = MathHomeworkAnalyzer()
        
        # Force document type to math_exam to ensure processing
        analyzer.document_classifier.force_document_type = "math_exam"
        
        # Analyze the homework
        math_results = analyzer.analyze_homework(image_path)
        
        if math_results:
            print("\nMath Analyzer Results:")
            print(f"Final score: {math_results.get('final_score', 'N/A')}%")
            print(f"Questions analyzed: {len(math_results.get('questions', []))}")
            
            for q_idx, question in enumerate(math_results.get('questions', [])):
                print(f"\nQuestion {q_idx+1}:")
                print(f"  Text: {question.get('text', 'N/A')}")
                print(f"  Score: {question.get('score', 'N/A')}%")
                if 'feedback_text' in question:
                    print(f"  Feedback: {question.get('feedback_text', 'N/A')[:100]}...")
        else:
            print("\nMath Analyzer did not return results for this image.")
    except Exception as e:
        print(f"\nError running MathHomeworkAnalyzer: {str(e)}")
        print("This is expected as we've had issues with OCR on this image previously.")
    
    print("\n" + "="*80)
    print("Analysis complete! All results saved to the 'results' directory.")
    print("="*80)

if __name__ == "__main__":
    main()
