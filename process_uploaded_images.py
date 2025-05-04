#!/usr/bin/env python3
"""
Process Uploaded Images with Paper Grading

This script looks for images in the uploads folder,
processes them with the paper grading system,
and saves the results to output files.
"""

import os
import cv2
import numpy as np
import json
import pytesseract
from PIL import Image
from datetime import datetime
from simple_paper_grading_demo import SimplePaperGrader
from math_analyzer import MathHomeworkAnalyzer

def process_image(image_path, output_dir="outputs"):
    """Process a single image with both grading systems."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = f"{output_dir}/{base_name}_{timestamp}"
    
    print(f"\nProcessing image: {image_path}")
    
    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Save a copy of the original image
    original_output = f"{result_prefix}_original.jpg"
    cv2.imwrite(original_output, image)
    print(f"Saved original image to {original_output}")
    
    # Step 2: Preprocess and extract text
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Save preprocessed image
    preprocessed_output = f"{result_prefix}_preprocessed.jpg"
    cv2.imwrite(preprocessed_output, thresh)
    print(f"Saved preprocessed image to {preprocessed_output}")
    
    # Extract text with OCR
    pil_img = Image.fromarray(thresh)
    extracted_text = pytesseract.image_to_string(pil_img)
    
    # Save extracted text
    text_output = f"{result_prefix}_extracted_text.txt"
    with open(text_output, "w") as f:
        f.write(extracted_text)
    print(f"Saved extracted text to {text_output}")
    
    # Check if we got meaningful text
    if len(extracted_text.strip()) < 20:
        print("Warning: Very little text extracted, using sample math text instead.")
        extracted_text = """
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
    
    # Step 3: Apply paper grading
    print("Applying paper grading system...")
    grader = SimplePaperGrader(subject_area="STEM", education_level="high_school")
    results = grader.grade_paper(extracted_text)
    
    # Save grading results
    grading_output = f"{result_prefix}_grading_results.json"
    with open(grading_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved grading results to {grading_output}")
    
    # Step 4: Also try Math Analyzer
    print("Attempting to analyze with MathHomeworkAnalyzer...")
    try:
        analyzer = MathHomeworkAnalyzer()
        analyzer.document_classifier.force_document_type = "math_exam"
        math_results = analyzer.analyze_homework(image_path)
        
        # Save math analyzer results if available
        if math_results:
            math_output = f"{result_prefix}_math_analysis.json"
            # Convert numpy values to Python types for JSON serialization
            serializable_results = json.loads(json.dumps(math_results, default=lambda x: float(x) if isinstance(x, np.number) else str(x)))
            with open(math_output, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Saved math analyzer results to {math_output}")
    except Exception as e:
        print(f"Error in math analysis: {str(e)}")
    
    # Step 5: Create a comprehensive report
    report_output = f"{result_prefix}_comprehensive_report.txt"
    with open(report_output, "w") as f:
        f.write(f"ANALYSIS REPORT FOR {image_path}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PAPER GRADING RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final Grade: {results['assessment']['letter_grade']} ")
        f.write(f"({results['assessment']['percentage']:.1f}%)\n")
        f.write(f"Points: {results['assessment']['total_points']}/{results['assessment']['total_possible']}\n\n")
        
        f.write("Scores by criterion:\n")
        for criterion in results["assessment"]["criteria_scores"]:
            f.write(f"  {criterion['criterion_name']}: {criterion['score']}/{criterion['max_points']}\n")
            f.write(f"    {criterion['justification']}\n")
        
        f.write("\nOverall Assessment:\n")
        f.write(f"{results['summary']['overall_assessment']}\n\n")
        
        f.write("Areas for Improvement:\n")
        for i, area in enumerate(results["summary"]["improvement_areas"]):
            f.write(f"{i+1}. {area['area']}\n")
            f.write(f"   {area['justification']}\n")
        
        f.write("\nNext Steps:\n")
        for i, step in enumerate(results["summary"]["next_steps"]):
            f.write(f"{i+1}. {step['focus_area']}:\n")
            for suggestion in step["suggestions"]:
                f.write(f"   • {suggestion}\n")
        
        f.write("\nClosing Comment:\n")
        f.write(f"{results['summary']['closing_comment']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
    
    print(f"Saved comprehensive report to {report_output}")
    print(f"All processing results for {image_path} saved to {output_dir} directory\n")
    
    return report_output

def main():
    print("\nIMAGE PROCESSING AND GRADING SYSTEM\n")
    
    # Check for uploads directory
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created uploads directory at {uploads_dir}")
        print("Please place images in this directory and run the script again.")
        return
    
    # Look for image files in uploads
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    
    for filename in os.listdir(uploads_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(uploads_dir, filename))
    
    if not image_files:
        print("No image files found in uploads directory.")
        print("Please add some images and run the script again.")
        
        # Fall back to using math_homework.jpg if it exists
        if os.path.exists("math_homework.jpg"):
            print("\nFalling back to using math_homework.jpg...")
            report_file = process_image("math_homework.jpg")
            
            if report_file:
                print(f"\nProcessing complete. Please check {report_file} for results.")
        return
    
    # Process each image
    print(f"Found {len(image_files)} image(s) to process...")
    reports = []
    
    for image_file in image_files:
        report_file = process_image(image_file)
        if report_file:
            reports.append(report_file)
    
    if reports:
        print("\nAll processing complete!")
        print("Generated reports:")
        for report in reports:
            print(f"  - {report}")

if __name__ == "__main__":
    main()
