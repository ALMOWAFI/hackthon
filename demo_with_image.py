#!/usr/bin/env python3
"""
Simple demonstration of math analysis with teaching module feedback
using the specific math_homework.jpg image.
"""

import os
import cv2
import numpy as np
from math_analyzer import MathHomeworkAnalyzer
from math_analyzer.teaching_module import TeachingPerspectives
from math_analyzer.feedback_enhancement import FeedbackEnhancer

def main():
    print("\nMATH ANALYZER WITH TEACHING DEMONSTRATION\n")
    
    # Step 1: Display information about the image
    image_path = "math_homework.jpg"
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return
        
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Step 2: Create sample problems based on common math homework
    problems = [
        {
            "text": "2 + 3 = ?",
            "student_answer": "5",
            "correct_answer": "5",
            "is_correct": True,
            "error_type": None,
            "error_description": None
        },
        {
            "text": "7 × 8 = ?",
            "student_answer": "54",
            "correct_answer": "56",
            "is_correct": False,
            "error_type": "COMPUTATIONAL",
            "error_description": "Multiplication error"
        },
        {
            "text": "Solve for x: 2x + 5 = 13",
            "student_answer": "x = 9",
            "correct_answer": "x = 4",
            "is_correct": False,
            "error_type": "ALGEBRAIC",
            "error_description": "Error in solving for variable"
        },
        {
            "text": "Find the area of a circle with radius 3",
            "student_answer": "9*pi",
            "correct_answer": "9*pi",
            "is_correct": True,
            "error_type": None,
            "error_description": None
        }
    ]
    
    # Step 3: Generate teaching-style feedback for each problem
    teaching = TeachingPerspectives()
    
    print("\nGenerated Teacher-Like Feedback for Sample Math Problems:\n")
    print("-" * 80)
    
    for i, problem in enumerate(problems):
        print(f"Problem {i+1}: {problem['text']}")
        print(f"Student answer: {problem['student_answer']}")
        print(f"Correct answer: {problem['correct_answer']}")
        print(f"Is correct: {'Yes' if problem['is_correct'] else 'No'}")
        print()
        
        # Generate rich feedback
        feedback = teaching.generate_feedback(
            question_text=problem['text'],
            student_answer=problem['student_answer'],
            correct_answer=problem['correct_answer'],
            error_type=problem['error_type'],
            error_description=problem['error_description'],
            is_correct=problem['is_correct']
        )
        
        # Get concept domain
        domain = teaching.identify_relevant_mcp(problem['text'], 
                                              problem['error_description'] or "")
        
        # Get teaching style
        style = teaching.select_teaching_style(problem['error_type'] or "GENERAL")
        
        print("Teacher Feedback:")
        print(feedback)
        print()
        print(f"Mathematical Domain: {domain.replace('_', ' ')}")
        print(f"Teaching Style: {style}")
        print("-" * 80)
    
    # Step 4: Attempt basic segmentation on the actual image
    print("\nAttempting basic segmentation on math_homework.jpg...\n")
    
    analyzer = MathHomeworkAnalyzer()
    
    try:
        # Force document type to math_exam
        analyzer.document_classifier.force_document_type = "math_exam"
        
        # Get image and segment it
        segmenter = analyzer.segmenter
        segments = segmenter.segment_image(image)
        
        print(f"Successfully segmented image into {len(segments)} regions")
        
        # Try to extract some text from each segment
        for i, segment in enumerate(segments):
            text = analyzer.ocr.extract_text(segment)
            if text and text.strip():
                print(f"Region {i+1} text: {text.strip()}")
            else:
                print(f"Region {i+1}: No text extracted")
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
    
    print("\nTeaching module demonstration complete!")

if __name__ == "__main__":
    main()
