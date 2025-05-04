#!/usr/bin/env python3
"""
Explore Advanced Teaching Perspectives and Mathematical Concept Principles

This script demonstrates how to extract and utilize the most advanced teaching perspectives
and mathematical concept principles from the math_analyzer system.
"""

import json
import os
from math_analyzer.feedback_enhancement import FeedbackEnhancer
from math_analyzer.teaching_module import TeachingPerspectives
from math_analyzer import MathHomeworkAnalyzer
import random

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def explore_wildest_mcps():
    """Explore the most advanced Mathematical Concept Principles"""
    print_section("EXPLORING THE WILDEST MATHEMATICAL CONCEPT PRINCIPLES")
    
    # Initialize the teaching module
    teaching = TeachingPerspectives()
    
    # Get the wildest MCPs
    wild_mcps = teaching.get_wildest_mcps()
    
    print("The most advanced and challenging mathematical concept principles are:\n")
    
    for i, mcp in enumerate(wild_mcps):
        print(f"{i+1}. {mcp['domain'].replace('_', ' ')}")
        print(f"   Key concepts: {', '.join(mcp['concepts'])}")
        print(f"   Common misconception: {mcp['misconceptions']}")
        print(f"   Teaching approach: {mcp['remediation']}")
        print()

def demonstrate_teacher_voice():
    """Demonstrate how the system can write feedback in a teacher's voice"""
    print_section("TEACHER VOICE FEEDBACK EXAMPLES")
    
    # Sample questions with incorrect answers
    sample_questions = [
        {
            "text": "x² + y² = r²",
            "student_answer": "y² = r² - x²",
            "correct_answer": "r² = x² + y²",
            "error_type": "REPRESENTATION",
            "error_description": "The equation is rearranged differently, though mathematically equivalent."
        },
        {
            "text": "lim(x→0) sin(x)/x",
            "student_answer": "0",
            "correct_answer": "1",
            "error_type": "CONCEPTUAL",
            "error_description": "The limit sin(x)/x as x approaches 0 is 1, not 0."
        },
        {
            "text": "∫ x²dx",
            "student_answer": "2x",
            "correct_answer": "x³/3 + C",
            "error_type": "PROCEDURAL",
            "error_description": "The power rule for integration is incorrect."
        },
        {
            "text": "P(A∩B) = P(A) × P(B) is true when...",
            "student_answer": "always",
            "correct_answer": "A and B are independent events",
            "error_type": "CONCEPTUAL",
            "error_description": "The formula only applies when events are independent."
        }
    ]
    
    teaching = TeachingPerspectives()
    
    print("Here are examples of how our system generates feedback in a teacher's voice:\n")
    
    for i, question in enumerate(sample_questions):
        print(f"Question {i+1}: {question['text']}")
        print(f"Student answered: {question['student_answer']}")
        print(f"Correct answer: {question['correct_answer']}")
        print("\nTeacher feedback:")
        
        feedback = teaching.generate_feedback(
            question_text=question['text'],
            student_answer=question['student_answer'],
            correct_answer=question['correct_answer'],
            error_type=question['error_type'],
            error_description=question['error_description'],
            is_correct=False
        )
        
        print(f"{feedback}\n")
        print("-" * 80)

def generate_comprehensive_report():
    """Generate a comprehensive report of all teaching perspectives"""
    print_section("GENERATING COMPREHENSIVE TEACHING REPORT")
    
    enhancer = FeedbackEnhancer()
    report_path = 'results/teaching_perspective_report.json'
    
    success = enhancer.generate_teaching_perspective_report(report_path)
    
    if success:
        print(f"Successfully generated teaching perspective report at: {report_path}")
        
        # Display snippets from the report
        with open(report_path, 'r') as f:
            report = json.load(f)
            
        # Show the most advanced teaching styles
        print("\nAdvanced Teaching Styles:")
        for style, details in report['advanced_perspectives']['advanced_styles'].items():
            print(f"- {style}: {details['description']}")
            print(f"  Suitable for: {', '.join(details['suitable_for'])}")
        
        # Show the most advanced pedagogical frameworks
        print("\nAdvanced Pedagogical Frameworks:")
        for framework, details in report['advanced_perspectives']['advanced_frameworks'].items():
            print(f"- {framework}: {details['description']}")
            print(f"  Strategy example: {random.choice(details['strategies'])}")
    else:
        print("Failed to generate teaching perspective report.")

def enhance_sample_feedback():
    """Demonstrate enhancing feedback on sample math problems"""
    print_section("ENHANCING SAMPLE FEEDBACK")
    
    # Create a sample analysis result
    sample_analysis = {
        "final_score": 25.0,
        "total_questions": 4,
        "questions": [
            {
                "question_number": 1,
                "text": "x² + y² = r²",
                "analysis": {
                    "question": "x² + y² = r²",
                    "student_answer": "r²",
                    "correct_answer": "r²",
                    "errors": [],
                    "score": 100
                },
                "score": 100
            },
            {
                "question_number": 2,
                "text": "1 + 1 = 3",
                "analysis": {
                    "question": "1 + 1 = 3",
                    "student_answer": "3",
                    "correct_answer": "2",
                    "errors": [{
                        "type": "CALCULATION",
                        "description": "1 + 1 equals 2, not 3"
                    }],
                    "score": 0
                },
                "score": 0
            }
        ]
    }
    
    # Enhance the feedback
    enhancer = FeedbackEnhancer()
    enhanced_results = enhancer.enhance_feedback(sample_analysis)
    
    print("Original vs Enhanced Feedback:\n")
    
    for i, question in enumerate(enhanced_results['questions']):
        print(f"Question {i+1}: {question['text']}")
        print(f"Score: {question['score']}%")
        print("\nEnhanced Feedback:")
        print(f"{question['feedback_text']}")
        print("\nConceptual Domain:", question.get('feedback', {}).get('concept_domain', '').replace('_', ' '))
        print("\nLearning Strategy:", question.get('feedback', {}).get('learning_strategy', ''))
        print("\n" + "-" * 80)

def apply_to_real_document():
    """Apply the enhanced feedback to a real document analysis"""
    print_section("APPLYING TO REAL DOCUMENT ANALYSIS")
    
    # Initialize analyzer and enhancer
    analyzer = MathHomeworkAnalyzer()
    enhancer = FeedbackEnhancer()
    
    try:
        # Try to analyze a sample document
        image_path = "math_homework.jpg"
        if os.path.exists(image_path):
            print(f"Analyzing document: {image_path}")
            results = analyzer.analyze_homework(image_path)
            
            # Enhance the results
            enhanced_results = enhancer.enhance_feedback(results)
            
            # Save enhanced results
            output_path = 'results/enhanced_analysis_results.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(enhanced_results, f, indent=2)
                
            print(f"Enhanced analysis saved to: {output_path}")
            
            # Display a sample of the enhanced feedback
            if enhanced_results.get('questions'):
                sample_question = enhanced_results['questions'][0]
                print("\nSample enhanced feedback:")
                print(f"Question: {sample_question['text']}")
                print(f"Enhanced feedback: {sample_question.get('feedback_text', '')}")
        else:
            print(f"Sample document not found: {image_path}")
    except Exception as e:
        print(f"Error applying to real document: {str(e)}")

def main():
    """Main function to demonstrate all features"""
    print("\nWELCOME TO THE ADVANCED MATH TEACHING EXPLORATION TOOL")
    print("This tool demonstrates the most advanced teaching perspectives and mathematical concepts.\n")
    
    # Explore wildest MCPs
    explore_wildest_mcps()
    
    # Demonstrate teacher voice
    demonstrate_teacher_voice()
    
    # Generate comprehensive report
    generate_comprehensive_report()
    
    # Enhance sample feedback
    enhance_sample_feedback()
    
    # Apply to real document
    apply_to_real_document()
    
    print_section("CONCLUSION")
    print("""
The advanced teaching module provides rich pedagogical frameworks and mathematical 
concept principles that can dramatically improve the quality of feedback in the 
document analyzer system.

By integrating these advanced perspectives, the system can now:
1. Generate feedback that sounds like an expert math teacher
2. Address deeper conceptual misunderstandings
3. Provide tailored learning strategies
4. Connect specific errors to broader mathematical principles
5. Adapt teaching styles based on student needs and error types

These enhancements transform the document analyzer from a simple grading tool
into a sophisticated teaching assistant that supports deeper mathematical learning.
    """)

if __name__ == "__main__":
    main()
