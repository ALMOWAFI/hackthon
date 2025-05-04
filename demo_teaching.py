#!/usr/bin/env python3
"""
Demonstrate Advanced Teaching Module

This script shows how the advanced teaching module can generate
teacher-like feedback for math problems, without relying on the document analyzer.
"""

import os
import json
from math_analyzer.teaching_module import TeachingPerspectives

def format_section(title):
    """Format a section title with decorative borders."""
    separator = "=" * 80
    return f"\n{separator}\n{title.center(78)}\n{separator}\n"

def main():
    """Demonstrate advanced teaching perspectives with example problems."""
    print("\nADVANCED TEACHING PERSPECTIVES DEMONSTRATION\n")
    
    # Initialize the teaching module
    teaching = TeachingPerspectives()
    
    # Example math problems with incorrect answers
    example_problems = [
        {
            "question": "x² + y² = r²",
            "student_answer": "y² = r² - x²",
            "correct_answer": "x² + y² = r²",
            "error_type": None,
            "error_description": None,
            "is_correct": True
        },
        {
            "question": "1 + 1 = ?",
            "student_answer": "3",
            "correct_answer": "2",
            "error_type": "CALCULATION",
            "error_description": "1 + 1 equals 2, not 3",
            "is_correct": False
        },
        {
            "question": "1 - 1 = ?",
            "student_answer": "4",
            "correct_answer": "0",
            "error_type": "CALCULATION",
            "error_description": "1 - 1 equals 0, not 4",
            "is_correct": False
        },
        {
            "question": "1/0 = ?",
            "student_answer": "0",
            "correct_answer": "undefined",
            "error_type": "CONCEPTUAL",
            "error_description": "Division by zero is undefined",
            "is_correct": False
        },
        # Advanced examples
        {
            "question": "lim(x→0) sin(x)/x = ?",
            "student_answer": "0/0 = undefined",
            "correct_answer": "1",
            "error_type": "CONCEPTUAL",
            "error_description": "This is an indeterminate form requiring limit evaluation techniques.",
            "is_correct": False
        },
        {
            "question": "∫ x² dx = ?",
            "student_answer": "x³",
            "correct_answer": "x³/3 + C",
            "error_type": "PROCEDURAL",
            "error_description": "The power rule for integration requires dividing by the new power and adding a constant.",
            "is_correct": False
        }
    ]
    
    # Display teacher-like feedback for each problem
    print(format_section("TEACHER-LIKE FEEDBACK FOR MATH PROBLEMS"))
    
    all_feedback = []
    
    for i, problem in enumerate(example_problems):
        print(f"\nProblem {i+1}: {problem['question']}")
        print(f"Student answer: {problem['student_answer']}")
        print(f"Correct answer: {problem['correct_answer']}")
        print(f"Is correct: {'Yes' if problem['is_correct'] else 'No'}")
        
        # Generate teacher-like feedback
        feedback = teaching.generate_feedback(
            question_text=problem['question'],
            student_answer=problem['student_answer'],
            correct_answer=problem['correct_answer'],
            error_type=problem['error_type'],
            error_description=problem['error_description'],
            is_correct=problem['is_correct']
        )
        
        # Identify mathematical concept domain
        concept_domain = teaching.identify_relevant_mcp(
            problem['question'], 
            problem['error_description'] or ""
        )
        
        # Display the feedback
        print("\nTeacher Feedback:")
        print(f"{feedback}")
        print(f"\nMathematical Domain: {concept_domain.replace('_', ' ')}")
        print(f"Teaching Style: {teaching.select_teaching_style(problem['error_type'] or 'CONCEPTUAL')}")
        print("-" * 80)
        
        # Save all feedback for the output file
        all_feedback.append({
            "problem": problem['question'],
            "student_answer": problem['student_answer'],
            "correct_answer": problem['correct_answer'],
            "is_correct": problem['is_correct'],
            "feedback": feedback,
            "concept_domain": concept_domain,
            "teaching_style": teaching.select_teaching_style(problem['error_type'] or 'CONCEPTUAL')
        })
    
    # Get the wildest MCPs
    print(format_section("THE MOST ADVANCED MATHEMATICAL CONCEPT PRINCIPLES"))
    
    wild_mcps = teaching.get_wildest_mcps()
    for i, mcp in enumerate(wild_mcps):
        print(f"{i+1}. {mcp['domain'].replace('_', ' ')}")
        print(f"   Key concepts: {', '.join(mcp['concepts'])}")
        print(f"   Common misconception: {mcp['misconceptions']}")
        print(f"   Teaching approach: {mcp['remediation']}")
        print()
    
    # Save all examples to a JSON file
    output = {
        "teaching_examples": all_feedback,
        "wildest_mcps": wild_mcps
    }
    
    output_path = "results/teaching_examples.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\nAll teaching examples saved to: {output_path}")
    
    # Create a human-readable feedback file
    feedback_path = "results/enhanced_feedback_examples.txt"
    with open(feedback_path, 'w') as f:
        f.write("ADVANCED MATHEMATICAL TEACHING FEEDBACK EXAMPLES\n\n")
        
        for i, item in enumerate(all_feedback):
            f.write(f"Problem {i+1}: {item['problem']}\n")
            f.write(f"Student answer: {item['student_answer']}\n")
            f.write(f"Correct answer: {item['correct_answer']}\n")
            f.write(f"Status: {'Correct' if item['is_correct'] else 'Incorrect'}\n\n")
            f.write("Teacher Feedback:\n")
            f.write(f"{item['feedback']}\n\n")
            f.write(f"Mathematical Domain: {item['concept_domain'].replace('_', ' ')}\n")
            f.write(f"Teaching Style: {item['teaching_style']}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"Human-readable feedback examples saved to: {feedback_path}")

if __name__ == "__main__":
    main()
