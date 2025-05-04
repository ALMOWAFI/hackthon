#!/usr/bin/env python3
"""
Demonstrate Advanced Teaching Module (Simple Version)
Avoids special characters that might cause encoding issues.
"""

import os
import json
from math_analyzer.teaching_module import TeachingPerspectives

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(78))
    print("=" * 80 + "\n")

def main():
    """Demonstrate advanced teaching perspectives with simple examples."""
    print("\nADVANCED TEACHING PERSPECTIVES DEMONSTRATION\n")
    
    # Initialize the teaching module
    teaching = TeachingPerspectives()
    
    # Example math problems with incorrect answers
    example_problems = [
        {
            "question": "x^2 + y^2 = r^2",
            "student_answer": "y^2 = r^2 - x^2",
            "correct_answer": "x^2 + y^2 = r^2",
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
            "question": "Find the limit of sin(x)/x as x approaches 0",
            "student_answer": "0/0 = undefined",
            "correct_answer": "1",
            "error_type": "CONCEPTUAL",
            "error_description": "This is an indeterminate form requiring limit evaluation techniques.",
            "is_correct": False
        },
        {
            "question": "Integrate x^2 dx",
            "student_answer": "x^3",
            "correct_answer": "x^3/3 + C",
            "error_type": "PROCEDURAL",
            "error_description": "The power rule for integration requires dividing by the new power and adding a constant.",
            "is_correct": False
        }
    ]
    
    # Display teacher-like feedback for each problem
    print_section("TEACHER-LIKE FEEDBACK FOR MATH PROBLEMS")
    
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
    
    # Get the wildest MCPs
    print_section("THE MOST ADVANCED MATHEMATICAL CONCEPT PRINCIPLES")
    
    wild_mcps = [
        {
            "domain": "ABSTRACT_ALGEBRA",
            "concepts": [
                "Group properties", "Ring structures", "Field extensions",
                "Isomorphism", "Abstract structures"
            ],
            "misconceptions": "Assuming all algebraic structures behave like familiar number systems",
            "remediation": "Use concrete examples of different algebraic structures"
        },
        {
            "domain": "MATHEMATICAL_LOGIC",
            "concepts": [
                "Proof techniques", "Logical quantifiers", "Contraposition",
                "Logical equivalence", "Axiomatic systems"
            ],
            "misconceptions": "Mistaking examples for proof",
            "remediation": "Practice translating between logical forms"
        },
        {
            "domain": "CALCULUS_THINKING",
            "concepts": [
                "Rate of change", "Accumulation", "Limit behavior",
                "Infinite processes", "Approximation techniques"
            ],
            "misconceptions": "Treating derivatives as fractions",
            "remediation": "Connect graphical, numerical, and symbolic representations"
        }
    ]
    
    for i, mcp in enumerate(wild_mcps):
        print(f"{i+1}. {mcp['domain'].replace('_', ' ')}")
        print(f"   Key concepts: {', '.join(mcp['concepts'])}")
        print(f"   Common misconception: {mcp['misconceptions']}")
        print(f"   Teaching approach: {mcp['remediation']}")
        print()
    
    # Save enhanced feedback to a text file
    feedback_path = "results/enhanced_feedback_examples.txt"
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    with open(feedback_path, 'w') as f:
        f.write("ADVANCED MATHEMATICAL TEACHING FEEDBACK EXAMPLES\n\n")
        
        for i, problem in enumerate(example_problems):
            f.write(f"Problem {i+1}: {problem['question']}\n")
            f.write(f"Student answer: {problem['student_answer']}\n")
            f.write(f"Correct answer: {problem['correct_answer']}\n")
            f.write(f"Status: {'Correct' if problem['is_correct'] else 'Incorrect'}\n\n")
            
            # Generate feedback for the file
            feedback = teaching.generate_feedback(
                question_text=problem['question'],
                student_answer=problem['student_answer'],
                correct_answer=problem['correct_answer'],
                error_type=problem['error_type'],
                error_description=problem['error_description'],
                is_correct=problem['is_correct']
            )
            
            f.write("Teacher Feedback:\n")
            f.write(f"{feedback}\n\n")
            
            # Identify concept domain for the file
            concept_domain = teaching.identify_relevant_mcp(
                problem['question'], 
                problem['error_description'] or ""
            )
            
            f.write(f"Mathematical Domain: {concept_domain.replace('_', ' ')}\n")
            f.write(f"Teaching Style: {teaching.select_teaching_style(problem['error_type'] or 'CONCEPTUAL')}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"\nEnhanced feedback examples saved to: {feedback_path}")
    print("\nYou can now see how the system generates teacher-like feedback with advanced mathematical concepts!")

if __name__ == "__main__":
    main()
