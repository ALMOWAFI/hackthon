#!/usr/bin/env python3
"""
Advanced MCPs and Teacher Voice Demo

This script demonstrates the most advanced Mathematical Concept Principles
and shows how they can be used to generate sophisticated teacher-like feedback.
"""

from math_analyzer.teaching_module import TeachingPerspectives
import json

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    """Demonstrate the most advanced MCPs and teacher voice"""
    print("\nADVANCED MATHEMATICAL CONCEPT PRINCIPLES & TEACHER PERSPECTIVE\n")
    
    # Initialize the teaching module
    teaching = TeachingPerspectives()
    
    # Get the most advanced MCPs
    print_section("THE MOST ADVANCED MATHEMATICAL CONCEPT PRINCIPLES")
    
    wild_mcps = [
        # Abstract Algebra
        {
            "domain": "ABSTRACT_ALGEBRA",
            "concepts": [
                "Group properties", "Ring structures", "Field extensions",
                "Isomorphism", "Abstract structures"
            ],
            "misconceptions": "Assuming all algebraic structures behave like familiar number systems",
            "remediation": "Use concrete examples of different algebraic structures"
        },
        # Mathematical Logic
        {
            "domain": "MATHEMATICAL_LOGIC",
            "concepts": [
                "Proof techniques", "Logical quantifiers", "Contraposition",
                "Logical equivalence", "Axiomatic systems"
            ],
            "misconceptions": "Mistaking examples for proof",
            "remediation": "Practice translating between logical forms"
        },
        # Calculus Thinking
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
    
    # Advanced teaching styles
    print_section("MOST SOPHISTICATED TEACHING APPROACHES")
    
    advanced_styles = [
        {
            "name": "SOCRATIC",
            "description": "Uses questions to guide student discovery",
            "suitable_for": ["conceptual_errors", "logical_errors"],
            "examples": [
                "What might happen if you approached this problem differently?",
                "How does this connect to other principles we've learned?",
                "What pattern do you notice in successful solutions?"
            ]
        },
        {
            "name": "METACOGNITIVE",
            "description": "Focuses on thinking about thinking",
            "suitable_for": ["procedural_errors", "conceptual_errors"],
            "examples": [
                "Let's analyze the strategy you used here.",
                "What monitoring steps could help catch this error?",
                "How would you evaluate your approach to this problem?"
            ]
        }
    ]
    
    for style in advanced_styles:
        print(f"{style['name']}: {style['description']}")
        print(f"Best used for: {', '.join(style['suitable_for'])}")
        print("Example prompts:")
        for example in style['examples']:
            print(f"  - \"{example}\"")
        print()
    
    # Teacher voice examples
    print_section("TEACHER VOICE EXAMPLES")
    
    # Advanced math problems
    examples = [
        {
            "topic": "Abstract Algebra",
            "question": "Prove that Z_4 is not isomorphic to Z_2 × Z_2.",
            "student_answer": "They both have 4 elements, so they're isomorphic.",
            "correct_answer": "Z_4 has an element of order 4, while Z_2 × Z_2 doesn't, so they're not isomorphic.",
            "feedback": """
I see you've noted that both structures have 4 elements, which is a good starting point.

However, isomorphism requires more than just having the same number of elements. Two groups are isomorphic when they have the same algebraic structure - meaning there's a one-to-one mapping that preserves the operation.

The key insight here is to examine the order of elements in each group. In Z_4, the element 1 has order 4 (you need to add it to itself 4 times to get back to 0). But in Z_2 × Z_2, every non-identity element has order 2.

This structural difference means these groups cannot be isomorphic, despite having the same size. When working with abstract structures, always check the properties beyond just cardinality.

Try creating the operation tables for both groups to visualize their different structures.
"""
        },
        {
            "topic": "Calculus",
            "question": "Find the limit of sin(x)/x as x approaches 0.",
            "student_answer": "0/0, which is undefined",
            "correct_answer": "1",
            "feedback": """
You've correctly identified that direct substitution gives us 0/0, which looks like an indeterminate form. This is a good observation!

However, when we encounter indeterminate forms, it's a signal to use more advanced techniques, not to stop our analysis. This particular limit is foundational in calculus.

The limit of sin(x)/x as x approaches 0 is actually 1. We can see this geometrically (comparing the sine function to its argument), through Taylor series expansion, or using L'Hôpital's rule.

A helpful visualization: if you draw a unit circle and compare the length of a small arc (x) with the height of its sine value, you'll notice they become nearly identical as x gets very small.

Try exploring this limit using different methods to deepen your understanding of why it equals 1.
"""
        }
    ]
    
    for example in examples:
        print(f"Topic: {example['topic']}")
        print(f"Question: {example['question']}")
        print(f"Student answer: {example['student_answer']}")
        print(f"Correct answer: {example['correct_answer']}")
        print("\nTeacher feedback:")
        print(example['feedback'])
        print("-" * 80)
    
    # Write to file
    output = {
        "wildest_mcps": wild_mcps,
        "advanced_teaching_styles": advanced_styles,
        "teacher_voice_examples": examples
    }
    
    output_path = 'results/advanced_teaching_content.json'
    
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nAdvanced teaching content written to {output_path}")
    except Exception as e:
        print(f"\nError writing to file: {str(e)}")

if __name__ == "__main__":
    main()
