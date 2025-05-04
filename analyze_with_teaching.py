#!/usr/bin/env python3
"""
Apply Advanced Teaching Perspectives to Real Math Homework

This script analyzes a math homework document and applies the advanced
teaching perspectives to generate rich, teacher-like feedback.
"""

import os
import json
from math_analyzer import MathHomeworkAnalyzer
from math_analyzer.teaching_module import TeachingPerspectives
from math_analyzer.feedback_enhancement import FeedbackEnhancer

def format_feedback(feedback_text):
    """Format feedback text for better readability."""
    separator = "=" * 80
    return f"\n{separator}\n{feedback_text}\n{separator}\n"

def main():
    """Apply advanced teaching perspectives to a real math homework document."""
    print("\nAPPLYING ADVANCED TEACHING PERSPECTIVES TO MATH HOMEWORK\n")
    
    # Initialize components
    analyzer = MathHomeworkAnalyzer()
    teaching = TeachingPerspectives()
    enhancer = FeedbackEnhancer()
    
    # Path to homework image
    image_path = "math_homework.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Homework image not found at {image_path}")
        return
    
    # Analyze the homework
    print(f"Analyzing homework image: {image_path}")
    # Force the document type to be math_exam
    print("Setting document type to math_exam...")
    analyzer.document_classifier.force_document_type = "math_exam"
    results = analyzer.analyze_homework(image_path)
    
    # If results is None, it means something went wrong with the analysis
    if results is None:
        print("ERROR: Analysis failed to return results. Trying fallback approach...")
        # Try direct analysis without document classification
        try:
            # Read the image directly with OpenCV
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                print(f"ERROR: Could not read image from {image_path}")
                return
                
            segments = analyzer.segmenter.segment_image(image)
            questions = []
            for i, region in enumerate(segments):
                text = analyzer.ocr.extract_text(region)
                print(f"Question text: {text}")
                if text:
                    analysis = analyzer.analyzer.analyze_question(text)
                    if analysis:
                        questions.append({
                            'question_number': i + 1,
                            'text': text,
                            'analysis': analysis,
                            'score': analysis.get('score', 0) * 100
                        })
            
            # Create a basic results structure
            results = {
                'final_score': sum(q['score'] for q in questions) / len(questions) if questions else 0,
                'questions': questions
            }
        except Exception as e:
            print(f"Fallback analysis also failed: {str(e)}")
            print("Unable to analyze the document. Please check the image quality and format.")
            return
    
    # Display basic analysis results
    print("\nBASIC ANALYSIS RESULTS:")
    print(f"Final score: {results['final_score']:.1f}%")
    print(f"Questions analyzed: {len(results['questions'])}")
    
    # Enhance with advanced teaching perspectives
    print("\nENHANCING WITH ADVANCED TEACHING PERSPECTIVES...")
    enhanced_results = enhancer.enhance_feedback(results)
    
    # Save enhanced results
    output_path = "results/enhanced_teaching_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # Create a simplified version for JSON serialization
        serializable_results = {
            "final_score": float(enhanced_results['final_score']),
            "total_questions": enhanced_results.get('total_questions', len(enhanced_results['questions'])),
            "questions": []
        }
        
        for question in enhanced_results['questions']:
            q_simple = {
                "question_number": question.get('question_number', 0),
                "text": question.get('text', ''),
                "score": float(question.get('score', 0)),
                "feedback_text": question.get('feedback_text', ''),
                "concept_domain": question.get('feedback', {}).get('concept_domain', ''),
                "conceptual_insight": question.get('feedback', {}).get('conceptual_insight', ''),
                "learning_strategy": question.get('feedback', {}).get('learning_strategy', '')
            }
            serializable_results["questions"].append(q_simple)
            
        json.dump(serializable_results, f, indent=2)
    
    print(f"Enhanced analysis saved to: {output_path}")
    
    # Generate and display teacher-like feedback for each question
    print("\nTEACHER-LIKE FEEDBACK FOR EACH QUESTION:")
    
    for question in enhanced_results['questions']:
        question_text = question.get('text', '')
        analysis = question.get('analysis', {})
        score = question.get('score', 0)
        
        print(f"\nQuestion: {question_text}")
        print(f"Score: {score}%")
        
        # Generate advanced teacher feedback
        if score >= 70:
            is_correct = True
            error_type = None
            error_description = None
        else:
            is_correct = False
            errors = analysis.get('errors', [])
            error_type = errors[0].get('type', 'CONCEPTUAL') if errors else 'CONCEPTUAL'
            error_description = errors[0].get('description', '') if errors else ''
        
        student_answer = analysis.get('student_answer', '')
        correct_answer = analysis.get('correct_answer', '')
        
        # Generate teacher-like feedback
        teacher_feedback = teaching.generate_feedback(
            question_text=question_text,
            student_answer=student_answer,
            correct_answer=correct_answer,
            error_type=error_type,
            error_description=error_description,
            is_correct=is_correct
        )
        
        # Identify mathematical concept domain
        concept_domain = teaching.identify_relevant_mcp(question_text, error_description or "")
        
        # Print the feedback
        print(format_feedback(teacher_feedback))
        print(f"Mathematical concept domain: {concept_domain.replace('_', ' ')}")
        print(f"Teaching style: {teaching.select_teaching_style(error_type or 'CONCEPTUAL')}")
        print("-" * 60)
    
    # Create a comprehensive teaching-enhanced feedback text file
    feedback_path = "results/enhanced_feedback.txt"
    with open(feedback_path, 'w') as f:
        f.write("ADVANCED MATHEMATICAL ANALYSIS AND FEEDBACK\n\n")
        f.write(f"Overall Score: {enhanced_results['final_score']:.1f}%\n\n")
        
        for question in enhanced_results['questions']:
            f.write(f"Question {question.get('question_number', 0)}: {question.get('text', '')}\n")
            f.write(f"Score: {question.get('score', 0)}%\n\n")
            
            # Include enhanced feedback
            if 'feedback_text' in question:
                f.write(f"{question['feedback_text']}\n\n")
            
            # Include concept domain
            concept_domain = question.get('feedback', {}).get('concept_domain', '')
            if concept_domain:
                f.write(f"Mathematical Concept Domain: {concept_domain.replace('_', ' ')}\n")
            
            # Include learning strategy
            strategy = question.get('feedback', {}).get('learning_strategy', '')
            if strategy:
                f.write(f"Learning Strategy: {strategy}\n")
            
            f.write("-" * 60 + "\n\n")
    
    print(f"\nComprehensive enhanced feedback saved to: {feedback_path}")
    print("\nDone! The math homework has been analyzed with advanced teaching perspectives.")

if __name__ == "__main__":
    main()
