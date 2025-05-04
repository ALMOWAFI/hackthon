from math_analyzer import MathHomeworkAnalyzer
import cv2
import os

def main():
    # Create analyzer instance
    analyzer = MathHomeworkAnalyzer()
    
    # Example image path (replace with your image path)
    image_path = "math_homework.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please provide a valid image path.")
        return
        
    try:
        # Analyze homework
        results = analyzer.analyze_homework(image_path)
        
        # Print results
        print("\nAnalysis Results:")
        print(f"Final Score: {results['final_score']:.1f}%")
        
        for question in results['questions']:
            print(f"\nQuestion {question['question_number']}: {question['text']}")
            print(f"Score: {question['score']}%")
            
            # Show correct/incorrect status
            is_correct = question['score'] > 0
            status = "Correct!" if is_correct else "Incorrect"
            print(f"Status: {status}")
            
            # Show feedback
            if 'feedback_text' in question:
                print(f"Feedback: {question['feedback_text']}")
            
            # For incorrect answers, show the correct answer
            if not is_correct and 'analysis' in question:
                correct = question['analysis'].get('correct_answer', '')
                print(f"Correct answer: {correct}")
                
                # Show errors
                for error in question['analysis'].get('errors', []):
                    print(f"- {error.get('description', '')}")
            
            # Show recommendations
            if question.get('recommendations'):
                print("\nRecommendations:")
                for rec in question['recommendations']:
                    if 'suggestion' in rec:
                        print(f"- {rec['suggestion']}")
                        if 'resources' in rec:
                            print(f"  Resources: {', '.join(rec['resources'])}")
                            
        print("\nResults have been saved to the 'output' directory:")
        print("- marked_math_homework.jpg (original with markings)")
        print("- feedback_math_homework.jpg (detailed feedback)")
        print("- feedback.txt")
        print("- analysis_results.json")
        
    except Exception as e:
        print(f"Error analyzing homework: {str(e)}")

if __name__ == "__main__":
    main() 