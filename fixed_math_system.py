"""
Fixed Math Feedback System with improved integration between OCR and math analysis
"""

import os
import cv2
import numpy as np
import json
import math
import re
from pathlib import Path

from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
from math_analyzer.azure_integration import AzureMathAnalyzer
from math_analyzer.advanced_math_analyzer import AdvancedMathAnalyzer


class MathFeedbackSystem:
    """
    Integrated system for handwritten math recognition and feedback
    """
    
    def __init__(self):
        """
        Initialize the math feedback system
        """
        try:
            # Initialize handwritten math OCR
            self.handwritten_math_ocr = HandwrittenMathOCR()
            print("Handwritten Math OCR initialized")
            
            # Initialize Azure OCR
            self.azure_math_analyzer = AzureMathAnalyzer()
            print("Azure Math Analyzer initialized")
            
            # Initialize advanced math analyzer
            self.advanced_math_analyzer = AdvancedMathAnalyzer()
            print("Advanced Math Analyzer initialized")
            
        except Exception as e:
            print(f"Error initializing Math Feedback System: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize performance metrics
        self.performance_metrics = {
            'ocr_accuracy': [],
            'problem_detection_rate': [],
            'error_detection_rate': []
        }
    
    def process_image(self, image_path, output_dir="results"):
        """
        Process a homework image with handwritten math problems
        
        Args:
            image_path: Path to the homework image
            output_dir: Directory to save output files
            
        Returns:
            Feedback data structure
        """
        print(f"Processing homework image: {image_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Extract math problems using OCR
        problems = self._detect_math_problems(image_path)
        print(f"Detected {len(problems)} math problems in the image")
        
        # Step 2: Analyze each detected problem
        analyzed_problems = self._analyze_math_problems(problems)
        
        # Step 3: Generate comprehensive feedback
        feedback = self._generate_feedback(analyzed_problems, image_path)
        
        # Save feedback to JSON file
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_feedback.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        print(f"Feedback saved to: {output_path}")
        return feedback
    
    def _detect_math_problems(self, image_path):
        """
        Detect math problems in an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of detected math problems
        """
        problems = []
        
        # Try using handwritten math OCR
        hw_problems_count = 0
        try:
            ocr_result = self.handwritten_math_ocr.recognize_expression(image_path)
            if ocr_result and 'expressions' in ocr_result:
                for i, expr in enumerate(ocr_result['expressions']):
                    problem = {
                        'id': f"hw_{i+1}",
                        'text': expr.get('text', ''),
                        'confidence': expr.get('confidence', 0.0),
                        'bounding_box': expr.get('bounding_box', [0, 0, 0, 0]),
                        'source': 'handwritten_ocr'
                    }
                    problems.append(problem)
                    hw_problems_count += 1
                print(f"Handwritten OCR detected {hw_problems_count} problems")
            else:
                print("Handwritten OCR did not detect any expressions")
        except Exception as e:
            print(f"Handwritten OCR failed: {e}")
        
        # Also use Azure OCR - force document type to be math_homework for better results
        azure_problems_count = 0
        try:
            # First try with math_homework document type
            azure_result = self.azure_math_analyzer.analyze_image(image_path, document_type="math_homework")
            
            # If no results, try general OCR
            if not azure_result.get('ocr_result', {}).get('lines'):
                print("Trying with general document type...")
                azure_result = self.azure_math_analyzer.analyze_image(image_path)
                
            if 'ocr_result' in azure_result and 'lines' in azure_result['ocr_result']:
                for i, line in enumerate(azure_result['ocr_result']['lines']):
                    # Use a more permissive check for math content
                    text = line.get('text', '')
                    # Check if this looks like a math problem (more inclusive patterns)
                    if any(op in text for op in ['+', '-', '*', '/', '=', '^', '\u00d7', '\u00f7', '\u221a']) or \
                       any(c.isdigit() for c in text) or \
                       re.search(r'[a-zA-Z]\s*[=<>]\s*\d', text):
                        problem = {
                            'id': f"azure_{i+1}",
                            'text': text,
                            'confidence': line.get('confidence', 0.0),
                            'bounding_box': line.get('bounding_box', [0, 0, 0, 0]),
                            'source': 'azure_ocr'
                        }
                        problems.append(problem)
                        azure_problems_count += 1
                print(f"Azure OCR detected {azure_problems_count} problems")
            else:
                print("Azure OCR did not detect any math problems")
                
            # Save raw OCR result for debugging
            if 'ocr_result' in azure_result:
                output_dir = "results"
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{Path(image_path).stem}_raw_ocr.json"), 'w', encoding='utf-8') as f:
                    json.dump(azure_result['ocr_result'], f, indent=2, ensure_ascii=False)
                print(f"Raw OCR results saved to results/{Path(image_path).stem}_raw_ocr.json")
                    
        except Exception as e:
            print(f"Azure OCR failed: {e}")
        
        # Deduplicate problems (in case both OCR systems detected the same problem)
        unique_problems = []
        seen_texts = set()
        
        for problem in problems:
            # Normalize the text for comparison
            norm_text = problem['text'].strip().lower()
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                unique_problems.append(problem)
        
        return unique_problems
    
    def _analyze_math_problems(self, problems):
        """
        Analyze math problems using the advanced math analyzer
        
        Args:
            problems: List of detected problems
            
        Returns:
            List of analyzed problems
        """
        for problem in problems:
            try:
                # Get the problem text
                text = problem.get('text', '')
                
                # Skip empty text
                if not text.strip():
                    continue
                
                # Normalize the expression
                normalized_text = self.advanced_math_analyzer.normalize_expression(text)
                problem['normalized_text'] = normalized_text
                
                # Analyze the expression
                analysis = self.advanced_math_analyzer.analyze_expression(normalized_text)
                problem['analysis'] = analysis
                
                # Extract problem type
                problem_type = analysis.get('type', 'UNKNOWN')
                problem['math_type'] = problem_type
                
                # Safely print avoiding Unicode encoding errors
                try:
                    print(f"Analyzed problem: '{text}' (type: {problem_type})")
                except UnicodeEncodeError:
                    # Use ASCII-only representation for console output
                    safe_text = text.encode('ascii', 'replace').decode('ascii')
                    print(f"Analyzed problem: '{safe_text}' (type: {problem_type})")
                
            except Exception as e:
                # Safely print error messages
                try:
                    print(f"Error analyzing problem '{text}': {e}")
                except UnicodeEncodeError:
                    safe_text = text.encode('ascii', 'replace').decode('ascii')
                    print(f"Error analyzing problem '{safe_text}': {str(e).encode('ascii', 'replace').decode('ascii')}")
                    
                problem['analysis_error'] = str(e)
        
        return problems
    
    def _generate_feedback(self, problems, image_path):
        """
        Generate comprehensive feedback for math problems
        
        Args:
            problems: List of analyzed problems
            image_path: Path to the original image
            
        Returns:
            Feedback data structure
        """
        feedback = {
            'image_path': image_path,
            'problems': problems,
            'summary': {
                'total_problems': len(problems),
                'errors_found': sum(1 for p in problems if 'analysis' in p and p['analysis'].get('errors', [])),
                'problem_types': {}
            }
        }
        
        # Count problem types
        for problem in problems:
            if 'math_type' in problem:
                math_type = problem['math_type']
                if math_type in feedback['summary']['problem_types']:
                    feedback['summary']['problem_types'][math_type] += 1
                else:
                    feedback['summary']['problem_types'][math_type] = 1
        
        # Generate pedagogical feedback
        feedback['pedagogical_feedback'] = self._generate_pedagogical_feedback(problems)
        
        return feedback
    
    def _generate_pedagogical_feedback(self, problems):
        """
        Generate pedagogical feedback for math problems
        
        Args:
            problems: List of analyzed problems
            
        Returns:
            List of pedagogical feedback items
        """
        pedagogical_feedback = []
        
        for problem in problems:
            if 'analysis' not in problem:
                continue
                
            analysis = problem['analysis']
            
            # Skip problems without errors
            if not analysis.get('errors', []):
                continue
                
            # Generate feedback based on problem type and errors
            feedback_item = {
                'problem_id': problem['id'],
                'problem_text': problem['text'],
                'problem_type': problem.get('math_type', 'UNKNOWN'),
                'errors': analysis.get('errors', []),
                'socratic_approach': [],
                'direct_instruction': [],
                'visualization_suggestion': {}
            }
            
            # Generate socratic questions based on problem type
            if problem.get('math_type') == 'LINEAR_EQUATION':
                feedback_item['socratic_approach'] = [
                    "Have you combined like terms on both sides?",
                    "What happens when you isolate the variable?",
                    "Can you check your solution by substituting it back?"
                ]
                feedback_item['direct_instruction'] = [
                    "Combine like terms on each side of the equation.",
                    "Move all variable terms to one side and constants to the other.",
                    "Divide both sides by the coefficient of the variable."
                ]
                feedback_item['visualization_suggestion'] = {
                    'number_line': "Show how the solution fits on a number line",
                    'balance_scale': "Visualize the equation as a balance scale"
                }
                
            elif problem.get('math_type') == 'QUADRATIC_EQUATION':
                feedback_item['socratic_approach'] = [
                    "What is the discriminant of this equation?",
                    "How many solutions would you expect?",
                    "Can you factor this expression?"
                ]
                feedback_item['direct_instruction'] = [
                    "Rearrange the equation to standard form: ax² + bx + c = 0",
                    "Calculate the discriminant: b² - 4ac",
                    "Use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a"
                ]
                feedback_item['visualization_suggestion'] = {
                    'parabola': "Graph the quadratic function and find where it crosses the x-axis"
                }
                
            elif problem.get('math_type') == 'PYTHAGOREAN':
                feedback_item['socratic_approach'] = [
                    "What does the Pythagorean theorem tell us?",
                    "How do we verify if this is a right triangle?",
                    "Can you identify the hypotenuse?"
                ]
                feedback_item['direct_instruction'] = [
                    "The Pythagorean theorem states: a² + b² = c², where c is the hypotenuse",
                    "Square each side length and verify the equation",
                    "Make sure you're correctly identifying the longest side as the hypotenuse"
                ]
                feedback_item['visualization_suggestion'] = {
                    'right_triangle': "Draw a right triangle with these side lengths"
                }
            
            else:
                # Generic feedback for other problem types
                feedback_item['socratic_approach'] = [
                    "What type of problem is this?",
                    "What steps would you take to solve it?",
                    "Can you verify your solution?"
                ]
                feedback_item['direct_instruction'] = [
                    "Identify the operation being performed",
                    "Follow the steps for that type of operation",
                    "Check your work by verifying your answer"
                ]
            
            pedagogical_feedback.append(feedback_item)
        
        return pedagogical_feedback


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize the system
    math_system = MathFeedbackSystem()
    
    # Check if an image path was provided as a command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"Processing provided image: {image_path}")
            feedback = math_system.process_image(image_path)
        else:
            print(f"Image not found: {image_path}")
            sys.exit(1)
    else:
        # Process a test image if no command-line argument
        image_path = "math_test_sample.jpg"
        if os.path.exists(image_path):
            feedback = math_system.process_image(image_path)
        else:
            print(f"Test image not found: {image_path}")
            # Try to find any test image
            for img_name in ["math_homework.jpg", "math_formulas_test.jpg"]:
                if os.path.exists(img_name):
                    print(f"Using alternative test image: {img_name}")
                    feedback = math_system.process_image(img_name)
                    break
            else:
                print("No test images found. Please add a math image to test.")
                sys.exit(1)
