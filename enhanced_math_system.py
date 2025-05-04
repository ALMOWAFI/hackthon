"""
Enhanced Math Feedback System with vLLM Integration

This module integrates the math feedback system with vLLM capabilities
for advanced analysis, personalized feedback, and worksheet generation.
"""

import os
import cv2
import numpy as np
import json
import math
import re
from pathlib import Path
from datetime import datetime
import sys

# Ensure hackthon module is in the path for accessing fine-tuned models
hackthon_path = os.path.join(os.path.dirname(__file__), 'hackthon')
if os.path.exists(hackthon_path) and hackthon_path not in sys.path:
    sys.path.append(hackthon_path)

# Import core components
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
from math_analyzer.azure_integration import AzureMathAnalyzer
# Import enhanced components with vLLM
from math_analyzer.enhanced_math_analyzer import EnhancedMathAnalyzer
from math_analyzer.worksheet_generator import WorksheetGenerator


class EnhancedMathFeedbackSystem:
    """
    Integrated system for handwritten math recognition and feedback with vLLM
    """
    
    def __init__(self, vllm_url=None, vllm_key=None):
        """
        Initialize the enhanced math feedback system
        
        Args:
            vllm_url: URL of the vLLM API endpoint
            vllm_key: API key for authentication
        """
        try:
            # Initialize handwritten math OCR
            self.handwritten_math_ocr = HandwrittenMathOCR()
            print("Handwritten Math OCR initialized")
            
            # Initialize Azure OCR
            self.azure_math_analyzer = AzureMathAnalyzer()
            print("Azure Math Analyzer initialized")
            
            # Initialize enhanced math analyzer with vLLM using existing fine-tuned models
            self.math_analyzer = EnhancedMathAnalyzer(vllm_url=vllm_url, vllm_key=vllm_key)
            print("Enhanced Math Analyzer initialized with fine-tuned models from hackthon")
            
            # Initialize worksheet generator
            self.worksheet_generator = WorksheetGenerator(vllm_url=vllm_url, vllm_key=vllm_key)
            print("Worksheet Generator initialized")
            
        except Exception as e:
            print(f"Error initializing Enhanced Math Feedback System: {e}")
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
            Enhanced feedback data structure
        """
        print(f"Processing homework image: {image_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Extract math problems using OCR
        problems = self._detect_math_problems(image_path)
        print(f"Detected {len(problems)} math problems in the image")
        
        # Step 2: Analyze each detected problem with enhanced analyzer
        analyzed_problems = self._analyze_math_problems(problems)
        
        # Step 3: Generate comprehensive feedback
        feedback = self._generate_feedback(analyzed_problems, image_path)
        
        # Step 4: Generate practice worksheet based on feedback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        worksheet_filename = f"{Path(image_path).stem}_worksheet_{timestamp}.json"
        worksheet_path = self.worksheet_generator.generate_worksheet(
            feedback_data=feedback,
            output_dir=output_dir,
            filename=worksheet_filename
        )
        
        # Generate HTML version of the worksheet
        html_worksheet_path = self.worksheet_generator.generate_html_worksheet(
            worksheet_data=worksheet_path,
            output_dir=output_dir
        )
        
        # Add worksheet paths to feedback
        feedback['worksheet'] = {
            'json_path': worksheet_path,
            'html_path': html_worksheet_path
        }
        
        # Save enhanced feedback to JSON file
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_enhanced_feedback.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced feedback saved to: {output_path}")
        print(f"Practice worksheet saved to: {html_worksheet_path}")
        
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
            print(f"Handwritten OCR failed: {str(e)}")
        
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
                    # Check if this looks like a math problem
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
                with open(os.path.join("results", f"{Path(image_path).stem}_raw_ocr.json"), 'w', encoding='utf-8') as f:
                    json.dump(azure_result['ocr_result'], f, indent=2, ensure_ascii=False)
                print(f"Raw OCR results saved to results/{Path(image_path).stem}_raw_ocr.json")
                    
        except Exception as e:
            print(f"Azure OCR failed: {str(e)}")
        
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
        Analyze math problems using the enhanced math analyzer
        
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
                normalized_text = self.math_analyzer.normalize_expression(text)
                problem['normalized_text'] = normalized_text
                
                # Enhanced analysis with vLLM
                analysis = self.math_analyzer.analyze_expression(normalized_text)
                problem['analysis'] = analysis
                
                # Extract problem type
                problem_type = analysis.get('type', 'UNKNOWN')
                problem['math_type'] = problem_type
                
                # Generate custom feedback for each problem using the fine-tuned models from hackthon
                problem['custom_feedback'] = {
                    'detailed': self.math_analyzer.generate_custom_feedback(
                        problem=text,
                        answer=text,  # Using the same text as both problem and answer
                        error_type=analysis.get('errors', [{}])[0].get('type', 'UNKNOWN') if analysis.get('errors') else 'UNKNOWN',
                        style='detailed'
                    ),
                    'encouraging': self.math_analyzer.generate_custom_feedback(
                        problem=text,
                        answer=text,
                        error_type=analysis.get('errors', [{}])[0].get('type', 'UNKNOWN') if analysis.get('errors') else 'UNKNOWN',
                        style='encouraging'
                    ),
                    'historical_mathematician': self.math_analyzer.generate_custom_feedback(
                        problem=text,
                        answer=text,
                        error_type=analysis.get('errors', [{}])[0].get('type', 'UNKNOWN') if analysis.get('errors') else 'UNKNOWN',
                        style='historical_mathematician'
                    )
                }
                
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
            Enhanced feedback data structure
        """
        feedback = {
            'image_path': image_path,
            'problems': problems,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        
        # Generate overall feedback summary for the student
        feedback['overall_feedback'] = self._generate_overall_feedback(problems)
        
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
            
            # Skip problems without errors if you want to focus on errors
            # if not analysis.get('errors', []):
            #     continue
                
            # Generate feedback based on problem type and errors
            feedback_item = {
                'problem_id': problem['id'],
                'problem_text': problem['text'],
                'problem_type': problem.get('math_type', 'UNKNOWN'),
                'errors': analysis.get('errors', []),
                'vllm_feedback': problem.get('custom_feedback', {}),
                'socratic_approach': [],
                'direct_instruction': [],
                'visualization_suggestion': {}
            }
            
            # Add built-in feedback templates based on problem type
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
    
    def _generate_overall_feedback(self, problems):
        """
        Generate overall feedback summary for the student
        
        Args:
            problems: List of analyzed problems
            
        Returns:
            Overall feedback dictionary
        """
        # Count errors by type
        error_counts = {}
        total_problems = len(problems)
        problems_with_errors = 0
        
        for problem in problems:
            if 'analysis' in problem and problem['analysis'].get('errors', []):
                problems_with_errors += 1
                for error in problem['analysis']['errors']:
                    error_type = error.get('type', 'UNKNOWN')
                    if error_type in error_counts:
                        error_counts[error_type] += 1
                    else:
                        error_counts[error_type] = 1
        
        # Calculate performance metrics
        correct_percentage = ((total_problems - problems_with_errors) / total_problems * 100) if total_problems > 0 else 0
        
        # Determine areas of strength and weakness
        problem_types = {}
        for problem in problems:
            if 'math_type' in problem:
                math_type = problem['math_type']
                if math_type not in problem_types:
                    problem_types[math_type] = {
                        'count': 0,
                        'errors': 0
                    }
                
                problem_types[math_type]['count'] += 1
                if 'analysis' in problem and problem['analysis'].get('errors', []):
                    problem_types[math_type]['errors'] += 1
        
        # Calculate error rates by problem type
        for math_type, stats in problem_types.items():
            if stats['count'] > 0:
                stats['error_rate'] = stats['errors'] / stats['count'] * 100
            else:
                stats['error_rate'] = 0
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for math_type, stats in problem_types.items():
            if stats['count'] >= 2:  # Only consider types with enough examples
                if stats['error_rate'] <= 25:  # 25% or less error rate is a strength
                    strengths.append(math_type)
                elif stats['error_rate'] >= 50:  # 50% or more error rate is a weakness
                    weaknesses.append(math_type)
        
        # Generate the overall feedback
        overall_feedback = {
            'performance_summary': {
                'total_problems': total_problems,
                'correct': total_problems - problems_with_errors,
                'incorrect': problems_with_errors,
                'correct_percentage': correct_percentage,
                'error_types': error_counts
            },
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': []
        }
        
        # Add personalized recommendations
        if strengths:
            overall_feedback['recommendations'].append(
                f"You're showing good understanding of {', '.join(strengths)}. Keep up the good work!"
            )
            
        if weaknesses:
            overall_feedback['recommendations'].append(
                f"Focus on improving your understanding of {', '.join(weaknesses)}. The practice worksheet will help with these areas."
            )
            
        if correct_percentage >= 80:
            overall_feedback['recommendations'].append(
                "Excellent work overall! Challenge yourself with more advanced problems."
            )
        elif correct_percentage >= 60:
            overall_feedback['recommendations'].append(
                "Good progress. Continue practicing to reinforce your understanding."
            )
        else:
            overall_feedback['recommendations'].append(
                "Keep practicing the fundamentals. Work through the practice worksheet step by step."
            )
            
        return overall_feedback


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize the system
    math_system = EnhancedMathFeedbackSystem()
    
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
