"""
Integration module to connect handwritten math OCR with the Math Feedback System
"""

import os
import cv2
import numpy as np
import json
import math
import re
from pathlib import Path
from .handwritten_math_ocr import HandwrittenMathOCR
from .azure_integration import AzureMathAnalyzer
from .analysis import MathAnalyzer

class MathRecognitionSystem:
    """
    Integrated system for handwritten math recognition and feedback
    """
    
    def __init__(self):
        """
        Initialize the math recognition system
        """
        try:
            # Initialize handwritten math OCR
            from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
            self.handwritten_math_ocr = HandwrittenMathOCR()
            
            # Initialize Azure OCR
            from math_analyzer.azure_integration import AzureMathAnalyzer
            self.azure_math_analyzer = AzureMathAnalyzer()
            
            print("Math Recognition System initialized successfully")
        except Exception as e:
            print(f"Error initializing Math Recognition System: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize math analysis module
        self.math_analyzer = MathAnalyzer()
        
        # Initialize performance metrics
        self.performance_metrics = {
            'ocr_accuracy': [],
            'problem_detection_rate': [],
            'error_detection_rate': []
        }
        
        # Define common mathematical errors for pattern matching
        self.common_errors = {
            'addition': {
                'carrying': 'Forgot to carry the 1',
                'place_value': 'Misaligned place values',
                'simple_fact': 'Basic addition fact error'
            },
            'subtraction': {
                'borrowing': 'Incorrect borrowing',
                'place_value': 'Misaligned place values',
                'negative': 'Failed to recognize negative result needed',
                'simple_fact': 'Basic subtraction fact error'
            },
            'multiplication': {
                'carrying': 'Carrying error in multiplication',
                'place_value': 'Misaligned place values',
                'simple_fact': 'Basic multiplication fact error',
                'zero_rule': 'Error in multiplying with zero'
            },
            'division': {
                'remainder': 'Incorrect remainder',
                'place_value': 'Misaligned place values during long division',
                'simple_fact': 'Basic division fact error',
                'zero_division': 'Division by zero error'
            }
        }
    
    def process_homework_image(self, image_path, output_dir="results"):
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
        
        # Step 1: Extract problems using OCR
        # Start with specialized OCR but always also run Azure for comparison and hybrid approach
        try:
            # Run specialized OCR
            specialized_results = self.handwritten_math_ocr.recognize_expression(image_path)
            specialized_problems = self._extract_problems_from_ocr(specialized_results)
            print(f"Specialized OCR detected {len(specialized_problems)} potential problems")
        except Exception as e:
            print(f"Specialized OCR failed: {str(e)}. Will rely on Azure OCR.")
            specialized_problems = []
        
        # Also run Azure OCR for all images - we'll use a hybrid approach
        print("Using Azure OCR...")
        try:
            print("Submitting image to Azure OCR...")
            azure_results = self.azure_math_analyzer.analyze_image(image_path)
            azure_problems = self._extract_problems_from_azure(azure_results)
            print(f"Azure OCR detected {len(azure_problems)} potential problems")
        except Exception as e:
            print(f"Azure OCR failed: {str(e)}")
            azure_problems = []

        # Use a hybrid approach - combine and deduplicate problems from both systems
        problems = self._merge_problem_results(specialized_problems, azure_problems)
        print(f"Combined detection found {len(problems)} unique problems")
        
        # Step 2: Analyze each detected problem
        analysis_results = self._analyze_problems(problems, image_path)
        
        # Step 3: Generate comprehensive feedback
        feedback = self._generate_comprehensive_feedback(analysis_results, image_path)
        
        # Step 4: Create visualizations
        visualization_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_feedback_visualization.jpg")
        self._create_error_visualization(image_path, feedback, visualization_path)
        
        # Save feedback to JSON file
        feedback_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_math_feedback.json")
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=4, ensure_ascii=False)
        
        return feedback
    
    def _merge_problem_results(self, specialized_problems, azure_problems):
        """
        Merge problem results from specialized OCR and Azure OCR,
        using the strengths of each system and deduplicating overlapping detections
        
        Args:
            specialized_problems: Problems detected by specialized OCR
            azure_problems: Problems detected by Azure OCR
            
        Returns:
            Combined list of unique problems with the best detection for each
        """
        merged_problems = specialized_problems.copy()
        
        # Track problems we've already merged to avoid duplicates
        processed_azure_indices = set()
        
        # First, try to match Azure problems with specialized problems and enhance them
        for i, spec_prob in enumerate(specialized_problems):
            best_match_idx = -1
            best_match_score = 0.3  # Minimum threshold for considering a match
            
            for j, azure_prob in enumerate(azure_problems):
                if j in processed_azure_indices:
                    continue
                
                # Check if problems are duplicates
                if self._are_problems_duplicates(spec_prob, azure_prob):
                    match_score = self._calculate_problem_match_score(spec_prob, azure_prob)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = j
            
            # If we found a good match, merge the Azure problem into the specialized problem
            if best_match_idx >= 0:
                merged_problems[i] = self._merge_problem_info(spec_prob, azure_problems[best_match_idx])
                processed_azure_indices.add(best_match_idx)
        
        # Add any unmatched Azure problems
        for j, azure_prob in enumerate(azure_problems):
            if j not in processed_azure_indices:
                # Check if this Azure problem is likely a duplicate of any in our final list
                is_duplicate = False
                for merged_prob in merged_problems:
                    if self._are_problems_duplicates(merged_prob, azure_prob):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Mark it as coming from Azure
                    azure_prob["source"] = "azure"
                    merged_problems.append(azure_prob)
        
        return merged_problems
    
    def _are_problems_duplicates(self, prob1, prob2):
        """
        Check if two problems are likely duplicates
        
        Args:
            prob1: First problem
            prob2: Second problem
            
        Returns:
            True if problems are likely duplicates based on IoU and text similarity
        """
        # Check bounding box overlap
        if "bbox" in prob1 and "bbox" in prob2:
            iou = self._calculate_iou(prob1["bbox"], prob2["bbox"])
            # If boxes overlap significantly, they're likely the same problem
            if iou > 0.5:
                return True
        
        # Check text similarity if text is available
        if "text" in prob1 and "text" in prob2:
            text_sim = self._text_similarity(prob1["text"], prob2["text"])
            # If text is very similar, they're likely the same problem
            if text_sim > 0.7:
                return True
        
        # Check for equation similarity
        if ("equation" in prob1 and "equation" in prob2 and 
            prob1["equation"] and prob2["equation"]):
            eq_sim = self._text_similarity(prob1["equation"], prob2["equation"])
            if eq_sim > 0.7:
                return True
        
        return False
    
    def _calculate_problem_match_score(self, prob1, prob2):
        """
        Calculate a match score between two problems
        
        Args:
            prob1: First problem
            prob2: Second problem
            
        Returns:
            Match score between 0 and 1
        """
        score = 0
        score_components = 0
        
        # Check bounding box overlap
        if "bbox" in prob1 and "bbox" in prob2:
            iou = self._calculate_iou(prob1["bbox"], prob2["bbox"])
            score += iou
            score_components += 1
        
        # Check text similarity
        if "text" in prob1 and "text" in prob2:
            text_sim = self._text_similarity(prob1["text"], prob2["text"])
            score += text_sim
            score_components += 1
        
        # Check equation similarity
        if "equation" in prob1 and "equation" in prob2 and prob1["equation"] and prob2["equation"]:
            eq_sim = self._text_similarity(prob1["equation"], prob2["equation"])
            score += eq_sim
            score_components += 1
        
        # Avoid division by zero
        if score_components == 0:
            return 0
            
        return score / score_components
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union for two bounding boxes
        
        Args:
            box1: First bounding box [x, y, w, h] or [x1, y1, x2, y2]
            box2: Second bounding box [x, y, w, h] or [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            if box1[2] < box1[0] or box1[3] < box1[1]:  # It's [x, y, w, h]
                box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
            if box2[2] < box2[0] or box2[3] < box2[1]:  # It's [x, y, w, h]
                box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Return IoU
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    
    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for quick comparison
        if not text1 or not text2:
            return 0.0
            
        # Normalize and tokenize
        text1 = text1.lower().replace('\n', ' ').strip()
        text2 = text2.lower().replace('\n', ' ').strip()
        
        # Quick check for exact match
        if text1 == text2:
            return 1.0
            
        # Convert to sets of words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _merge_problem_info(self, specialized_problem, azure_problem):
        """
        Merge information from Azure problem into specialized problem
        
        Args:
            specialized_problem: Problem detected by specialized OCR
            azure_problem: Problem detected by Azure OCR
            
        Returns:
            Merged problem with the best information from both sources
        """
        merged = specialized_problem.copy()
        
        # Always keep the source information
        merged["source"] = "hybrid"
        
        # Use Azure's text if specialized OCR couldn't detect it well
        if ("text" not in specialized_problem or not specialized_problem["text"]) and "text" in azure_problem:
            merged["text"] = azure_problem["text"]
        
        # Use Azure's equation if specialized OCR couldn't detect it well
        if ("equation" not in specialized_problem or not specialized_problem["equation"]) and "equation" in azure_problem:
            merged["equation"] = azure_problem["equation"]
        
        # Prefer Azure's confidence if higher
        if "confidence" in azure_problem:
            if "confidence" not in specialized_problem or azure_problem["confidence"] > specialized_problem["confidence"]:
                merged["confidence"] = azure_problem["confidence"]
        
        # Merge other fields selectively
        for key, value in azure_problem.items():
            if key not in merged and value:
                merged[key] = value
        
        # Enhanced problem metadata
        if "is_equation" in azure_problem:
            merged["is_equation"] = True
        
        if "operation" in azure_problem:
            merged["operation"] = azure_problem["operation"]
        
        if "math_concept" in azure_problem:
            merged["math_concept"] = azure_problem["math_concept"]
            
        # Special handling for Pythagorean theorem
        if "pythagorean_form" in azure_problem and azure_problem["pythagorean_form"]:
            merged["pythagorean_form"] = True
            merged["math_concept"] = "pythagorean_theorem"
        
        return merged
        
    def process_homework_image(self, image_path, create_visualization=False):
        """
        Process a math homework image with hybrid OCR (our specialized OCR + Azure OCR)
        
        Args:
            image_path: Path to the image to process
            create_visualization: Whether to create a visualization of the results
            
        Returns:
            Analysis results with detailed feedback
        """
        try:
            print(f"Processing homework image: {image_path}")
            
            # Load image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return {"error": f"Image file not found: {image_path}"}
                image = cv2.imread(image_path)
            else:
                # Assume it's already an image
                image = image_path
                
            if image is None:
                return {"error": "Failed to load image"}
                
            # Process with specialized OCR first
            try:
                specialized_results = self._process_with_specialized_ocr(image)
                specialized_problems = specialized_results.get("problems", [])
                print(f"Specialized OCR detected {len(specialized_problems)} problems")
            except Exception as e:
                print(f"Specialized OCR failed: {str(e)}")
                specialized_problems = []
                
            # Process with Azure OCR
            try:
                azure_results = self._process_with_azure_ocr(image_path)
                azure_problems = azure_results.get("problems", [])
                print(f"Azure OCR detected {len(azure_problems)} problems")
            except Exception as e:
                print(f"Azure OCR failed: {str(e)}")
                azure_problems = []
                
            # Merge the results for the best of both worlds
            all_problems = self._merge_problem_results(specialized_problems, azure_problems)
            
            # Analyze each problem for errors
            analysis_results = []
            
            for problem in all_problems:
                problem_analysis = self._analyze_problem(problem)
                if problem_analysis:
                    analysis_results.append(problem_analysis)
                    
            # Create result object
            result = {
                "problems": all_problems,
                "analysis": analysis_results,
                "problem_count": len(all_problems),
                "error_count": sum(1 for p in analysis_results if p.get("has_error", False))
            }
            
            # Create visualization if requested
            if create_visualization:
                visualization = self._create_visualization(image, all_problems, analysis_results)
                result["visualization"] = visualization
                
            return result
            
        except Exception as e:
            print(f"Error processing homework image: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Processing failed: {str(e)}"}
    
    def _process_with_specialized_ocr(self, image):
        """
        Process image with our specialized OCR system with improved uncertainty handling
        
        Args:
            image: Image data
            
        Returns:
            Dictionary with detected problems
        """
        problems = []
        
        try:
            # Extract math expressions using our specialized OCR
            recognized_expressions = self.handwritten_math_ocr.recognize_expression(image)
            
            # If the result is a dict with a single expression, wrap it in a list
            if isinstance(recognized_expressions, dict):
                recognized_expressions = [recognized_expressions]
            
            for i, expr in enumerate(recognized_expressions):
                # Extract text and confidence
                text = expr.get("text", "")
                confidence = expr.get("confidence", 0.0)
                
                # Get symbols
                symbols = expr.get("symbols", [])
                
                # Skip if no content
                if not text.strip():
                    continue
                
                # Clean up text to handle uncertain characters
                cleaned_text = self.handwritten_math_ocr.clean_recognized_text(text)
                
                # If still contains numerous question marks, try equation completion
                if cleaned_text.count('?') > 2:
                    completed_text = self.handwritten_math_ocr.complete_uncertain_equation(cleaned_text)
                else:
                    completed_text = cleaned_text
                
                # Get bounding box for the expression
                if symbols:
                    bbox = self.handwritten_math_ocr._get_expression_bbox(symbols)
                else:
                    # Default bbox if none available
                    bbox = [0, 0, 100, 30]
                
                problem = {
                    "id": i + 1,
                    "text": completed_text,
                    "raw_text": text,
                    "equation": completed_text,
                    "bbox": bbox,
                    "confidence": confidence,
                    "source": "specialized_ocr",
                    "symbols": symbols,
                    "uncertain_chars_count": text.count('?'),
                    "improvement_level": self._calculate_improvement_level(text, completed_text)
                }
                
                # Parse the equation to extract its structure
                parsed = self._parse_equation(completed_text)
                if parsed:
                    for key, value in parsed.items():
                        problem[key] = value
                
                problems.append(problem)
                
        except Exception as e:
            print(f"Specialized OCR processing error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {"problems": problems}
    
    def _calculate_improvement_level(self, original_text, improved_text):
        """
        Calculate how much the text was improved by uncertain character handling
        
        Args:
            original_text: Original text with uncertain characters
            improved_text: Improved text after processing
            
        Returns:
            Improvement level (0-1)
        """
        if not original_text:
            return 0
        
        # Count question marks in original and improved
        original_uncertain = original_text.count('?')
        improved_uncertain = improved_text.count('?')
        
        # If no uncertain characters in original, no improvement needed
        if original_uncertain == 0:
            return 1.0
            
        # Calculate reduction in uncertain characters
        if original_uncertain > 0:
            return min(1.0, (original_uncertain - improved_uncertain) / original_uncertain)
        
        return 0.0
    
    def _process_with_azure_ocr(self, image_path):
        """
        Process image with Azure OCR
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with detected problems
        """
        try:
            # Call Azure OCR
            azure_results = self.azure_math_analyzer.analyze_image(image_path)
            
            if "error" in azure_results:
                print(f"Azure OCR error: {azure_results['error']}")
                return {"problems": []}
            
            # Extract math problems from Azure OCR results
            problems = self._extract_problems_from_azure(azure_results)
            
            return {"problems": problems}
            
        except Exception as e:
            print(f"Azure OCR processing error: {str(e)}")
            return {"problems": []}
    
    def _extract_problems_from_azure(self, azure_results):
        """
        Extract math problems from Azure OCR results
        
        Args:
            azure_results: Results from Azure OCR
            
        Returns:
            List of extracted problems
        """
        problems = []
        
        if "ocr_result" not in azure_results:
            return problems
            
        ocr_result = azure_results["ocr_result"]
        
        # Process each line detected by Azure
        for i, line in enumerate(ocr_result.get("lines", [])):
            text = line.get("text", "").strip()
            
            # Skip empty lines
            if not text:
                continue
                
            # Check if it's a math expression
            if "=" in text or any(op in text for op in "+-*/×÷"):
                bbox = line.get("bounding_box", [0, 0, 100, 30])
                
                # Convert bbox format if needed
                if len(bbox) == 4:  # [x, y, w, h]
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                problem = {
                    "id": i + 1,
                    "text": text,
                    "equation": text,
                    "bbox": bbox,
                    "confidence": line.get("confidence", 0.8),
                    "source": "azure_ocr"
                }
                
                # Add any enhanced information from Azure
                if "is_equation" in line:
                    problem["is_equation"] = line["is_equation"]
                
                if "operation" in line:
                    problem["operation"] = line["operation"]
                
                if "math_concept" in line:
                    problem["math_concept"] = line["math_concept"]
                
                if "pythagorean_form" in line:
                    problem["pythagorean_form"] = line["pythagorean_form"]
                
                # Parse the equation to extract its structure
                parsed = self._parse_equation(text)
                if parsed:
                    for key, value in parsed.items():
                        problem[key] = value
                
                problems.append(problem)
        
        return problems
    
    def _parse_equation(self, equation_text):
        """
        Parse an equation to extract its structure
        
        Args:
            equation_text: Text of the equation
            
        Returns:
            Dictionary with parsed components
        """
        result = {}
        
        # Skip if empty
        if not equation_text:
            return result
            
        # Basic normalization
        equation_text = equation_text.replace('×', '*').replace('÷', '/')
        
        # Check for Pythagorean theorem pattern
        pythagorean_pattern = r'([a-zA-Z0-9]+)\s*[\^²]\s*2\s*\+\s*([a-zA-Z0-9]+)\s*[\^²]\s*2\s*=\s*([a-zA-Z0-9]+)\s*[\^²]\s*2'
        pythagorean_match = re.search(pythagorean_pattern, equation_text)
        
        if pythagorean_match:
            result["type"] = "pythagorean_theorem"
            result["a"] = pythagorean_match.group(1)
            result["b"] = pythagorean_match.group(2)
            result["c"] = pythagorean_match.group(3)
            
            # Try to convert to numbers if possible
            try:
                a_val = float(result["a"])
                b_val = float(result["b"])
                c_val = float(result["c"])
                
                # Verify if it's a valid Pythagorean triple
                result["is_valid"] = abs(a_val**2 + b_val**2 - c_val**2) < 0.01
                result["expected_c"] = math.sqrt(a_val**2 + b_val**2)
                
            except ValueError:
                # If we can't convert to numbers, it might be variables
                result["is_valid"] = None
                
            return result
        
        # Check for basic operation patterns
        basic_op_pattern = r'([a-zA-Z0-9.]+)\s*([\+\-\*/])\s*([a-zA-Z0-9.]+)\s*=\s*([a-zA-Z0-9.]+)'
        op_match = re.search(basic_op_pattern, equation_text)
        
        if op_match:
            first = op_match.group(1)
            operation = op_match.group(2)
            second = op_match.group(3)
            result_val = op_match.group(4)
            
            result["first_number"] = first
            result["operation"] = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}[operation]
            result["second_number"] = second
            result["result"] = result_val
            
            # Try to validate the equation
            try:
                n1 = float(first)
                n2 = float(second)
                res = float(result_val)
                
                if operation == "+":
                    result["is_valid"] = abs(n1 + n2 - res) < 0.01
                    result["expected_result"] = n1 + n2
                elif operation == "-":
                    result["is_valid"] = abs(n1 - n2 - res) < 0.01
                    result["expected_result"] = n1 - n2
                elif operation == "*":
                    result["is_valid"] = abs(n1 * n2 - res) < 0.01
                    result["expected_result"] = n1 * n2
                elif operation == "/":
                    if n2 != 0:
                        result["is_valid"] = abs(n1 / n2 - res) < 0.01
                        result["expected_result"] = n1 / n2
                    else:
                        result["is_valid"] = False
                        result["error"] = "division_by_zero"
            except ValueError:
                # If we can't convert to numbers, it might contain variables
                result["is_valid"] = None
                
            return result
        
        # For other equation types, just mark it as an equation
        if "=" in equation_text:
            result["type"] = "general_equation"
            result["left_side"] = equation_text.split("=")[0].strip()
            result["right_side"] = equation_text.split("=")[1].strip() if len(equation_text.split("=")) > 1 else ""
            
        return result
    
def _parse_equation(self, equation_text):
    """
    Parse an equation to extract its structure
        
    Args:
        equation_text: Text of the equation
            
    Returns:
        Dictionary with parsed components
    """
    result = {}
        
    # Skip if empty
    if not equation_text:
        return result
            
    # Basic normalization
    equation_text = equation_text.replace('×', '*').replace('÷', '/')
        
    # Check for Pythagorean theorem pattern
    pythagorean_pattern = r'([a-zA-Z0-9]+)\s*[\^²]\s*2\s*\+\s*([a-zA-Z0-9]+)\s*[\^²]\s*2\s*=\s*([a-zA-Z0-9]+)\s*[\^²]\s*2'
    pythagorean_match = re.search(pythagorean_pattern, equation_text)
        
    if pythagorean_match:
        result["type"] = "pythagorean_theorem"
        result["a"] = pythagorean_match.group(1)
        result["b"] = pythagorean_match.group(2)
        result["c"] = pythagorean_match.group(3)
            
        # Try to convert to numbers if possible
        try:
            a_val = float(result["a"])
            b_val = float(result["b"])
            c_val = float(result["c"])
                
            # Verify if it's a valid Pythagorean triple
            result["is_valid"] = abs(a_val**2 + b_val**2 - c_val**2) < 0.01
            result["expected_c"] = math.sqrt(a_val**2 + b_val**2)
                
        except ValueError:
            # If we can't convert to numbers, it might be variables
            result["is_valid"] = None
                
        return result
        
    # Check for basic operation patterns
    basic_op_pattern = r'([a-zA-Z0-9.]+)\s*([\+\-\*/])\s*([a-zA-Z0-9.]+)\s*=\s*([a-zA-Z0-9.]+)'
    op_match = re.search(basic_op_pattern, equation_text)
        
    if op_match:
        first = op_match.group(1)
        operation = op_match.group(2)
        second = op_match.group(3)
        result_val = op_match.group(4)
            
        result["first_number"] = first
        result["operation"] = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}[operation]
        result["second_number"] = second
        result["result"] = result_val
            
        # Try to validate the equation
        try:
            n1 = float(first)
            n2 = float(second)
            res = float(result_val)
                
            if operation == "+":
                result["is_valid"] = abs(n1 + n2 - res) < 0.01
                result["expected_result"] = n1 + n2
            elif operation == "-":
                result["is_valid"] = abs(n1 - n2 - res) < 0.01
                result["expected_result"] = n1 - n2
            elif operation == "*":
                result["is_valid"] = abs(n1 * n2 - res) < 0.01
                result["expected_result"] = n1 * n2
            elif operation == "/":
                if n2 != 0:
                    result["is_valid"] = abs(n1 / n2 - res) < 0.01
                    result["expected_result"] = n1 / n2
                else:
                    result["is_valid"] = False
                    result["error"] = "division_by_zero"
        except ValueError:
            # If we can't convert to numbers, it might contain variables
            result["is_valid"] = None
                
        return result
        
    # For other equation types, just mark it as an equation
    if "=" in equation_text:
        result["type"] = "general_equation"
        result["left_side"] = equation_text.split("=")[0].strip()
        result["right_side"] = equation_text.split("=")[1].strip() if len(equation_text.split("=")) > 1 else ""
            
    return result
    
def _analyze_problems(self, problems, image_path):
    """
    Analyze detected math problems using the AdvancedMathAnalyzer
        
    Args:
        problems: List of detected problems
        image_path: Path to the original image
            
    Returns:
        Analysis results with enhanced math insights
    """
    # Initialize the advanced math analyzer if not already done
    try:
        from math_analyzer.advanced_math_analyzer import AdvancedMathAnalyzer
        advanced_analyzer = AdvancedMathAnalyzer()
    except Exception as e:
        print(f"Warning: Could not initialize AdvancedMathAnalyzer: {e}")
        advanced_analyzer = None
            
    # Use the math analyzer to analyze each problem
    for problem in problems:
        try:
            # Extract the text from the problem
            text = problem.get('text', '')
                
            # Skip empty text
            if not text.strip():
                continue
                
            # First try the advanced analyzer for more sophisticated analysis
            if advanced_analyzer:
                try:
                    # Normalize and analyze the expression
                    normalized_text = advanced_analyzer.normalize_expression(text)
                    advanced_analysis = advanced_analyzer.analyze_expression(normalized_text)
                        
                    # Store advanced analysis 
                    problem['analysis'] = advanced_analysis
                    problem['normalized_text'] = normalized_text
                        
                    # Extract problem type
                    problem_type = advanced_analysis.get('type', 'UNKNOWN')
                    problem['math_type'] = problem_type
                        
                    # Check if errors were found
                    has_errors = len(advanced_analysis.get('errors', [])) > 0
                        
                    print(f"Advanced analysis of '{text}' (type: {problem_type}), errors found: {has_errors}")
                except Exception as e:
                    print(f"Advanced analysis failed for '{text}': {e}")
                    # Fall back to basic analysis
                    advanced_analysis = None
                        
            # If advanced analysis failed or is not available, use basic analyzer
            if not advanced_analyzer or 'analysis' not in problem:
                # Basic analysis
                basic_analysis = self.math_analyzer.analyze(text)
                problem['analysis'] = basic_analysis
                
            # Update performance metrics
            if 'errors' in problem['analysis'] and len(problem['analysis']['errors']) > 0:
                self.performance_metrics['error_detection_rate'].append(1.0)
            else:
                self.performance_metrics['error_detection_rate'].append(0.0)
                    
        except Exception as e:
            print(f"Error analyzing problem '{problem.get('text', '')}':{str(e)}")
            problem['analysis'] = {
                'errors': [{
                    'type': 'analysis_error',
                    'description': f'Error during analysis: {str(e)}'
                }]
            }
                
    return problems
    
def _analyze_problem(self, problem):
    """
    Analyze a single math problem
        
    Args:
        problem: Problem data
            
    Returns:
        Analysis result
    """
    analysis = {
        "id": problem["id"],
        "text": problem["text"],
        "bbox": problem.get("bbox", [0, 0, 0, 0]),
        "parsed": {},
        "is_correct": None,
        "error_type": None,
        "error_message": None,
        "expected_result": None,
        "confidence": problem.get("confidence", 0)
    }
        
    # Extract operation and operands
    parsed_data = self._parse_equation(problem["text"])
    if parsed_data:
        analysis["parsed"] = parsed_data
            
        # If we have a valid parsed expression, check correctness
        if all(k in parsed_data for k in ['first_number', 'operation', 'second_number', 'result']):
            # Calculate expected result
            expected = self._calculate_expected_result(
                parsed_data['first_number'],
                parsed_data['operation'],
                parsed_data['second_number']
            )
            analysis["expected_result"] = expected
                
            # Check if correct
            if expected is not None and parsed_data['result'] is not None:
                analysis["is_correct"] = abs(expected - parsed_data['result']) < 0.001
                    
                # Determine error type if incorrect
                if not analysis["is_correct"]:
                    analysis["error_type"] = self._determine_error_type(
                        parsed_data['first_number'],
                        parsed_data['operation'],
                        parsed_data['second_number'],
                        parsed_data['result'],
                        expected
                    )
                        analysis["is_correct"] = abs(expected - parsed_data['result']) < 0.001
                        
                        # Determine error type if incorrect
                        if not analysis["is_correct"]:
                            analysis["error_type"] = self._determine_error_type(
                                parsed_data['first_number'],
                                parsed_data['operation'],
                                parsed_data['second_number'],
                                parsed_data['result'],
                                expected
                            )
            
            # Handle special types of problems
            if problem.get("structure_type") == 'pythagorean_theorem':
                # Extract the sides from the structure data
                sides = problem.get("structure_data", {}).get("sides", [])
                if len(sides) == 3 and all(sides):
                    try:
                        # If using variables, we can't fully validate
                        if not all(s.isdigit() for s in sides):
                            analysis["is_correct"] = problem.get("is_valid")
                        else:
                            a, b, c = map(int, sides)
                            analysis["parsed"] = {
                                "operation": "pythagorean",
                                "side_a": a,
                                "side_b": b,
                                "side_c": c
                            }
                            expected = (a**2 + b**2 == c**2)
                            analysis["is_correct"] = expected
                            if not expected:
                                analysis["error_type"] = "pythagorean_theorem_error"
                                analysis["error_message"] = f"{a}² + {b}² = {a**2 + b**2}, not equal to {c}² = {c**2}"
                    except:
                        pass
            
            # For general equations, use the validation result
            elif '=' in problem["text"] and problem.get("is_valid") is not None:
                analysis["is_correct"] = problem.get("is_valid")
                if not analysis["is_correct"]:
                    analysis["error_message"] = problem.get("validation_message", 'Equation is not mathematically valid')
                    analysis["error_type"] = "equation_error"
                
            analysis_results.append(analysis)
        
        return analysis_results
    
    def _analyze_problem(self, problem):
        """
        Analyze a single math problem
        
        Args:
            problem: Problem data
        
        Returns:
            Analysis result
        """
        analysis = {
            "id": problem["id"],
            "text": problem["text"],
            "bbox": problem.get("bbox", [0, 0, 0, 0]),
            "parsed": {},
            "is_correct": None,
            "error_type": None,
            "error_message": None,
            "expected_result": None,
            "confidence": problem.get("confidence", 0)
        }
        
        # Extract operation and operands
        parsed_data = self._parse_equation(problem["text"])
        if parsed_data:
            analysis["parsed"] = parsed_data
            
            # If we have a valid parsed expression, check correctness
            if all(k in parsed_data for k in ['first_number', 'operation', 'second_number', 'result']):
                # Calculate expected result
                expected = self._calculate_expected_result(
                    parsed_data['first_number'],
                    parsed_data['operation'],
                    parsed_data['second_number']
                )
                analysis["expected_result"] = expected
                
                # Check if correct
                if expected is not None and parsed_data['result'] is not None:
                    analysis["is_correct"] = abs(expected - parsed_data['result']) < 0.001
                    
                    # Determine error type if incorrect
                    if not analysis["is_correct"]:
                        analysis["error_type"] = self._determine_error_type(
                            parsed_data['first_number'],
                            parsed_data['operation'],
                            parsed_data['second_number'],
                            parsed_data['result'],
                            expected
                        )
        
        # Handle special types of problems
        if problem.get("structure_type") == 'pythagorean_theorem':
            # Extract the sides from the structure data
            sides = problem.get("structure_data", {}).get("sides", [])
            if len(sides) == 3 and all(sides):
                try:
                    # If using variables, we can't fully validate
                    if not all(s.isdigit() for s in sides):
                        analysis["is_correct"] = problem.get("is_valid")
                    else:
                        a, b, c = map(int, sides)
                        analysis["parsed"] = {
                            "operation": "pythagorean",
                            "side_a": a,
                            "side_b": b,
                            "side_c": c
                        }
                        expected = (a**2 + b**2 == c**2)
                        analysis["is_correct"] = expected
                        if not expected:
                            analysis["error_type"] = "pythagorean_theorem_error"
                            analysis["error_message"] = f"{a}² + {b}² = {a**2 + b**2}, not equal to {c}² = {c**2}"
                except:
                    pass
        
        # For general equations, use the validation result
        elif '=' in problem["text"] and problem.get("is_valid") is not None:
            analysis["is_correct"] = problem.get("is_valid")
            if not analysis["is_correct"]:
                analysis["error_message"] = problem.get("validation_message", 'Equation is not mathematically valid')
                analysis["error_type"] = "equation_error"
        
        return analysis
    
    def _calculate_expected_result(self, a, operation, b):
        """
        Calculate the expected result based on operation
        
        Args:
            a: First operand
            operation: Operation type
            b: Second operand
            
        Returns:
            Expected result or None if error
        """
        try:
            if operation == '+':
                return a + b
            elif operation == '-':
                return a - b
            elif operation == '*':
                return a * b
            elif operation == '/':
                # Handle division by zero
                if b == 0:
                    return None
                return a / b
            else:
                return None
        except:
            return None
    
    def _determine_error_type(self, a, operation, b, result, expected):
        """
        Determine the type of error made in the calculation
        
        Args:
            a: First operand
            operation: Operation
            b: Second operand
            result: Student's result
            expected: Expected result
            
        Returns:
            Error type as string
        """
        # Handle addition errors
        if operation == '+':
            if result == a - b:
                return 'subtracted_instead_of_added'
            elif result == a * b:
                return 'multiplied_instead_of_added'
            elif result == expected + 1:
                return 'off_by_one_high'
            elif result == expected - 1:
                return 'off_by_one_low'
            elif result == expected + 10:
                return 'place_value'
            elif (result == b - a and b > a):
                return 'reversed_operands'
        
        # Handle subtraction errors
        elif operation == '-':
            if result == a + b:
                return 'added_instead_of_subtracted'
            elif result == a * b:
                return 'multiplied_instead_of_subtracted'
            elif result == b - a:
                return 'reversed_operands'
            elif result == expected + 1:
                return 'off_by_one_high'
            elif result == expected - 1:
                return 'off_by_one_low'
            elif result >= 0 and expected < 0:
                return 'negative_result'
                
        # Handle multiplication errors
        elif operation == '*':
            if result == a + b:
                return 'added_instead_of_multiplied'
            elif result == a / b and b != 0:
                return 'divided_instead_of_multiplied'
            elif result == expected + a:
                return 'multiplication_table'
            elif result == expected - a:
                return 'multiplication_table'
                
        # Handle division errors
        elif operation == '/':
            if result == a * b:
                return 'multiplied_instead_of_divided'
            elif result == a + b:
                return 'added_instead_of_divided'
            elif result == a - b:
                return 'subtracted_instead_of_divided'
            elif result == b / a and a != 0:
                return 'reversed_operands'
            
        # Pythagorean theorem specific errors
        elif operation == 'pythagorean':
            return 'pythagorean_theorem_error'
            
        # Default case - general calculation error
        return 'calculation_error'
    
    def _generate_problem_feedback(self, problem_data):
        """
        Generate detailed pedagogical feedback for a specific problem
        
        Args:
            problem_data: Analysis data for the problem
            
        Returns:
            Feedback dictionary
        """
        # Initialize feedback structure
        feedback = {
            "concept_explanation": "",
            "learning_strategy": "",
            "socratic_questions": [],
            "direct_instruction": [],
            "visual_aids": {}
        }
        
        # Get error type
        error_type = problem_data.get('error_type', 'unknown')
        operation = problem_data.get('parsed', {}).get('operation', 'unknown')
        
        # Generate concept explanation based on error type and operation
        concept_explanations = {
            'subtracted_instead_of_added': "Addition means combining quantities. When we add, the total amount increases.",
            'added_instead_of_subtracted': "Subtraction means taking away or finding the difference between two numbers.",
            'multiplied_instead_of_added': "Addition and multiplication are different operations. Addition combines quantities, while multiplication involves repeated addition.",
            'added_instead_of_multiplied': "Multiplication is repeated addition. For example, 3 × 4 means 3 + 3 + 3 + 3 or 4 + 4 + 4.",
            'off_by_one_high': "Be careful when counting. It's easy to count one too many.",
            'off_by_one_low': "Be careful when counting. It's easy to miss one.",
            'place_value': "Each position in a number represents a different value (ones, tens, hundreds).",
            'reversed_operands': "The order of numbers matters in subtraction and division, but not in addition and multiplication.",
            'negative_result': "When subtracting a larger number from a smaller one, the result is negative.",
            'multiplication_fact': "Memorizing multiplication facts helps solve problems quickly and accurately.",
            'pythagorean_theorem_error': "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides (a² + b² = c²)."
        }
        
        # Generate learning strategy based on error type and operation
        learning_strategies = {
            'subtracted_instead_of_added': "Use objects to physically represent the problem.",
            'added_instead_of_subtracted': "Draw a number line to visualize subtraction.",
            'multiplied_instead_of_added': "Circle the operation symbol to help remember which operation to perform.",
            'added_instead_of_multiplied': "Practice skip counting to build multiplication skills.",
            'off_by_one_high': "Practice counting up from the larger number.",
            'off_by_one_low': "Double-check your counting by using a different method.",
            'place_value': "Use place value blocks or charts to understand how numbers work.",
            'reversed_operands': "Remember to always put the larger number first when doing subtraction if you haven't learned negative numbers.",
            'negative_result': "Use a number line that includes negative numbers.",
            'multiplication_fact': "Create flashcards for multiplication facts.",
            'pythagorean_theorem_error': "Practice with right triangles using grid paper to visualize the theorem."
        }
        
        # Generate Socratic questions based on error type
        socratic_questions = {
            'general': [
                "Can you explain how you approached this problem?",
                "Is there another way to check your answer?",
                "What part of the problem do you find most challenging?"
            ],
            'operation_confusion': [
                f"What does the {operation} symbol mean?",
                "How would you explain this operation to a friend?",
                "Can you draw a picture to show what this operation means?"
            ],
            'calculation': [
                "Can you break this problem into smaller steps?",
                "What strategy could you use to make this calculation easier?",
                "How could you check if your answer is reasonable?"
            ],
            'pythagorean_theorem': [
                "What is the relationship between the sides of a right triangle?",
                "How can you check if a triangle is a right triangle?",
                "Why do we square the values in the Pythagorean theorem?"
            ]
        }
        
        # Generate direct instruction based on error type and operation
        direct_instruction = {
            'general': [
                "Break down the problem into smaller steps.",
                "Use estimation to check if your answer is reasonable.",
                "Practice similar problems to reinforce the concept."
            ],
            'addition': [
                "When adding, combine all quantities together.",
                "For larger numbers, line up the digits by place value before adding.",
                "Remember to carry when the sum in a place value exceeds 9."
            ],
            'subtraction': [
                "In subtraction, you're finding the difference between two numbers.",
                "You may need to borrow from the tens place if you can't subtract directly.",
                "Check your work by adding the answer to the number you subtracted."
            ],
            'multiplication': [
                "Multiplication can be thought of as repeated addition.",
                "Memorize basic multiplication facts to build fluency.",
                "For larger numbers, break down the problem using the distributive property."
            ],
            'division': [
                "Division is the opposite of multiplication.",
                "Check division by multiplying your answer by the divisor.",
                "Remember that you cannot divide by zero."
            ],
            'pythagorean_theorem': [
                "Always remember the formula: a² + b² = c², where c is the hypotenuse.",
                "Square each side length first, then add the shorter sides, and compare to the square of the longest side.",
                "The formula only works for right triangles."
            ]
        }
        
        # Select appropriate feedback elements
        feedback["concept_explanation"] = concept_explanations.get(error_type, 
            "Understanding mathematical operations involves careful step-by-step work. Double-checking your calculations can help catch errors.")
        
        feedback["learning_strategy"] = learning_strategies.get(error_type,
            "Practice breaking down problems into smaller steps.")
        
        # Select appropriate Socratic questions
        if error_type in ['subtracted_instead_of_added', 'added_instead_of_subtracted', 
                         'multiplied_instead_of_added', 'added_instead_of_multiplied']:
            feedback["socratic_questions"] = socratic_questions['operation_confusion']
        elif error_type == 'pythagorean_theorem_error':
            feedback["socratic_questions"] = socratic_questions['pythagorean_theorem']
        elif error_type in ['off_by_one_high', 'off_by_one_low', 'place_value']:
            feedback["socratic_questions"] = socratic_questions['calculation']
        else:
            feedback["socratic_questions"] = socratic_questions['general']
        
        # Select appropriate direct instruction
        if operation == '+':
            feedback["direct_instruction"] = direct_instruction['addition']
        elif operation == '-':
            feedback["direct_instruction"] = direct_instruction['subtraction']
        elif operation == '*':
            feedback["direct_instruction"] = direct_instruction['multiplication']
        elif operation == '/':
            feedback["direct_instruction"] = direct_instruction['division']
        elif operation == 'pythagorean':
            feedback["direct_instruction"] = direct_instruction['pythagorean_theorem']
        else:
            feedback["direct_instruction"] = direct_instruction['general']
        
        return feedback
    
    def _generate_comprehensive_feedback(self, analysis_results, image_path):
        """
        Generate comprehensive feedback based on all problems
        
        Args:
            analysis_results: Results from problem analysis
            image_path: Path to the original image
            
        Returns:
            Comprehensive feedback data structure
        """
        # Create base feedback structure
        feedback = {
            "student_info": {
                "name": "Student",
                "grade_level": "Elementary"  # Default level
            },
            "assignment_info": {
                "title": "Math Homework",
                "date": "2025-05-03",  # Could use current date
                "total_problems": len(analysis_results),
                "total_correct": 0,  # Will update below
                "score_percentage": 0  # Will calculate below
            },
            "problems": [],
            "summary": {},
            "pedagogical_approach": {
                "teaching_style": "Multi-sensory",
                "differentiation_suggestions": []
            }
        }
        
        # Keep track of operations and error types
        correct_operations = set()
        incorrect_operations = set()
        error_types = set()
        
        # Calculate overall score - handle None values for is_correct
        total_correct = 0
        valid_problems = 0
        
        for problem in analysis_results:
            is_correct = problem.get('is_correct')
            if is_correct is not None:  # Only count problems with definite correctness
                valid_problems += 1
                if is_correct:
                    total_correct += 1
        
        feedback["assignment_info"]["total_correct"] = total_correct
        
        # Calculate percentage based on valid problems
        if valid_problems > 0:
            feedback["assignment_info"]["score_percentage"] = total_correct / valid_problems
        else:
            feedback["assignment_info"]["score_percentage"] = 0.0
        
        # Process each problem
        for analysis in analysis_results:
            problem_id = analysis.get('id', 0)
            question = analysis.get('text', '')
            parsed = analysis.get('parsed', {})
            
            # Extract operation
            operation = parsed.get('operation', '')
            
            # Track which operations are correct/incorrect
            is_correct = analysis.get('is_correct')
            if is_correct is True:
                if operation:
                    correct_operations.add(operation)
            elif is_correct is False:
                if operation:
                    incorrect_operations.add(operation)
                
                # Extract error types
                if 'error_type' in analysis:
                    error_types.add(analysis['error_type'])
            
            # Create problem feedback
            problem_feedback = {
                "id": problem_id,
                "text": question,
                "is_correct": is_correct,
                "parsed": parsed
            }
            
            # Add detailed feedback for incorrect problems
            if is_correct is False:
                # Generate specific feedback based on problem details
                feedback_details = self._generate_problem_feedback(analysis)
                problem_feedback["feedback"] = feedback_details
                
                # Add error details
                if 'error_type' in analysis:
                    problem_feedback["error_details"] = {
                        "type": analysis['error_type'],
                        "description": self._get_error_description(analysis['error_type']),
                        "common_mistake": analysis.get('error_type', 'unknown')
                    }
            
            feedback["problems"].append(problem_feedback)
        
        # Generate summary feedback
        feedback["summary"] = self._generate_summary_feedback(
            feedback["problems"], error_types, correct_operations, incorrect_operations
        )
        
        # Add differentiation suggestions
        feedback["pedagogical_approach"]["differentiation_suggestions"] = [
            "Use concrete manipulatives to build conceptual understanding",
            "Practice with number lines to visualize operations",
            "Incorporate games that reinforce basic math facts"
        ]
        
        return feedback
        
    def _get_error_description(self, error_type):
        """Get a human-readable description for the error type"""
        error_descriptions = {
            "off_by_one": "Result is off by 1 - check your counting",
            "place_value": "Error in place value alignment",
            "carrying": "Forgot to carry during addition",
            "borrowing": "Incorrect borrowing in subtraction",
            "addition_fact": "Incorrect basic addition fact",
            "subtraction_fact": "Incorrect basic subtraction fact",
            "multiplication_fact": "Incorrect basic multiplication fact",
            "division_fact": "Incorrect basic division fact",
            "negative_result": "Failed to get a negative result",
            "decimal_error": "Error with decimal calculation",
            "division_by_zero": "Cannot divide by zero",
            "calculation_error": "Calculation error in arithmetic",
            "unknown": "Unknown error type"
        }
        
        return error_descriptions.get(error_type, "Error in calculation")
    
    def _generate_summary_feedback(self, problems, error_types, correct_operations, incorrect_operations):
        """
        Generate summary feedback based on all problems
        
        Args:
            problems: All problem data
            error_types: Set of error types encountered
            correct_operations: Set of operations that were correct
            incorrect_operations: Set of operations that were incorrect
            
        Returns:
            Summary feedback structure
        """
        summary = {
            "strengths": [],
            "areas_for_improvement": [],
            "teacher_recommendations": [],
            "next_steps": []
        }
        
        # Identify strengths
        if correct_operations:
            strengths_ops = ", ".join(correct_operations)
            summary["strengths"].append(f"The student correctly used {strengths_ops} operation(s) in some problems.")
        
        if any(p["is_correct"] for p in problems):
            summary["strengths"].append("The student answered some problems correctly.")
        
        if not incorrect_operations:
            summary["strengths"].append("The student demonstrates good understanding of basic arithmetic operations.")
        
        # Identify areas for improvement
        if incorrect_operations:
            improve_ops = ", ".join(incorrect_operations)
            summary["areas_for_improvement"].append(f"Practice with {improve_ops} operation(s) is needed.")
        
        if "CALCULATION" in error_types:
            summary["areas_for_improvement"].append("Basic calculation skills need reinforcement.")
        
        if "PROCEDURAL" in error_types:
            summary["areas_for_improvement"].append("Understanding of mathematical procedures needs development.")
        
        if "CONCEPTUAL" in error_types:
            summary["areas_for_improvement"].append("Conceptual understanding of operations needs strengthening.")
        
        # Generate recommendations
        common_mistakes = [p.get("error_details", {}).get("common_mistake") for p in problems if not p.get("is_correct", False)]
        
        if "off_by_one_high" in common_mistakes or "off_by_one_low" in common_mistakes:
            summary["teacher_recommendations"].append("Practice careful counting forward and backward.")
            summary["teacher_recommendations"].append("Use number lines to visualize each step in counting.")
        
        if "multiplied_instead_of_added" in common_mistakes:
            summary["teacher_recommendations"].append("Review the meaning of operation symbols.")
            summary["teacher_recommendations"].append("Practice distinguishing between different operations.")
        
        if "+" in incorrect_operations:
            summary["teacher_recommendations"].append("Use manipulatives to build addition concepts.")
        
        if "-" in incorrect_operations:
            summary["teacher_recommendations"].append("Practice subtraction with concrete objects before moving to abstract problems.")
        
        # Add generic recommendations if specific ones are limited
        if len(summary["teacher_recommendations"]) < 2:
            summary["teacher_recommendations"].extend([
                "Practice counting forward from a number",
                "Use visual aids like number lines or manipulatives",
                "Work on number recognition and value understanding"
            ])
        
        # Next steps
        summary["next_steps"] = [
            "Review basic arithmetic facts",
            "Practice counting objects",
            "Build understanding of number values with manipulatives"
        ]
        
        return summary
    
    def _create_error_visualization(self, image_path, feedback, visualization_path):
        """
        Create a visualization of the errors on the original image
        
        Args:
            image_path: Path to the original image
            feedback: Feedback data structure
            visualization_path: Path to save the visualization
            
        Returns:
            Visualization image
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Add a padding area at the bottom for the feedback summary
        padding_height = 300  # Increased for more detailed feedback
        padded_vis = np.zeros((h + padding_height, w, 3), dtype=np.uint8)
        padded_vis[:h, :] = image
        padded_vis[h:, :] = (255, 255, 255)  # White background for the summary
        
        # Draw a dividing line
        cv2.line(padded_vis, (0, h), (w, h), (0, 0, 0), 2)
        
        # Add summary text
        score = feedback["assignment_info"]["score_percentage"] * 100
        total_problems = feedback["assignment_info"]["total_problems"]
        correct_problems = feedback["assignment_info"]["total_correct"]
        
        cv2.putText(padded_vis, f"Score: {score:.1f}% ({correct_problems}/{total_problems} correct)", 
                   (20, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add areas for improvement
        y_offset = h + 60
        cv2.putText(padded_vis, "Areas for improvement:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                   
        for i, area in enumerate(feedback["summary"]["areas_for_improvement"][:3]):
            cv2.putText(padded_vis, f"• {area}", 
                       (40, y_offset + (i+1)*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add conceptual explanation section
        concept_offset = y_offset + 100
        cv2.putText(padded_vis, "CONCEPTUAL FEEDBACK:", 
                   (20, concept_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add a sample conceptual feedback if available
        concept_text = ""
        for problem in feedback["problems"]:
            if not problem["is_correct"] and "feedback" in problem:
                concept_text = problem["feedback"].get("concept_explanation", "")
                if concept_text:
                    break
        
        if concept_text:
            # Word wrap for the concept text
            max_width = w - 40
            words = concept_text.split()
            lines = []
            current_line = words[0]
            
            for word in words[1:]:
                # Check if adding this word would exceed the max width
                test_line = current_line + " " + word
                test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
                
                if test_size < max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            lines.append(current_line)  # Add the last line
            
            # Draw the wrapped text
            for i, line in enumerate(lines):
                cv2.putText(padded_vis, line, 
                           (40, concept_offset + 25 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            cv2.putText(padded_vis, "Practice the concepts shown in your work.", 
                       (40, concept_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Mark errors on the original image
        operation_colors = {
            '+': (0, 255, 0),    # Green for addition
            '-': (0, 0, 255),    # Red for subtraction
            '×': (255, 0, 0),    # Blue for multiplication
            '/': (255, 165, 0),  # Orange for division
        }
        
        for problem in feedback["problems"]:
            if "bbox" in problem:
                # Get the bounding box and operation
                x, y, w_box, h_box = problem.get("bbox", (0, 0, 0, 0))
                operation = problem.get("parsed", {}).get("operation", "+")
                
                # Choose color based on operation type
                color = operation_colors.get(operation, (0, 0, 0))
                
                # Draw rectangle around the problem
                cv2.rectangle(padded_vis, (x, y), (x + w_box, y + h_box), color, 2)
                
                # Mark if correct or incorrect
                is_correct = problem.get("is_correct", None)
                
                if is_correct is True:
                    # Correct - green checkmark
                    cv2.putText(padded_vis, "✓", (x + w_box + 5, y + h_box//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif is_correct is False:
                    # Incorrect - red X
                    cv2.putText(padded_vis, "✗", (x + w_box + 5, y + h_box//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Add the correct answer nearby
                    if "expected_result" in problem:
                        correct_result = problem["expected_result"]
                        cv2.putText(padded_vis, f"Correct: {correct_result}", 
                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Add the problem identifier
                cv2.putText(padded_vis, f"#{problem['id']}", (x - 20, y + h_box//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite(visualization_path, padded_vis)
        
    def _create_visualization(self, image, problems, analysis_results):
        """
        Create a visualization of the analyzed math homework
        
        Args:
            image: Original image
            problems: Detected problems
            analysis_results: Analysis results
            
        Returns:
            Visualization image
        """
        # Make a copy of the image for visualization
        vis_image = image.copy()
        
        # Use different colors for correct and incorrect problems
        COLOR_CORRECT = (0, 255, 0)  # Green
        COLOR_INCORRECT = (0, 0, 255)  # Red
        COLOR_UNKNOWN = (255, 165, 0)  # Orange
        
        # Add detected problems to visualization
        for i, problem in enumerate(problems):
            if "bbox" not in problem:
                continue
                
            # Get the bounding box
            bbox = problem["bbox"]
            if len(bbox) == 4:  # [x, y, w, h] format
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:  # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = bbox
                
            # Find the corresponding analysis result
            analysis = None
            for result in analysis_results:
                if result.get("id") == problem.get("id"):
                    analysis = result
                    break
                    
            # Determine color based on correctness
            if analysis and analysis.get("is_correct") is not None:
                color = COLOR_CORRECT if analysis.get("is_correct") else COLOR_INCORRECT
            else:
                color = COLOR_UNKNOWN
                
            # Draw bounding box
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add problem ID
            cv2.putText(vis_image, f"#{problem.get('id', i+1)}", 
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add correctness indicator
            if analysis and analysis.get("is_correct") is not None:
                indicator = "✓" if analysis.get("is_correct") else "✗"
                cv2.putText(vis_image, indicator, 
                           (int(x2) + 10, int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add a title and legend
        title = "Math Homework Analysis"
        cv2.putText(vis_image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add legend
        legend_y = 60
        legend_x = 20
        
        cv2.putText(vis_image, "Correct:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(vis_image, (legend_x + 80, legend_y - 15), 
                     (legend_x + 110, legend_y + 5), COLOR_CORRECT, -1)
        
        legend_y += 30
        cv2.putText(vis_image, "Incorrect:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(vis_image, (legend_x + 80, legend_y - 15), 
                     (legend_x + 110, legend_y + 5), COLOR_INCORRECT, -1)
        
        legend_y += 30
        cv2.putText(vis_image, "Unknown:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(vis_image, (legend_x + 80, legend_y - 15), 
                     (legend_x + 110, legend_y + 5), COLOR_UNKNOWN, -1)
        
        return vis_image
    
    def track_performance(self, student_id, feedback):
        """
        Track student performance over time
        
        Args:
            student_id: Identifier for the student
            feedback: Feedback from the most recent assessment
            
        Returns:
            Updated performance metrics
        """
        # This is a placeholder for a more sophisticated tracking system
        # In a real implementation, this would store data in a database
        
        # Extract metrics from feedback
        total_problems = feedback["assignment_info"]["total_problems"]
        correct_problems = feedback["assignment_info"]["total_correct"]
        score = feedback["assignment_info"]["score_percentage"]
        
        # Store performance
        # In a real implementation, this would be stored in a database with timestamps
        
        return {
            "student_id": student_id,
            "date": "2025-05-03",
            "score": score,
            "total_problems": total_problems,
            "correct_problems": correct_problems,
            "areas_for_improvement": feedback["summary"]["areas_for_improvement"]
        }


# Example usage:
if __name__ == "__main__":
    # Initialize the integrated system
    system = MathRecognitionSystem()
    
    # Process a sample homework image
    # result = system.process_homework_image("path/to/image.jpg")
    
    # Track student performance
    # performance = system.track_performance("student123", result["feedback"])
