import json
import os
from pathlib import Path
import cv2
import numpy as np

# Try to use relative imports within the package
try:
    from math_analyzer.feedback_generator import MathFeedbackGenerator
    from math_analyzer.train_model import MathFeedbackTrainer
    from math_analyzer.paper_grading import PaperGradingSystem
    from math_analyzer.detector import MathExpressionDetector
    from math_analyzer.feedback_templates import get_error_feedback, get_correct_feedback, get_next_steps
except ImportError:
    # Fall back to local imports if the module structure isn't found
    from feedback_generator import MathFeedbackGenerator
    from train_model import MathFeedbackTrainer
    from paper_grading import PaperGradingSystem
    from detector import MathExpressionDetector
    from feedback_templates import get_error_feedback, get_correct_feedback, get_next_steps

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"  # put your YOLO model here

class MathImageAnalyzer:
    """Analyzes math images with teacher-like precision"""
    
    def __init__(self):
        # For simplicity, create mock objects if imports fail
        try:
            self.feedback_gen = MathFeedbackGenerator()
            self.grading_system = PaperGradingSystem(subject_area="math")
        except:
            # Create simple mock objects
            self.feedback_gen = type('', (), {})()
            self.grading_system = type('', (), {'subject_area': 'math'})()
        
    def analyze(self, image_path):
        """Main analysis function to process the math image"""
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get image dimensions for scaling
        img_height, img_width = image.shape[:2]
        
        # Calculate scale-dependent parameters for consistent visual output
        line_thickness = max(1, int(min(img_width, img_height) / 500))
        font_scale = max(0.4, min(img_width, img_height) / 1000)
        text_margin = int(min(img_width, img_height) / 60)
        
        # Create results structure
        analysis_results = {
            "image": str(image_path),
            "problems": [],
            "visual_feedback": {
                "marked_image_path": "",
                "error_locations": []
            }
        }
        
        # -------------------- detection --------------------
        detector = MathExpressionDetector(MODEL_PATH if MODEL_PATH.exists() else None)
        detected_boxes = detector.detect(image, image_path)
        
        # Create a clean copy for marking
        marked_image = image.copy()
        
        # Process each detected expression
        for box in detected_boxes:
            x1, y1, x2, y2, expr_text = box
            
            # Skip empty or unknown expressions
            if expr_text == "<unknown>":
                # Try to extract text from the region using our contour detection
                # (a simple placeholder - in a real system, this would be OCR)
                roi = image[y1:y2, x1:x2]
                if min(roi.shape[:2]) > 10:  # Skip tiny regions
                    # Simple feature: average darkness in region as a proxy for "has content"
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                    if np.mean(gray_roi) < 220:  # Assuming white background (0=black, 255=white)
                        # Use a placeholder expression based on region properties
                        w, h = x2-x1, y2-y1
                        expr_text = f"expression_at_{x1}_{y1}"
                    else:
                        continue  # Skip empty regions
                else:
                    continue  # Skip tiny regions
            
            # Analyze the expression
            result = self._analyze_expression(expr_text)
            if result:
                # Add visual feedback
                if result["errors"]:
                    # Mark incorrect expressions with red boxes - scale thickness with image size
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
                    
                    # Add error annotations with consistent size and position
                    for i, error in enumerate(result["errors"]):
                        error_text = f"{error['type']}"
                        cv2.putText(marked_image, error_text, 
                                  (x1, y1 - text_margin),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                  (0, 0, 255), line_thickness)
                    
                    # Store error location for reference
                    analysis_results["visual_feedback"]["error_locations"].append({
                        "expression": result["expression"],
                        "box": [x1, y1, x2, y2]
                    })
                else:
                    # Mark correct expressions with green checkmark - scaled appropriately
                    check_size = max(5, int(min(img_width, img_height) / 150))
                    check_x = x1 - check_size * 2
                    check_y = y1 + check_size
                    
                    cv2.circle(marked_image, (check_x, check_y), check_size, (0, 255, 0), line_thickness)
                    cv2.line(marked_image, 
                           (check_x - check_size/2, check_y), 
                           (check_x, check_y + check_size/2), 
                           (0, 255, 0), line_thickness)
                    cv2.line(marked_image, 
                           (check_x, check_y + check_size/2), 
                           (check_x + check_size, check_y - check_size/2), 
                           (0, 255, 0), line_thickness)
                
                # Store location
                result["location"] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                analysis_results["problems"].append(result)
        
        # Save the marked image
        output_image_path = Path(__file__).parent.parent / "results" / f"{Path(image_path).stem}_marked.jpg"
        cv2.imwrite(str(output_image_path), marked_image)
        analysis_results["visual_feedback"]["marked_image_path"] = str(output_image_path)
        
        # Save detailed analysis
        output_path = Path(__file__).parent.parent / "results" / f"{Path(image_path).stem}_detailed_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results
    
    def _analyze_expression(self, expr):
        """Analyze a math expression for errors and generate feedback"""
        # Skip non-mathematical expressions
        if not any(c.isdigit() for c in expr) and not any(c in expr for c in "+-*/=xy"):
            return None
            
        # Basic structure for the analysis result
        result = {
            "expression": expr,
            "errors": [],
            "feedback": {},
            "next_steps": {
                "practice_problems": [],
                "concepts_to_review": [],
                "suggested_resources": []
            }
        }
        
        # Check for calculation errors
        if "=" in expr:
            parts = expr.split("=")
            if len(parts) == 2:
                try:
                    # For expressions like x=3, don't mark as error
                    if parts[0].strip() in ['x', 'y', 'z'] and parts[1].strip().isdigit():
                        pass  # This is a solution statement, not an error
                    else:
                        # Try direct evaluation for simple expressions like 2+2=5
                        left = self._safe_eval(parts[0].strip())
                        right = self._safe_eval(parts[1].strip())
                        if left is not None and right is not None and left != right:
                            result["errors"].append({
                                "type": "CALCULATION",
                                "location": "result",
                                "severity": "high",
                                "confidence": 0.98
                            })
                except:
                    # If we can't evaluate, don't assume it's wrong
                    pass
        
        # Add notation errors for specific cases
        if "." in expr and not any(c.isalpha() for c in expr):
            result["errors"].append({
                "type": "NOTATION",
                "location": "usage of .",
                "severity": "medium",
                "confidence": 0.9
            })
        
        # Add procedural errors for specific patterns
        if "/" in expr and not expr.endswith("/"):
            result["errors"].append({
                "type": "PROCEDURAL",
                "location": "division operation",
                "severity": "medium",
                "confidence": 0.85
            })
        
        # Add conceptual errors for specific cases
        if expr.count("=") > 1:
            result["errors"].append({
                "type": "CONCEPTUAL",
                "location": "multiple equals signs",
                "severity": "high",
                "confidence": 0.9
            })
            
        # Generate appropriate feedback based on errors
        if result["errors"]:
            # Use our enhanced templates for different teaching styles
            error_types = [error["type"] for error in result["errors"]]
            
            # Get feedback for each teaching style
            teaching_styles = ["socratic", "direct", "growth_mindset", "constructivist", "inquiry_based"]
            result["feedback"] = {}
            
            # Get appropriate correct answer where possible
            correct_result = None
            if "CALCULATION" in error_types and "=" in expr:
                try:
                    parts = expr.split("=")
                    correct_result = self._safe_eval(parts[0].strip())
                except:
                    pass
                    
            # Generate feedback for each teaching style
            for style in teaching_styles:
                # Use the first error type to determine primary feedback
                primary_error = result["errors"][0]["type"]
                feedback = get_error_feedback(
                    primary_error, style, expr, 
                    correct_result=f"{correct_result}" if correct_result is not None else "the correct result"
                )
                result["feedback"][style] = feedback
            
            # Add next steps from our enhanced templates
            result["next_steps"] = get_next_steps(error_types)
        else:
            # Use enhanced templates for correct answers
            teaching_styles = ["socratic", "direct", "growth_mindset", "constructivist", "inquiry_based"]
            result["feedback"] = {style: get_correct_feedback(style, expr) for style in teaching_styles}
            result["next_steps"] = get_next_steps([])  # Empty list means correct
            
        return result
        
    def _safe_eval(self, expr_str):
        """Safely evaluate a mathematical expression string."""
        # Only allow basic operations and numbers for security
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expr_str):
            return None
            
        try:
            # Use eval with limitations for mathematical expressions only
            # This is relatively safe since we filtered the input
            return eval(expr_str)
        except:
            return None

def analyze_math_image(image_path):
    """Analyze a math image and generate detailed feedback"""
    analyzer = MathImageAnalyzer()
    return analyzer.analyze(image_path)

if __name__ == "__main__":
    import sys
    
    # Test with specified image or default to math8.jpeg
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = Path(__file__).parent.parent / "test_images" / "math8.jpeg"
        
    result = analyze_math_image(image_path)
    print(json.dumps(result, indent=2))
