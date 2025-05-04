import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import math
import json
from enum import Enum

@dataclass
class ErrorLocation:
    """Represents the precise location of an error in student work"""
    # Text-based location
    line_number: int = 0
    start_char: int = 0
    end_char: int = 0
    error_text: str = ""
    
    # Coordinate-based location (normalized 0-1 range)
    page: int = 1
    top_left_x: float = 0.0
    top_left_y: float = 0.0
    bottom_right_x: float = 0.0
    bottom_right_y: float = 0.0
    
    # Error details
    error_type: str = ""
    correction: str = ""
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "text_position": {
                "line": self.line_number,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "error_text": self.error_text
            },
            "box_position": {
                "page": self.page,
                "normalized_coords": [
                    self.top_left_x, self.top_left_y, 
                    self.bottom_right_x, self.bottom_right_y
                ],
                # Add pixel coordinates when they're calculated for a specific image
            },
            "error_details": {
                "type": self.error_type,
                "correction": self.correction,
                "explanation": self.explanation
            }
        }

class ErrorType(Enum):
    """Types of mathematical errors"""
    ARITHMETIC = "arithmetic_error"  # Basic calculation mistakes
    ALGEBRAIC = "algebraic_error"    # Issues with algebraic manipulation
    SIGN = "sign_error"              # Mistakes with positive/negative signs
    PROCEDURAL = "procedural_error"  # Wrong steps or order of operations
    CONCEPTUAL = "conceptual_error"  # Fundamental misunderstandings
    MISSING_STEP = "missing_step"    # Skipped necessary steps
    NOTATION = "notation_error"      # Incorrect mathematical notation
    FRACTION = "fraction_error"      # Errors in fraction handling
    EXPONENT = "exponent_error"      # Issues with powers and exponents
    DISTRIBUTION = "distribution_error"  # Errors in distributing terms
    VARIABLE_SUBSTITUTION = "variable_substitution_error"  # Errors in substituting values
    FACTORING = "factoring_error"  # Errors in factoring expressions

class ErrorLocalizationEngine:
    """Engine for localizing and marking errors in student math work"""
    
    def __init__(self):
        # Regular expressions for matching common mathematical patterns
        self.patterns = {
            "equation": r'([^=]*)=(.*)',  # Matches equations with equals sign
            "expression": r'([0-9]+|[a-z])\s*([+\-*/^])\s*([0-9]+|[a-z])',  # Basic operations
            "term": r'([+-]?[0-9]*[a-z]?(?:\^[0-9]+)?)',  # Individual terms
            "fraction": r'\\frac\{([^}]*)\}\{([^}]*)\}|(\d+)/(\d+)',  # Fractions in various formats
        }
    
    def analyze_line_by_line(self, student_work: str, correct_solution: str) -> List[ErrorLocation]:
        """
        Analyze student work line by line to identify and localize errors
        
        Args:
            student_work: Multi-line string containing student's solution steps
            correct_solution: The correct solution steps or final answer
            
        Returns:
            List of ErrorLocation objects identifying where errors occur
        """
        # Split into lines for analysis
        student_lines = student_work.strip().split('\n')
        
        # Initialize error locations list
        error_locations = []
        
        # Track variables and their values through the solution
        variable_values = {}
        previous_expressions = []
        
        # Analyze each line
        for line_idx, line in enumerate(student_lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is an equation
            if '=' in line:
                left, right = line.split('=', 1)
                left = left.strip()
                right = right.strip()
                
                # Check for sign errors when moving terms
                if line_idx > 0 and '=' in student_lines[line_idx-1]:
                    prev_left, prev_right = student_lines[line_idx-1].split('=', 1)
                    prev_left = prev_left.strip()
                    prev_right = prev_right.strip()
                    
                    # Check if a term moved from left to right or right to left
                    sign_error = self._check_sign_error(prev_left, prev_right, left, right)
                    if sign_error:
                        term, position, correction = sign_error
                        
                        # Calculate character positions
                        start_char = line.find(term)
                        end_char = start_char + len(term)
                        
                        # Create error location
                        error_loc = ErrorLocation(
                            line_number=line_idx + 1,  # 1-indexed for user-facing content
                            start_char=start_char,
                            end_char=end_char,
                            error_text=term,
                            error_type=ErrorType.SIGN.value,
                            correction=correction,
                            explanation="When moving a term to the other side of the equation, the sign must be changed."
                        )
                        error_locations.append(error_loc)
                
                # Check for arithmetic errors
                arithmetic_error = self._check_arithmetic_error(left, right)
                if arithmetic_error:
                    term, correction, explanation = arithmetic_error
                    
                    # Calculate character positions
                    if term in left:
                        side = left
                        position = "left"
                    else:
                        side = right
                        position = "right"
                        
                    start_char = line.find(term)
                    end_char = start_char + len(term)
                    
                    # Create error location
                    error_loc = ErrorLocation(
                        line_number=line_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        error_text=term,
                        error_type=ErrorType.ARITHMETIC.value,
                        correction=correction,
                        explanation=explanation
                    )
                    error_locations.append(error_loc)
                    
                # Check for other types of errors
                # Distribution errors: Check if a term is incorrectly distributed
                distribution_error = self._check_distribution_error(left, right)
                if distribution_error:
                    term, correction, explanation = distribution_error
                    
                    # Calculate character positions
                    start_char = line.find(term)
                    end_char = start_char + len(term)
                    
                    # Create error location
                    error_loc = ErrorLocation(
                        line_number=line_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        error_text=term,
                        error_type=ErrorType.DISTRIBUTION.value,
                        correction=correction,
                        explanation=explanation
                    )
                    error_locations.append(error_loc)
                
                # Exponent errors: Check for incorrect application of exponent rules
                exponent_error = self._check_exponent_error(left, right)
                if exponent_error:
                    term, correction, explanation = exponent_error
                    
                    # Calculate character positions
                    start_char = line.find(term)
                    end_char = start_char + len(term)
                    
                    # Create error location
                    error_loc = ErrorLocation(
                        line_number=line_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        error_text=term,
                        error_type=ErrorType.EXPONENT.value,
                        correction=correction,
                        explanation=explanation
                    )
                    error_locations.append(error_loc)
        
                # Variable substitution errors: Check for incorrect substitution of values
                substitution_error = self._check_substitution_error(line, variable_values)
                if substitution_error:
                    term, correction, explanation = substitution_error
                    
                    # Calculate character positions
                    start_char = line.find(term)
                    end_char = start_char + len(term)
                    
                    # Create error location
                    error_loc = ErrorLocation(
                        line_number=line_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        error_text=term,
                        error_type=ErrorType.VARIABLE_SUBSTITUTION.value,
                        correction=correction,
                        explanation=explanation
                    )
                    error_locations.append(error_loc)
        
                # Factoring errors: Check for incorrect factorization of expressions
                factoring_error = self._check_factoring_error(left, right)
                if factoring_error:
                    term, correction, explanation = factoring_error
                    
                    # Calculate character positions
                    start_char = line.find(term)
                    end_char = start_char + len(term)
                    
                    # Create error location
                    error_loc = ErrorLocation(
                        line_number=line_idx + 1,
                        start_char=start_char,
                        end_char=end_char,
                        error_text=term,
                        error_type=ErrorType.FACTORING.value,
                        correction=correction,
                        explanation=explanation
                    )
                    error_locations.append(error_loc)
        
                # Save this expression for future comparisons
                previous_expressions.append(line)
        
        return error_locations
    
    def localize_errors_in_image(self, image: np.ndarray, 
                               student_work: str, 
                               error_locations: List[ErrorLocation]) -> List[ErrorLocation]:
        """
        Convert text-based error locations to pixel coordinates in the image
        
        Args:
            image: Image containing the student work
            student_work: Text of the student work (from OCR)
            error_locations: List of ErrorLocation objects with text positions
            
        Returns:
            Updated list of ErrorLocation objects with pixel coordinates
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Split student work into lines
        lines = student_work.strip().split('\n')
        
        # Detect lines in the image
        # This is a simplified approach - a production system would use more
        # sophisticated line detection based on horizontal projections
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
                                     
        # Find horizontal lines using projection profile
        h_projection = np.sum(binary, axis=1)
        
        # Detect significant lines in the projection
        line_positions = []
        in_line = False
        start_y = 0
        threshold = np.mean(h_projection) * 0.5
        
        for y in range(len(h_projection)):
            if not in_line and h_projection[y] > threshold:
                # Start of line
                in_line = True
                start_y = y
            elif in_line and (h_projection[y] < threshold or y == len(h_projection) - 1):
                # End of line
                line_positions.append((start_y, y))
                in_line = False
        
        # Update coordinate-based positions for each error
        for error_loc in error_locations:
            # Skip if no valid line number
            if error_loc.line_number <= 0 or error_loc.line_number > len(line_positions):
                continue
                
            # Get the line position in the image
            line_idx = error_loc.line_number - 1  # Convert from 1-indexed to 0-indexed
            if line_idx < len(line_positions):
                y_start, y_end = line_positions[line_idx]
                
                # Calculate approximate horizontal position based on character position
                # This is an estimation based on average character width
                line = lines[line_idx] if line_idx < len(lines) else ""
                avg_char_width = width / max(len(line), 1)
                
                x_start = error_loc.start_char * avg_char_width
                x_end = error_loc.end_char * avg_char_width
                
                # Ensure coordinates are within image bounds
                x_start = max(0, min(x_start, width-1))
                x_end = max(0, min(x_end, width-1))
                
                # Update with normalized coordinates (0-1 range)
                error_loc.top_left_x = float(x_start / width)
                error_loc.top_left_y = float(y_start / height)
                error_loc.bottom_right_x = float(x_end / width)
                error_loc.bottom_right_y = float(y_end / height)
        
        return error_locations
    
    def mark_errors_on_image(self, image: np.ndarray, 
                           error_locations: List[ErrorLocation],
                           box_color=(0, 0, 255),  # Red by default
                           box_thickness=2) -> np.ndarray:
        """
        Draw error boxes on the image at the specified locations
        
        Args:
            image: Input image
            error_locations: List of ErrorLocation objects with coordinate info
            box_color: Color of the error box (BGR format)
            box_thickness: Thickness of the error box lines
            
        Returns:
            Image with error boxes drawn
        """
        # Make a copy to avoid modifying the original
        marked_image = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Draw boxes for each error
        for i, error_loc in enumerate(error_locations):
            # Convert normalized coordinates to pixel values
            x1 = int(error_loc.top_left_x * width)
            y1 = int(error_loc.top_left_y * height)
            x2 = int(error_loc.bottom_right_x * width)
            y2 = int(error_loc.bottom_right_y * height)
            
            # Draw the rectangle
            cv2.rectangle(marked_image, (x1, y1), (x2, y2), box_color, box_thickness)
            
            # Add error number for reference
            cv2.putText(marked_image, f"#{i+1}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        return marked_image
        
    def _check_sign_error(self, prev_left, prev_right, curr_left, curr_right):
        """
        Check for sign errors when moving terms between sides of an equation
        
        Args:
            prev_left, prev_right: Left and right sides of previous equation
            curr_left, curr_right: Left and right sides of current equation
            
        Returns:
            Tuple of (term with error, position, correction) or None if no error
        """
        # Extract terms from each side
        prev_left_terms = self._extract_terms(prev_left)
        prev_right_terms = self._extract_terms(prev_right)
        curr_left_terms = self._extract_terms(curr_left)
        curr_right_terms = self._extract_terms(curr_right)
        
        # Check for terms that moved from left to right
        for term in prev_left_terms:
            if term not in curr_left_terms and term in curr_right_terms:
                # Term moved to right side - sign should change
                # In this simplified example, we'll just look for the term with opposite sign
                opposite_term = self._get_opposite_sign(term)
                if opposite_term not in curr_right_terms:
                    return (term, "right", opposite_term)
        
        # Check for terms that moved from right to left
        for term in prev_right_terms:
            if term not in curr_right_terms and term in curr_left_terms:
                # Term moved to left side - sign should change
                opposite_term = self._get_opposite_sign(term)
                if opposite_term not in curr_left_terms:
                    return (term, "left", opposite_term)
        
        return None
        
    def _extract_terms(self, expression):
        """Extract individual terms from an expression"""
        # Very simple term extraction - split by +/- and keep the sign
        terms = []
        expr = expression.replace(" ", "")
        
        # Add + if expression doesn't start with a sign
        if not expr.startswith('+') and not expr.startswith('-'):
            expr = '+' + expr
            
        # Find all terms with their signs
        matches = re.finditer(r'([+\-][^+\-]*)', expr)
        for match in matches:
            terms.append(match.group(1))
            
        return terms
        
    def _get_opposite_sign(self, term):
        """Get the term with opposite sign"""
        if term.startswith('+'):
            return '-' + term[1:]
        elif term.startswith('-'):
            return '+' + term[1:]
        else:
            # If no explicit sign, it's implicitly positive
            return '-' + term
            
    def _check_arithmetic_error(self, left, right):
        """
        Check for arithmetic errors in an equation
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Clean strings and convert to evaluable expressions when possible
        left = left.strip()
        right = right.strip()
        
        # Check for basic arithmetic operations
        # Addition: e.g., "1 + 1 = 3"
        addition_match_left = re.search(r'(\d+)\s*\+\s*(\d+)', left)
        if addition_match_left:
            num1 = int(addition_match_left.group(1))
            num2 = int(addition_match_left.group(2))
            correct_sum = num1 + num2
            
            # Check if the right side contains an incorrect value
            try:
                provided_result = int(right)
                if provided_result != correct_sum:
                    error_term = f"{num1} + {num2} = {provided_result}"
                    correction = f"{num1} + {num2} = {correct_sum}"
                    explanation = f"The sum of {num1} and {num2} should be {correct_sum}, not {provided_result}."
                    return (error_term, correction, explanation)
            except ValueError:
                pass
        
        # Subtraction: e.g., "1 - 1 = 4"
        subtraction_match_left = re.search(r'(\d+)\s*\-\s*(\d+)', left)
        if subtraction_match_left:
            num1 = int(subtraction_match_left.group(1))
            num2 = int(subtraction_match_left.group(2))
            correct_diff = num1 - num2
            
            # Check if the right side contains an incorrect value
            try:
                provided_result = int(right)
                if provided_result != correct_diff:
                    error_term = f"{num1} - {num2} = {provided_result}"
                    correction = f"{num1} - {num2} = {correct_diff}"
                    explanation = f"The difference of {num1} and {num2} should be {correct_diff}, not {provided_result}."
                    return (error_term, correction, explanation)
            except ValueError:
                pass
        
        # Division by zero: e.g., "1/0 = 0"
        division_match_left = re.search(r'(\d+)\s*\/\s*(\d+)', left)
        if division_match_left:
            num1 = int(division_match_left.group(1))
            num2 = int(division_match_left.group(2))
            
            # Check for division by zero
            if num2 == 0:
                # Any result for division by zero is wrong
                error_term = f"{num1}/{num2} = {right}"
                correction = f"{num1}/{num2} = undefined"
                explanation = f"Division by zero is undefined in standard arithmetic."
                return (error_term, correction, explanation)
            
            # Normal division
            correct_div = num1 / num2
            try:
                provided_result = float(right)
                if abs(provided_result - correct_div) > 0.0001:  # Allow for float comparison
                    error_term = f"{num1}/{num2} = {right}"
                    correction = f"{num1}/{num2} = {correct_div}"
                    explanation = f"The result of {num1} divided by {num2} should be {correct_div}, not {provided_result}."
                    return (error_term, correction, explanation)
            except ValueError:
                pass
                
        # Check for more complex expressions or variables would go here
        return None
        
    def _check_distribution_error(self, left, right):
        """
        Check for distribution errors (e.g., incorrect application of distributive property)
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Check for common distribution errors
        # Example: 2(x+3) = 2x + 3 instead of 2x + 6
        dist_match = re.search(r'(\d+)\(([^)]+)\)', left)
        if dist_match:
            multiplier = int(dist_match.group(1))
            inside_expr = dist_match.group(2)
            
            # Split inside expression by + or -
            inside_terms = re.findall(r'([+-]?\d*[a-z]?)', inside_expr)
            inside_terms = [term for term in inside_terms if term]  # Remove empty matches
            
            # Correct distribution: multiply each term by the multiplier
            correctly_distributed = []
            for term in inside_terms:
                if "+" in term or "-" in term:  # Skip operators
                    correctly_distributed.append(term)
                    continue
                    
                # Check if the term has a coefficient
                coef_match = re.search(r'^([+-]?\d*)([a-z]?)$', term)
                if coef_match:
                    coef = coef_match.group(1)
                    var = coef_match.group(2)
                    
                    if coef == "" or coef == "+":
                        new_coef = multiplier
                    elif coef == "-":
                        new_coef = -multiplier
                    else:
                        new_coef = int(coef) * multiplier
                        
                    correctly_distributed.append(f"{new_coef}{var}")
            
            correct_right = " + ".join(correctly_distributed)
            
            # If the right side doesn't match the correct distribution, it's an error
            if right != correct_right:
                error_term = f"{multiplier}({inside_expr}) = {right}"
                correction = f"{multiplier}({inside_expr}) = {correct_right}"
                explanation = f"When distributing {multiplier} across {inside_expr}, each term must be multiplied by {multiplier}."
                return (error_term, correction, explanation)
                
        return None
        
    def _check_exponent_error(self, left, right):
        """
        Check for exponent errors (e.g., incorrect application of exponent rules)
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Check for basic exponent errors
        # Example: x^2 * x^3 = x^5 (correct) vs x^2 * x^3 = x^6 (incorrect)
        mult_exponents = re.search(r'([a-z])\^(\d+)\s*\*\s*\1\^(\d+)', left)
        if mult_exponents:
            variable = mult_exponents.group(1)
            exp1 = int(mult_exponents.group(2))
            exp2 = int(mult_exponents.group(3))
            correct_exp = exp1 + exp2
            
            # Check if the result has the correct exponent
            result_pattern = re.escape(variable) + r'\^(\d+)'
            result_match = re.search(result_pattern, right)
            
            if result_match:
                result_exp = int(result_match.group(1))
                if result_exp != correct_exp:
                    error_term = f"{variable}^{exp1} * {variable}^{exp2} = {variable}^{result_exp}"
                    correction = f"{variable}^{exp1} * {variable}^{exp2} = {variable}^{correct_exp}"
                    explanation = f"When multiplying terms with the same base, add the exponents: {variable}^{exp1} * {variable}^{exp2} = {variable}^{exp1+exp2}."
                    return (error_term, correction, explanation)
        
        # Example: (x^2)^3 = x^6 (correct) vs (x^2)^3 = x^5 (incorrect)
        power_to_power = re.search(r'\(([a-z])\^(\d+)\)\^(\d+)', left)
        if power_to_power:
            variable = power_to_power.group(1)
            inner_exp = int(power_to_power.group(2))
            outer_exp = int(power_to_power.group(3))
            correct_exp = inner_exp * outer_exp
            
            # Check if the result has the correct exponent
            result_pattern = re.escape(variable) + r'\^(\d+)'
            result_match = re.search(result_pattern, right)
            
            if result_match:
                result_exp = int(result_match.group(1))
                if result_exp != correct_exp:
                    error_term = f"({variable}^{inner_exp})^{outer_exp} = {variable}^{result_exp}"
                    correction = f"({variable}^{inner_exp})^{outer_exp} = {variable}^{correct_exp}"
                    explanation = f"When raising a power to another power, multiply the exponents: ({variable}^{inner_exp})^{outer_exp} = {variable}^{inner_exp*outer_exp}."
                    return (error_term, correction, explanation)
        
        # Example: x^2 / x = x (correct) vs x^2 / x = x^2 (incorrect)
        div_exponents = re.search(r'([a-z])\^(\d+)\s*\/\s*\1(?:\^(\d+))?', left)
        if div_exponents:
            variable = div_exponents.group(1)
            numerator_exp = int(div_exponents.group(2))
            denominator_exp = 1 if div_exponents.group(3) is None else int(div_exponents.group(3))
            correct_exp = numerator_exp - denominator_exp
            
            # Special case: if result is x^0, it should simplify to 1
            correct_result = "1" if correct_exp == 0 else f"{variable}^{correct_exp}" if correct_exp > 1 else variable
            
            # Check if the right side matches the correct result
            if right != correct_result:
                # Construct the original expression more accurately
                original_denominator = variable if div_exponents.group(3) is None else f"{variable}^{denominator_exp}"
                error_term = f"{variable}^{numerator_exp} / {original_denominator} = {right}"
                correction = f"{variable}^{numerator_exp} / {original_denominator} = {correct_result}"
                explanation = f"When dividing terms with the same base, subtract the exponents: {variable}^{numerator_exp} / {variable}^{denominator_exp} = {variable}^{numerator_exp-denominator_exp}."
                return (error_term, correction, explanation)
                
        return None
        
    def _check_substitution_error(self, expression, variable_values):
        """
        Check for variable substitution errors (e.g., incorrect substitution of values)
        
        Args:
            expression: Expression to check for substitution errors
            variable_values: Dictionary of variable values to substitute
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Only proceed if we have variable values to check
        if not variable_values:
            return None
            
        # Find all variables in the expression
        variables = set(re.findall(r'[a-z]', expression))
        
        # Only variables with known values
        known_variables = variables.intersection(variable_values.keys())
        
        if not known_variables:
            return None
            
        # Check if the expression contains variable substitutions
        for var in known_variables:
            value = variable_values[var]
            
            # Check for direct substitution errors
            # Example: If x = 2, then "x + 3 = 6" is correct but "x + 3 = 7" is wrong
            var_pattern = r'(^|\s|\(|\+|\-|\*|\/)' + re.escape(var) + r'(\s|\)|\+|\-|\*|\/|$)'
            if re.search(var_pattern, expression):
                # Try to extract a simple arithmetic operation with this variable
                operation_pattern = r'(' + re.escape(var) + r'\s*[\+\-\*\/]\s*\d+|\d+\s*[\+\-\*\/]\s*' + re.escape(var) + r')\s*=\s*(\d+)'
                match = re.search(operation_pattern, expression)
                
                if match:
                    left_side = match.group(1)
                    provided_result = int(match.group(2))
                    
                    # Evaluate the correct result
                    try:
                        # Replace variable with its value and evaluate
                        evaluated_expr = left_side.replace(var, str(value))
                        correct_result = eval(evaluated_expr)
                        
                        if int(correct_result) != provided_result:
                            error_term = f"{left_side} = {provided_result}"
                            correction = f"{left_side} = {int(correct_result)}"
                            explanation = f"When substituting {var} = {value}, {left_side} equals {int(correct_result)}, not {provided_result}."
                            return (error_term, correction, explanation)
                    except:
                        # If evaluation fails, skip this check
                        pass
                
                # Check for direct value substitution
                # Example: If x = 2, then x = 3 is wrong
                equals_pattern = r'^' + re.escape(var) + r'\s*=\s*(\d+)$'
                match = re.search(equals_pattern, expression)
                
                if match:
                    provided_value = int(match.group(1))
                    if provided_value != value:
                        error_term = f"{var} = {provided_value}"
                        correction = f"{var} = {value}"
                        explanation = f"The value of {var} should be {value}, not {provided_value}."
                        return (error_term, correction, explanation)
                        
        return None
        
    def _check_factoring_error(self, left, right):
        """
        Check for factoring errors (e.g., incorrect factorization of expressions)
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Check for common factoring errors
        
        # Case 1: Factoring quadratics (x² + bx + c)
        # Example: x² + 5x + 6 = (x+2)(x+3) is correct, but x² + 5x + 6 = (x+1)(x+6) is wrong
        quadratic_match = re.search(r'([a-z])\^2\s*([+-]\s*\d+[a-z]?)?\s*([+-]\s*\d+)', left)
        if quadratic_match:
            variable = quadratic_match.group(1)
            
            # Extract b coefficient (middle term)
            b_coef = 0
            if quadratic_match.group(2):
                b_term = quadratic_match.group(2).replace(" ", "")
                b_match = re.search(r'([+-])(\d*)' + re.escape(variable), b_term)
                if b_match:
                    sign = -1 if b_match.group(1) == '-' else 1
                    value = 1 if b_match.group(2) == '' else int(b_match.group(2))
                    b_coef = sign * value
            
            # Extract c coefficient (constant term)
            c_coef = 0
            if quadratic_match.group(3):
                c_term = quadratic_match.group(3).replace(" ", "")
                c_match = re.search(r'([+-])(\d+)', c_term)
                if c_match:
                    sign = -1 if c_match.group(1) == '-' else 1
                    c_coef = sign * int(c_match.group(2))
            
            # Now check factorization on right side
            # Pattern for (x+p)(x+q)
            factorization_match = re.search(r'\(' + re.escape(variable) + r'([+-]\d+)\)\(' + re.escape(variable) + r'([+-]\d+)\)', right)
            if factorization_match:
                # Extract the factors
                factor1 = int(factorization_match.group(1).replace("+", ""))
                factor2 = int(factorization_match.group(2).replace("+", ""))
                
                # Check if these factors multiply to give c_coef and sum to give b_coef
                if factor1 * factor2 != c_coef or factor1 + factor2 != b_coef:
                    # Calculate the correct factors
                    # Find factors of c_coef that sum to b_coef
                    correct_factors = []
                    for i in range(-abs(c_coef), abs(c_coef) + 1):
                        if i == 0:
                            continue
                        if c_coef % i == 0:
                            j = c_coef // i
                            if i + j == b_coef:
                                correct_factors = [i, j]
                                break
                    
                    if correct_factors:
                        p, q = correct_factors
                        error_term = f"{left} = {right}"
                        correction = f"{left} = ({variable}{'+' if p >= 0 else ''}{p})({variable}{'+' if q >= 0 else ''}{q})"
                        explanation = f"When factoring {left}, the factors should multiply to {c_coef} and sum to {b_coef}."
                        return (error_term, correction, explanation)
            
        # Case 2: Factoring difference of squares
        # Example: x² - 4 = (x+2)(x-2) is correct, but x² - 4 = (x+4)(x-1) is wrong
        diff_squares_match = re.search(r'([a-z])\^2\s*-\s*(\d+)', left)
        if diff_squares_match:
            variable = diff_squares_match.group(1)
            constant = int(diff_squares_match.group(2))
            
            # Check if the constant is a perfect square
            root = math.isqrt(constant)
            if root * root == constant:
                # It's a perfect square, so should factor as (x+a)(x-a)
                correct_factorization = f"({variable}+{root})({variable}-{root})"
                
                # Check if right side matches the correct factorization
                if right != correct_factorization:
                    # Try to match the actual factorization
                    factorization_match = re.search(r'\(' + re.escape(variable) + r'([+-]\d+)\)\(' + re.escape(variable) + r'([+-]\d+)\)', right)
                    if factorization_match:
                        error_term = f"{left} = {right}"
                        correction = f"{left} = {correct_factorization}"
                        explanation = f"The difference of squares {left} should be factored as {correct_factorization}."
                        return (error_term, correction, explanation)
        
        # Case 3: Factoring common terms
        # Example: 2x + 4 = 2(x+2) is correct, but 2x + 4 = 2(x+1) is wrong
        common_factor_match = re.search(r'(\d+)([a-z])\s*([+-])\s*(\d+)', left)
        if common_factor_match:
            coef = int(common_factor_match.group(1))
            variable = common_factor_match.group(2)
            operator = common_factor_match.group(3)
            constant = int(common_factor_match.group(4))
            
            # Check for common factor
            gcd_value = math.gcd(coef, constant)
            if gcd_value > 1:
                # There's a common factor to extract
                factored_coef = coef // gcd_value
                factored_constant = constant // gcd_value
                correct_factorization = f"{gcd_value}({variable}{operator}{factored_constant})"
                
                # Check if right side matches
                if right != correct_factorization:
                    # Try to match the actual factorization pattern
                    factorization_match = re.search(r'(\d+)\(([a-z])([+-]\d+)\)', right)
                    if factorization_match:
                        error_term = f"{left} = {right}"
                        correction = f"{left} = {correct_factorization}"
                        explanation = f"When factoring {left}, extract the greatest common factor {gcd_value}."
                        return (error_term, correction, explanation)
                
        return None

class ErrorDetectionResult:
    """Container for error detection results"""
    
    def __init__(self, original_work: str, errors: List[ErrorLocation], 
                original_image: Optional[np.ndarray] = None):
        self.original_work = original_work
        self.errors = errors
        self.original_image = original_image
        self.marked_image = None
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "original_work": self.original_work,
            "errors": [error.to_dict() for error in self.errors],
            "error_count": len(self.errors)
        }
        
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
        
    def get_marked_image(self) -> Optional[np.ndarray]:
        """Get image with errors marked"""
        return self.marked_image

class MathErrorDetector:
    """Main class for detecting and localizing errors in math work"""
    
    def __init__(self):
        self.localization_engine = ErrorLocalizationEngine()
        
    def detect_errors(self, student_work: str, correct_solution: str, 
                    student_image: Optional[np.ndarray] = None) -> ErrorDetectionResult:
        """
        Detect and localize errors in student math work
        
        Args:
            student_work: Text of student's solution
            correct_solution: Correct solution
            student_image: Optional image of student work
            
        Returns:
            ErrorDetectionResult with detected errors
        """
        # Analyze work to find errors
        error_locations = self.localization_engine.analyze_line_by_line(
            student_work, correct_solution)
            
        # Create result container
        result = ErrorDetectionResult(student_work, error_locations, student_image)
        
        # If an image was provided, localize errors on it
        if student_image is not None and len(error_locations) > 0:
            # Convert text positions to image coordinates
            updated_locations = self.localization_engine.localize_errors_in_image(
                student_image, student_work, error_locations)
                
            # Mark errors on the image
            marked_image = self.localization_engine.mark_errors_on_image(
                student_image, updated_locations)
                
            # Update result with marked image
            result.marked_image = marked_image
            
        return result
        
    def integrate_with_feedback(self, error_result: ErrorDetectionResult, 
                              feedback_text: str) -> str:
        """
        Integrate error locations into feedback text
        
        Args:
            error_result: Error detection result
            feedback_text: Original feedback text
            
        Returns:
            Enhanced feedback text with error location references
        """
        # Find where we need to add error markers in the feedback
        enhanced_feedback = feedback_text
        
        # Simple integration - in reality, this would be more sophisticated
        for i, error in enumerate(error_result.errors):
            error_marker = f"Error #{i+1}: Line {error.line_number}, \"{error.error_text}\""
            
            # Add error position information if available
            if (error.top_left_x > 0 or error.top_left_y > 0 or 
                error.bottom_right_x > 0 or error.bottom_right_y > 0):
                normalized_coords = [
                    error.top_left_x, error.top_left_y,
                    error.bottom_right_x, error.bottom_right_y
                ]
                error_marker += f" (Position: {normalized_coords})"
                
            # Add to feedback
            if "Errors detected:" in enhanced_feedback:
                # Add after the "Errors detected:" line
                parts = enhanced_feedback.split("Errors detected:", 1)
                enhanced_feedback = parts[0] + "Errors detected:\n\n" + error_marker + "\n" + parts[1].lstrip()
            else:
                # Add at the beginning
                enhanced_feedback = "Errors detected:\n\n" + error_marker + "\n\n" + enhanced_feedback
                
        return enhanced_feedback
