import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import Eq, solve
import re
from collections import defaultdict

class AdvancedMathAnalyzer:
    def __init__(self):
        self.error_types = {
            'CALCULATION': 'Calculation Error',
            'PROCEDURAL': 'Procedural Error',
            'CONCEPTUAL': 'Conceptual Error',
            'SYNTAX': 'Syntax Error',
            'ALGEBRAIC': 'Algebraic Error',
            'GEOMETRIC': 'Geometric Error',
            'ARITHMETIC': 'Arithmetic Error'
        }
        
        # Common mathematical patterns
        self.patterns = {
            'LINEAR_EQUATION': r'([a-zA-Z])\s*[=x]\s*([\d\.]+)',
            'QUADRATIC_EQUATION': r'([a-zA-Z])\s*\^\s*2\s*[+-]\s*([\d\.]+)\s*([a-zA-Z])\s*[+-]\s*([\d\.]+)',
            'FRACTION': r'([\d\.]+)\s*/\s*([\d\.]+)',
            'PYTHAGOREAN': r'([a-zA-Z])\s*\^\s*2\s*[+\-]\s*([a-zA-Z])\s*\^\s*2\s*[=]\s*([a-zA-Z])\s*\^\s*2'
        }
    
    def normalize_expression(self, expr):
        """More sophisticated expression normalization"""
        if not expr or not isinstance(expr, str):
            return ""
            
        # First, clean up any spacing issues
        expr = expr.strip()
        
        try:
            # Replace unicode and special characters with Python-compatible operators
            replacements = {
                '×': '*',  # Multiplication sign
                '÷': '/',  # Division sign
                '^': '**',
                '²': '**2',  # Superscript 2
                '³': '**3',  # Superscript 3
                '√': 'sqrt',  # Square root
                'π': 'pi',  # Pi
                'θ': 'theta',  # Theta
                '≤': '<=',  # Less than or equal
                '≥': '>=',  # Greater than or equal
                '≠': '!=',  # Not equal
            }
            
            for old, new in replacements.items():
                expr = expr.replace(old, new)
                
            # Handle fractions (different formats)
            expr = re.sub(r'([\d\.a-zA-Z]+)\s*/\s*([\d\.a-zA-Z]+)', r'\1/\2', expr)
            
            # Handle exponents (different formats)
            expr = re.sub(r'([\d\.a-zA-Z]+)\s*\^\s*([\d\.a-zA-Z]+)', r'\1**\2', expr)
            
            # Handle spaces around operators consistently
            expr = re.sub(r'\s*([+\-*/=<>])\s*', r' \1 ', expr)
            expr = re.sub(r'\s+', ' ', expr)  # Normalize spacing
            
            return expr.strip()
            
        except Exception as e:
            print(f"Error normalizing expression: {str(e)}")
            return expr  # Return original if normalization fails
    
    def parse_math_expression(self, expr):
        """Improved expression parsing with error handling"""
        try:
            normalized = self.normalize_expression(expr)
            parsed = parse_expr(normalized)
            return parsed
        except Exception as e:
            return None
    
    def detect_math_type(self, expr):
        """Detect the type of math problem"""
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, expr):
                return pattern_name
        return 'UNKNOWN'
    
    def analyze_expression(self, expr):
        """Analyze a mathematical expression"""
        analysis = {
            'type': self.detect_math_type(expr),
            'errors': [],
            'feedback': []
        }
        
        try:
            parsed = self.parse_math_expression(expr)
            if parsed is None:
                analysis['errors'].append({
                    'type': self.error_types['SYNTAX'],
                    'description': 'Invalid mathematical expression'
                })
                return analysis
            
            # Check for common errors based on expression type
            if analysis['type'] == 'LINEAR_EQUATION':
                analysis.update(self.analyze_linear_equation(expr, parsed))
            elif analysis['type'] == 'QUADRATIC_EQUATION':
                analysis.update(self.analyze_quadratic_equation(expr, parsed))
            
            return analysis
        except Exception as e:
            analysis['errors'].append({
                'type': self.error_types['PROCEDURAL'],
                'description': f'Error analyzing expression: {str(e)}'
            })
            return analysis
    
    def analyze_linear_equation(self, expr, parsed):
        """Analyze linear equations"""
        analysis = {}
        try:
            # Extract variables and constants
            variables = parsed.free_symbols
            constants = [term for term in parsed.as_ordered_terms() if term.is_number]
            
            # Check for common errors
            if len(variables) > 1:
                analysis['errors'].append({
                    'type': self.error_types['ALGEBRAIC'],
                    'description': 'Too many variables in linear equation'
                })
            
            # Generate feedback
            analysis['feedback'].append({
                'type': 'STEPS',
                'content': 'To solve a linear equation:\n1. Isolate the variable\n2. Combine like terms\n3. Divide by the coefficient'
            })
            
            return analysis
        except Exception as e:
            analysis['errors'].append({
                'type': self.error_types['PROCEDURAL'],
                'description': f'Error analyzing linear equation: {str(e)}'
            })
            return analysis
    
    def analyze_quadratic_equation(self, expr, parsed):
        """Analyze quadratic equations"""
        analysis = {}
        try:
            # Extract coefficients
            x = sympy.Symbol('x')
            a = parsed.coeff(x**2)
            b = parsed.coeff(x)
            c = parsed.coeff(1)
            
            # Check for common errors
            if a == 0:
                analysis['errors'].append({
                    'type': self.error_types['ALGEBRAIC'],
                    'description': 'This is not a quadratic equation (a=0)'
                })
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c
            
            # Generate feedback based on discriminant
            if discriminant > 0:
                analysis['feedback'].append({
                    'type': 'SOLUTION',
                    'content': 'This equation has two real solutions. Use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a'
                })
            elif discriminant == 0:
                analysis['feedback'].append({
                    'type': 'SOLUTION',
                    'content': 'This equation has one real solution (repeated root). Use the quadratic formula: x = -b / 2a'
                })
            else:
                analysis['feedback'].append({
                    'type': 'SOLUTION',
                    'content': 'This equation has two complex solutions. Use the quadratic formula: x = (-b ± i√(4ac-b²)) / 2a'
                })
            
            return analysis
        except Exception as e:
            analysis['errors'].append({
                'type': self.error_types['PROCEDURAL'],
                'description': f'Error analyzing quadratic equation: {str(e)}'
            })
            return analysis
    
    def generate_feedback(self, analysis):
        """Generate comprehensive feedback based on analysis"""
        feedback = []
        
        # Add error-specific feedback
        for error in analysis.get('errors', []):
            feedback.append({
                'type': 'ERROR',
                'content': f"Error: {error['type']} - {error['description']}"
            })
        
        # Add type-specific feedback
        if analysis.get('type') == 'LINEAR_EQUATION':
            feedback.append({
                'type': 'STEPS',
                'content': 'To solve this linear equation:\n1. Combine like terms\n2. Isolate the variable\n3. Check your solution'
            })
        elif analysis.get('type') == 'QUADRATIC_EQUATION':
            feedback.append({
                'type': 'STEPS',
                'content': 'To solve this quadratic equation:\n1. Identify a, b, and c\n2. Use the quadratic formula\n3. Simplify the solutions'
            })
        
        return feedback
    
    def analyze_math_image(self, image_path):
        """Analyze a math image and provide comprehensive feedback"""
        # TODO: Implement proper image processing and OCR
        # This is a placeholder for now
        try:
            # Extract text from image (placeholder)
            text = "2x + 3 = 7"
            
            # Analyze the expression
            analysis = self.analyze_expression(text)
            
            # Generate feedback
            analysis['feedback'] = self.generate_feedback(analysis)
            
            return analysis
        except Exception as e:
            return {
                'errors': [{
                    'type': self.error_types['PROCEDURAL'],
                    'description': f'Error analyzing image: {str(e)}'
                }]
            }
