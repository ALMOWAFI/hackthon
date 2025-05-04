import numpy as np
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import editdistance

class MLValidator:
    """
    Machine learning-based answer validator that handles various math expressions
    and open-ended responses with context-aware evaluation.
    """
    
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 5))
        self.cached_vectorizer_fitted = False
    
    def validate_answer(self, student_answer, correct_answer, context=None, question_type='math'):
        """
        Validate student answer against correct answer with context awareness
        
        Args:
            student_answer (str): Student's provided answer
            correct_answer (str): Reference correct answer
            context (dict): Additional context about the question
            question_type (str): Type of question (math, text, etc.)
            
        Returns:
            dict: Validation results with score and explanations
        """
        if question_type == 'math':
            return self._validate_math_answer(student_answer, correct_answer, context)
        elif question_type == 'text':
            return self._validate_text_answer(student_answer, correct_answer, context)
        else:
            # Default to math validation with a warning
            print(f"Warning: Unsupported question type '{question_type}'. Using math validation.")
            return self._validate_math_answer(student_answer, correct_answer, context)
    
    def _validate_math_answer(self, student_answer, correct_answer, context=None):
        """Validate mathematical expressions and equations"""
        # Initial cleaning of expressions
        student_clean = self._normalize_math_expr(student_answer)
        correct_clean = self._normalize_math_expr(correct_answer)
        
        # Initialize results dictionary
        results = {
            'is_correct': False,
            'score': 0,
            'explanation': '',
            'partial_credit': False,
            'error_type': None,
            'error_location': None
        }
        
        # Try evaluating as expressions
        try:
            student_expr = parse_expr(student_clean)
            correct_expr = parse_expr(correct_clean)
            
            # Check exact symbolic equivalence
            if sympy.simplify(student_expr - correct_expr) == 0:
                results['is_correct'] = True
                results['score'] = 100
                results['explanation'] = "Correct! The answer is mathematically equivalent."
                return results
                
            # Check numerical equivalence (might be formatted differently)
            try:
                student_val = float(student_expr.evalf())
                correct_val = float(correct_expr.evalf())
                
                # Check if values are close (within small epsilon)
                if abs(student_val - correct_val) < 1e-10:
                    results['is_correct'] = True
                    results['score'] = 100
                    results['explanation'] = "Correct! The numerical value matches."
                    return results
                    
                # Check for rounding errors - partial credit
                if abs(student_val - correct_val) < 0.05 * abs(correct_val):
                    results['is_correct'] = False
                    results['partial_credit'] = True
                    results['score'] = 80
                    results['explanation'] = f"Close! Your answer {student_val} is very close to {correct_val}. Check for rounding errors."
                    results['error_type'] = "rounding"
                    return results
                
                # Way off but reached a numerical answer
                results['score'] = 25
                results['explanation'] = f"Incorrect. Your answer evaluates to {student_val}, but the correct value is {correct_val}."
                results['error_type'] = "calculation"
                
            except:
                # Could not convert to numerical value
                results['score'] = 20
                results['explanation'] = "Cannot numerically evaluate your expression to compare with the correct answer."
                results['error_type'] = "format"
                
        except Exception as e:
            # Parsing failed - try alternative approaches
            pass
        
        # If we got here, symbolic methods failed
        # Try fuzzy string matching with n-grams for expressions
        similarity = self._calculate_expression_similarity(student_clean, correct_clean)
        
        if similarity > 0.95:
            # Very close textually - might be minor formatting difference
            results['is_correct'] = True
            results['score'] = 95
            results['explanation'] = "Appears correct, though formatted slightly differently."
            return results
            
        elif similarity > 0.8:
            # Similar but not identical - partial credit
            results['partial_credit'] = True
            results['score'] = 70
            results['explanation'] = "Partially correct. Your answer is close to the right expression."
            results['error_type'] = "near_miss"
            return results
            
        # Check for common math errors based on patterns
        error_check = self._check_common_math_errors(student_answer, correct_answer)
        if error_check['error_found']:
            results['score'] = error_check['score']
            results['explanation'] = error_check['explanation']
            results['error_type'] = error_check['error_type']
            results['error_location'] = error_check.get('error_location')
            return results
        
        # If we've tried everything and failed
        if results['score'] == 0:
            results['score'] = 10  # Some credit for attempting
            results['explanation'] = "Incorrect. Your answer is substantially different from the expected result."
            results['error_type'] = "major_error"
            
        return results
    
    def _validate_text_answer(self, student_answer, correct_answer, context=None):
        """Validate text-based open-ended responses"""
        # Clean and normalize text
        student_clean = student_answer.lower().strip()
        correct_clean = correct_answer.lower().strip()
        
        # Initialize results
        results = {
            'is_correct': False,
            'score': 0,
            'explanation': '',
            'key_points_matched': []
        }
        
        # Exact match check
        if student_clean == correct_clean:
            results['is_correct'] = True
            results['score'] = 100
            results['explanation'] = "Perfect answer!"
            return results
            
        # Key point extraction and matching
        if context and 'key_points' in context:
            key_points = context['key_points']
            matched_points = []
            
            for point in key_points:
                if point.lower() in student_clean:
                    matched_points.append(point)
                    
            # Calculate score based on key points
            if matched_points:
                point_score = 100 * (len(matched_points) / len(key_points))
                results['score'] = point_score
                results['key_points_matched'] = matched_points
                
                if point_score > 90:
                    results['is_correct'] = True
                    results['explanation'] = "Excellent! You included all key points."
                elif point_score > 70:
                    results['explanation'] = f"Good answer - you covered {len(matched_points)} out of {len(key_points)} key points."
                elif point_score > 40:
                    results['explanation'] = "Partial credit - your answer includes some key points but misses others."
                else:
                    results['explanation'] = "Your answer touches on a few points but is mostly incorrect."
                    
                return results
        
        # Semantic similarity check (fallback if no key points)
        similarity = self._calculate_text_similarity(student_clean, correct_clean)
        
        if similarity > 0.9:
            results['is_correct'] = True
            results['score'] = 95
            results['explanation'] = "Your answer is semantically equivalent to the correct answer."
        elif similarity > 0.75:
            results['score'] = 80
            results['explanation'] = "Your answer is close to correct, but missing some elements."
        elif similarity > 0.5:
            results['score'] = 50
            results['explanation'] = "Partially correct, but your answer could be more comprehensive."
        else:
            results['score'] = 20
            results['explanation'] = "Your answer is significantly different from what was expected."
            
        return results
    
    def _normalize_math_expr(self, expr):
        """Normalize a mathematical expression for comparison"""
        if not expr:
            return ""
            
        # Convert to string if not already
        expr = str(expr).strip()
        
        # Replace common handwritten or different notations
        replacements = {
            '×': '*',
            '÷': '/',
            '^': '**',
            '²': '**2',
            '³': '**3',
            ' ': ''  # Remove spaces
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
            
        return expr
    
    def _calculate_expression_similarity(self, expr1, expr2):
        """Calculate similarity between two math expressions"""
        if not expr1 or not expr2:
            return 0.0
            
        # Use edit distance for short expressions
        if len(expr1) < 10 and len(expr2) < 10:
            max_len = max(len(expr1), len(expr2))
            if max_len == 0:
                return 1.0 if expr1 == expr2 else 0.0
            return 1.0 - (editdistance.eval(expr1, expr2) / max_len)
            
        # Use n-gram similarity for longer expressions
        docs = [expr1, expr2]
        
        # Fit vectorizer if not already
        if not self.cached_vectorizer_fitted:
            self.vectorizer.fit(docs)
            self.cached_vectorizer_fitted = True
            
        # Transform to vectors
        try:
            vectors = self.vectorizer.transform(docs)
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except:
            # Fallback to edit distance if vectorization fails
            max_len = max(len(expr1), len(expr2))
            return 1.0 - (editdistance.eval(expr1, expr2) / max_len)
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text answers"""
        if not text1 or not text2:
            return 0.0
            
        # Use word vectorizer for text
        word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
        
        try:
            # Transform to vectors
            vectors = word_vectorizer.fit_transform([text1, text2])
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except:
            # Fallback to character-level if word vectorization fails
            return self._calculate_expression_similarity(text1, text2)
    
    def _check_common_math_errors(self, student_answer, correct_answer):
        """Check for common mathematical errors to provide specific feedback"""
        result = {
            'error_found': False,
            'error_type': None,
            'score': 0,
            'explanation': "",
            'error_location': None
        }
        
        # Check for sign errors (e.g., + instead of -, or vice versa)
        if len(student_answer) > 2 and len(correct_answer) > 2:
            # Replace + with - and check if expressions match
            student_flipped = student_answer.replace('+', 'TEMP').replace('-', '+').replace('TEMP', '-')
            if student_flipped == correct_answer:
                result['error_found'] = True
                result['error_type'] = "sign_error"
                result['score'] = 60
                result['explanation'] = "Check your signs. You may have mixed up + and -."
                # Find position of first sign difference
                for i, (s_char, c_char) in enumerate(zip(student_answer, correct_answer)):
                    if (s_char == '+' and c_char == '-') or (s_char == '-' and c_char == '+'):
                        result['error_location'] = i
                        break
                return result
        
        # Check for division by zero
        if '/0' in student_answer or '÷0' in student_answer:
            result['error_found'] = True
            result['error_type'] = "division_by_zero"
            result['score'] = 20
            result['explanation'] = "Division by zero is undefined."
            # Find position of division by zero
            result['error_location'] = student_answer.find('/0')
            if result['error_location'] == -1:
                result['error_location'] = student_answer.find('÷0')
            return result
            
        # Check for missing parentheses (order of operations error)
        if ('*' in student_answer or '/' in student_answer) and ('+' in student_answer or '-' in student_answer):
            # This is a simple heuristic - in production you'd need more sophisticated parsing
            try:
                student_eval = eval(student_answer.replace('^', '**'))
                correct_eval = eval(correct_answer.replace('^', '**'))
                
                # Check if adding parens around certain parts would fix it
                # (This is a simplified example, would need more comprehensive testing)
                for op in ['+', '-']:
                    if op in student_answer:
                        parts = student_answer.split(op)
                        for i in range(len(parts)):
                            modified = student_answer.replace(parts[i], f"({parts[i]})")
                            try:
                                mod_eval = eval(modified.replace('^', '**'))
                                if abs(mod_eval - correct_eval) < abs(student_eval - correct_eval):
                                    result['error_found'] = True
                                    result['error_type'] = "parentheses_error"
                                    result['score'] = 70
                                    result['explanation'] = "Check your order of operations. You may need parentheses."
                                    return result
                            except:
                                pass
            except:
                pass
        
        return result
