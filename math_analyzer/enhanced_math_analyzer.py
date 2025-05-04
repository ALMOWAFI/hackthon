"""
Enhanced Math Analyzer with vLLM Integration

This module extends the AdvancedMathAnalyzer with vLLM capabilities for more
sophisticated analysis, better error detection, and detailed feedback generation.
"""

import os
import json
import re
from sympy import symbols, Eq, solve, sympify, SympifyError, simplify
from .advanced_math_analyzer import AdvancedMathAnalyzer
from .vllm_client import VLLMClient

class EnhancedMathAnalyzer(AdvancedMathAnalyzer):
    """Enhanced math analyzer that leverages vLLM for deeper analysis and feedback"""
    
    def __init__(self, vllm_url=None, vllm_key=None):
        """
        Initialize the enhanced math analyzer
        
        Args:
            vllm_url: URL of the vLLM API endpoint
            vllm_key: API key for vLLM authentication
        """
        # Initialize the base analyzer
        super().__init__()
        
        # Initialize the vLLM client
        self.vllm_client = VLLMClient(api_url=vllm_url, api_key=vllm_key)
        
        # Enhanced error types
        self.error_types.update({
            'CONCEPTUAL_MISUNDERSTANDING': 'Conceptual Misunderstanding',
            'PROCESS_ERROR': 'Process Error',
            'LOGICAL_ERROR': 'Logical Reasoning Error',
            'NOTATION_ERROR': 'Mathematical Notation Error',
            'APPLICATION_ERROR': 'Application Error'
        })
        
        # Enhanced patterns for problem identification
        self.patterns.update({
            'SYSTEM_OF_EQUATIONS': r'.*\n.*=.*\n.*=.*',
            'ALGEBRAIC_EXPRESSION': r'simplify\s+(.+)',
            'CALCULUS_DERIVATIVE': r'derivative\s+of\s+(.+)',
            'CALCULUS_INTEGRAL': r'integral\s+of\s+(.+)',
            'PROBABILITY': r'probability\s+(.+)',
            'STATISTICS': r'(mean|median|mode|variance|standard\s+deviation)\s+(.+)'
        })
    
    def analyze_expression(self, expr):
        """
        Analyze a math expression using both symbolic analysis and vLLM
        
        Args:
            expr: The math expression to analyze
            
        Returns:
            Analysis results as a dictionary
        """
        # First get the base analysis from the parent class
        base_analysis = super().analyze_expression(expr)
        
        # If it's a simple expression, just return the base analysis
        if not expr or len(expr.strip()) < 3:
            return base_analysis
        
        try:
            # Attempt to extract the problem and expected answer
            problem_parts = self._extract_problem_parts(expr)
            
            if problem_parts:
                # Use vLLM for deeper analysis if we can identify a problem structure
                vllm_analysis = self.vllm_client.analyze_math_expression(
                    problem=problem_parts.get('problem', expr),
                    answer=problem_parts.get('answer', '')
                )
                
                # Merge the analyses, prioritizing vLLM for detailed feedback
                enhanced_analysis = self._merge_analyses(base_analysis, vllm_analysis)
                return enhanced_analysis
            
        except Exception as e:
            # If vLLM analysis fails, log the error and return the base analysis
            print(f"vLLM analysis error: {str(e)}")
            if 'errors' not in base_analysis:
                base_analysis['errors'] = []
            base_analysis['errors'].append({
                'type': 'ANALYSIS_ERROR',
                'description': f'Enhanced analysis failed: {str(e)}'
            })
        
        return base_analysis
    
    def _extract_problem_parts(self, expr):
        """
        Extract problem and answer parts from an expression
        
        Args:
            expr: The expression to parse
            
        Returns:
            Dictionary with problem and answer components
        """
        # Check for equation format: problem = answer
        if '=' in expr:
            parts = expr.split('=', 1)
            if len(parts) == 2:
                return {
                    'problem': parts[0].strip(),
                    'answer': parts[1].strip()
                }
        
        # Check for common problem formats like "Solve for x: 2x + 3 = 7"
        solve_match = re.search(r'solve\s+for\s+([a-zA-Z]):\s*(.+)', expr, re.IGNORECASE)
        if solve_match:
            return {
                'problem': f"Solve for {solve_match.group(1)}: {solve_match.group(2)}",
                'answer': expr  # We don't have a separate answer, so use the whole expression
            }
        
        # Check for other common formats: "Find the derivative of f(x) = x^2"
        derivative_match = re.search(r'(find|calculate)\s+the\s+derivative\s+of\s+(.+)', expr, re.IGNORECASE)
        if derivative_match:
            return {
                'problem': f"Find the derivative of {derivative_match.group(2)}",
                'answer': expr  # We don't have a separate answer
            }
        
        # Default: treat the whole expression as both problem and answer
        return {
            'problem': expr,
            'answer': expr
        }
    
    def _merge_analyses(self, base_analysis, vllm_analysis):
        """
        Merge the base and vLLM analyses, prioritizing vLLM for detailed feedback
        
        Args:
            base_analysis: Analysis from the base AdvancedMathAnalyzer
            vllm_analysis: Analysis from the vLLM model
            
        Returns:
            Merged analysis dictionary
        """
        # Start with the base analysis
        merged = base_analysis.copy()
        
        # If vLLM didn't return a proper analysis, just return the base analysis
        if not isinstance(vllm_analysis, dict):
            return merged
        
        # Add vLLM-specific fields
        merged['vllm_enhanced'] = True
        
        # Add correctness assessment if available
        if 'is_correct' in vllm_analysis:
            merged['is_correct'] = vllm_analysis['is_correct']
        
        # Merge error information, prioritizing vLLM's more detailed analysis
        if 'errors' in vllm_analysis and vllm_analysis['errors']:
            # If vLLM found errors, use those instead of the base errors
            merged['errors'] = vllm_analysis['errors']
        elif 'error_detected' in vllm_analysis and vllm_analysis['error_detected'] != 'UNKNOWN':
            # Add the error if it's not already in the base analysis
            error_found = False
            for error in merged.get('errors', []):
                if error['type'] == vllm_analysis['error_detected']:
                    error_found = True
                    break
            
            if not error_found:
                if 'errors' not in merged:
                    merged['errors'] = []
                merged['errors'].append({
                    'type': vllm_analysis['error_detected'],
                    'description': vllm_analysis.get('analysis', 'Error detected by vLLM')
                })
        
        # Add detailed analysis and steps
        if 'analysis' in vllm_analysis:
            merged['detailed_analysis'] = vllm_analysis['analysis']
        
        if 'steps' in vllm_analysis:
            merged['solution_steps'] = vllm_analysis['steps']
        
        # Add any other fields from vLLM analysis
        for key, value in vllm_analysis.items():
            if key not in merged and key != 'raw_response':
                merged[key] = value
        
        return merged
    
    def generate_custom_feedback(self, problem, answer, error_type='UNKNOWN', style='constructive'):
        """
        Generate custom feedback for a math problem using vLLM
        
        Args:
            problem: The math problem
            answer: The student's answer
            error_type: Type of error detected
            style: Feedback style (constructive, socratic, direct)
            
        Returns:
            Detailed feedback as a string
        """
        try:
            return self.vllm_client.generate_feedback(
                problem=problem,
                answer=answer,
                error_type=error_type,
                style=style
            )
        except Exception as e:
            print(f"Error generating custom feedback: {str(e)}")
            return "I couldn't generate custom feedback at this time. Please check the problem again."
    
    def generate_practice_problems(self, topic, skill, count=3, difficulty='medium'):
        """
        Generate practice problems targeting a specific skill
        
        Args:
            topic: General math topic (e.g., "algebra", "fractions")
            skill: Specific skill to practice
            count: Number of problems to generate
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            List of problems with solutions
        """
        try:
            return self.vllm_client.generate_practice_worksheet(
                topic=topic,
                skill=skill,
                count=count,
                difficulty=difficulty
            )
        except Exception as e:
            print(f"Error generating practice problems: {str(e)}")
            return [{
                "problem": "Could not generate practice problems at this time.",
                "steps": "Please try again later.",
                "answer": "Error: " + str(e)
            }]


# Example usage
if __name__ == "__main__":
    analyzer = EnhancedMathAnalyzer()
    
    # Test with a simple equation
    result = analyzer.analyze_expression("2x + 3 = 7")
    print(json.dumps(result, indent=2))
    
    # Test with a specific problem/answer format
    result = analyzer.analyze_expression("Solve for x: 2x + 3 = 7")
    print(json.dumps(result, indent=2))
    
    # Test feedback generation
    feedback = analyzer.generate_custom_feedback(
        problem="Find the derivative of f(x) = x^2 + 3x",
        answer="f'(x) = 2x + 3",
        error_type="NONE",
        style="constructive"
    )
    print("\nFeedback:", feedback)
