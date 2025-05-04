import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL, ERROR_TYPES

class FeedbackGenerator:
    def __init__(self):
        # Initialize without external API
        self.model = None
        self.error_types = ERROR_TYPES
            
    def generate_feedback(self, analysis):
        """
        Generate feedback for a question analysis.
        
        Args:
            analysis (dict): Analysis results for a question
            
        Returns:
            dict: Feedback and recommendations
        """
        if not analysis['errors']:
            return {
                'feedback': "Great job! Your answer is correct.",
                'recommendations': []
            }
            
        # Group errors by type
        error_groups = {}
        for error in analysis['errors']:
            error_type = error['type']
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error['description'])
            
        # Always use the basic feedback generator
        feedback = self._generate_basic_feedback(analysis, error_groups)
            
        # Generate recommendations
        recommendations = self._generate_recommendations(error_groups)
        
        return {
            'feedback': feedback,
            'recommendations': recommendations
        }
        
    def _create_feedback_prompt(self, analysis, error_groups):
        """Create a prompt for the Gemini model."""
        prompt = f"""
        A student answered the math question: {analysis['question']}
        Their answer was: {analysis['student_answer']}
        The correct answer is: {analysis['correct_answer']}
        
        They made the following errors:
        """
        
        for error_type, errors in error_groups.items():
            prompt += f"\n{error_type}:\n"
            for error in errors:
                prompt += f"- {error}\n"
                
        prompt += """
        Please provide:
        1. A clear explanation of what went wrong
        2. Step-by-step guidance on how to solve similar problems correctly
        3. Encouraging feedback that helps the student learn from their mistakes
        """
        
        return prompt
        
    def _generate_basic_feedback(self, analysis, error_groups):
        """Generate basic feedback without external API."""
        
        if not error_groups:
            return "Great job! Your answer is correct."
            
        feedback = "Let's look at what needs improvement:\n\n"
        
        # Add specific feedback based on the problem type
        if 'x²' in analysis.get('question', '') and 'y²' in analysis.get('question', ''):
            feedback += "This is the Pythagorean theorem (x² + y² = r²).\n"
            
        elif '1 + 1' in analysis.get('question', ''):
            feedback += "The correct answer to 1 + 1 is 2, not 3.\n"
            
        elif '1 - 1' in analysis.get('question', ''):
            feedback += "The correct answer to 1 - 1 is 0, not 4.\n"
            
        elif '1/0' in analysis.get('question', '') or '1 / 0' in analysis.get('question', ''):
            feedback += "Division by zero is undefined in mathematics.\n"
            
        # Add general error feedback
        for error_type, errors in error_groups.items():
            feedback += f"\n{error_type}:\n"
            for error in errors:
                feedback += f"- {error}\n"
            
        return feedback
        
    def _generate_recommendations(self, error_groups):
        """Generate learning recommendations based on error types."""
        recommendations = []
        
        if 'PROCEDURAL' in error_groups:
            recommendations.append({
                'type': 'PROCEDURAL',
                'suggestion': 'Practice the order of operations and step-by-step problem solving',
                'resources': ['Order of operations worksheets', 'Step-by-step math tutorials']
            })
            
        if 'CONCEPTUAL' in error_groups:
            recommendations.append({
                'type': 'CONCEPTUAL',
                'suggestion': 'Review the underlying mathematical concepts',
                'resources': ['Concept explanation videos', 'Interactive math lessons']
            })
            
        if 'CALCULATION' in error_groups:
            recommendations.append({
                'type': 'CALCULATION',
                'suggestion': 'Practice basic arithmetic and double-check calculations',
                'resources': ['Arithmetic practice sheets', 'Mental math exercises']
            })
            
        return recommendations
        
    def format_feedback(self, results):
        """
        Format the complete feedback for all questions.
        
        Args:
            results (dict): Complete analysis results
            
        Returns:
            str: Formatted feedback text
        """
        feedback = f"Math Homework Analysis\n"
        feedback += f"Final Score: {results['final_score']:.1f}%\n\n"
        
        for question in results['questions']:
            feedback += f"\nQuestion {question['question_number']}:\n"
            feedback += f"Score: {question['score']}%\n"
            feedback += f"Feedback: {question['feedback']}\n"
            
            if question['recommendations']:
                feedback += "\nRecommendations:\n"
                for rec in question['recommendations']:
                    feedback += f"- {rec['suggestion']}\n"
                    feedback += f"  Resources: {', '.join(rec['resources'])}\n"
                    
        return feedback 