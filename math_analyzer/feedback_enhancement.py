"""
Advanced Feedback Enhancement Module
Integrates the TeachingPerspectives module to provide richer, more pedagogically
sound feedback in the document analyzer system.
"""

import os
import json
from .teaching_module import TeachingPerspectives

class FeedbackEnhancer:
    """
    Enhances feedback by applying advanced teaching principles and
    educational research to math problem analysis.
    """
    
    def __init__(self):
        self.teaching = TeachingPerspectives()
        
    def enhance_feedback(self, analysis_results):
        """
        Takes analysis results and enhances them with advanced pedagogical insights.
        
        Args:
            analysis_results (dict): The original analysis results
            
        Returns:
            dict: Enhanced analysis results with richer feedback
        """
        enhanced_results = analysis_results.copy()
        
        # Enhance each question with better feedback
        if 'questions' in enhanced_results:
            for i, question in enumerate(enhanced_results['questions']):
                # Get question details
                question_text = question.get('text', '')
                analysis = question.get('analysis', {})
                is_correct = analysis.get('score', 0) > 70
                
                student_answer = analysis.get('student_answer', '')
                correct_answer = analysis.get('correct_answer', '')
                
                # Get error information
                errors = analysis.get('errors', [])
                if errors:
                    error_type = errors[0].get('type', 'CONCEPTUAL')
                    error_description = errors[0].get('description', '')
                else:
                    error_type = None
                    error_description = None
                
                # Generate enhanced feedback
                enhanced_feedback = self.teaching.generate_feedback(
                    question_text=question_text,
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    error_type=error_type,
                    error_description=error_description,
                    is_correct=is_correct
                )
                
                # Identify relevant mathematical concept
                concept_domain = self.teaching.identify_relevant_mcp(question_text, error_description or "")
                
                # Generate conceptual explanation and learning strategy
                conceptual_insight = self.teaching.generate_conceptual_explanation(
                    concept_domain, 
                    error_description
                )
                
                strategy = self.teaching.generate_learning_strategy(
                    concept_domain, 
                    error_type or "CONCEPTUAL"
                )
                
                # Add enhanced feedback to the question
                question['feedback_text'] = enhanced_feedback
                
                # Add conceptual insights
                if 'feedback' not in question:
                    question['feedback'] = {}
                    
                question['feedback']['feedback'] = enhanced_feedback
                question['feedback']['concept_domain'] = concept_domain
                question['feedback']['conceptual_insight'] = conceptual_insight
                question['feedback']['learning_strategy'] = strategy
                
                # Add recommendations based on teaching approach
                question['recommendations'] = [
                    {
                        'suggestion': strategy,
                        'resources': [f"Mathematics {concept_domain.replace('_', ' ')} Guide", 
                                     "Interactive Math Practice", 
                                     "Visual Math Tutorials"]
                    }
                ]
                
                enhanced_results['questions'][i] = question
                
        return enhanced_results
    
    def get_advanced_teaching_perspectives(self):
        """
        Returns information about the most advanced teaching perspectives
        and mathematical concept principles.
        
        Returns:
            dict: Information about advanced teaching approaches
        """
        # Get the most advanced/challenging MCPs
        wildest_mcps = self.teaching.get_wildest_mcps()
        
        # Get the most sophisticated pedagogical frameworks
        advanced_frameworks = {
            'PRODUCTIVE_STRUGGLE': self.teaching.PEDAGOGICAL_FRAMEWORKS['PRODUCTIVE_STRUGGLE'],
            'CONCEPTUAL_CHANGE': self.teaching.PEDAGOGICAL_FRAMEWORKS['CONCEPTUAL_CHANGE'],
        }
        
        # Get most effective teaching styles for advanced concepts
        advanced_styles = {
            'SOCRATIC': self.teaching.TEACHING_STYLES['SOCRATIC'],
            'METACOGNITIVE': self.teaching.TEACHING_STYLES['METACOGNITIVE']
        }
        
        return {
            'wildest_mcps': wildest_mcps,
            'advanced_frameworks': advanced_frameworks,
            'advanced_styles': advanced_styles
        }
    
    def generate_teaching_perspective_report(self, output_path='results/teaching_perspective_report.json'):
        """
        Generates a comprehensive report of advanced teaching perspectives
        and mathematical concept principles.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            bool: Success status
        """
        report = {
            'math_concept_principles': self.teaching.MATH_CONCEPT_PRINCIPLES,
            'teaching_styles': self.teaching.TEACHING_STYLES,
            'pedagogical_frameworks': self.teaching.PEDAGOGICAL_FRAMEWORKS,
            'advanced_perspectives': self.get_advanced_teaching_perspectives()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            print(f"Error generating teaching perspective report: {str(e)}")
            return False


# Usage example:
"""
from math_analyzer.feedback_enhancement import FeedbackEnhancer

# Create enhancer
enhancer = FeedbackEnhancer()

# Get the most advanced mathematical concept principles
advanced_mcps = enhancer.get_advanced_teaching_perspectives()
print("Most advanced MCPs:", advanced_mcps['wildest_mcps'])

# Generate a comprehensive report of all teaching perspectives
enhancer.generate_teaching_perspective_report()
"""
