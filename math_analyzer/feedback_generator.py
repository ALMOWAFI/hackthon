import random
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

class TeachingStyle(Enum):
    SOCRATIC = "socratic"
    DIRECT = "direct"
    GROWTH_MINDSET = "growth_mindset"
    CONCEPTUAL = "conceptual"
    VISUAL = "visual"
    PROCEDURAL = "procedural"
    ANALOGICAL = "analogical"
    GAMIFIED = "gamified"

class ErrorType(Enum):
    CALCULATION = "calculation"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    NOTATION = "notation"
    
@dataclass
class FeedbackTemplate:
    style: TeachingStyle
    error_type: ErrorType
    templates: List[str]
    follow_up_questions: List[str]
    practice_problems: List[Dict[str, Any]]
    
class MathFeedbackGenerator:
    """Advanced generator for personalized math feedback using various teaching styles"""
    
    def __init__(self, templates_file=None):
        # Load default templates if no file provided
        self.templates = self._load_templates(templates_file)
        self.student_history = {}
        self.learning_styles = {}
        
    def _load_templates(self, file_path=None):
        """Load feedback templates from file or use defaults"""
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading templates: {e}")
                
        # Default templates by teaching style and error type
        return {
            # Socratic style templates
            TeachingStyle.SOCRATIC.value: {
                ErrorType.CALCULATION.value: [
                    "Let's look at your calculation steps. What happens when you compute {incorrect_step}?",
                    "I notice you got {student_answer}. Can you walk through how you calculated this step by step?"
                ],
                ErrorType.PROCEDURAL.value: [
                    "I see you went from {step1} to {step2}. What rule or property did you apply here?",
                    "What steps should we follow when solving {problem_type} problems?"
                ],
                ErrorType.CONCEPTUAL.value: [
                    "What does {concept} mean in the context of this problem?",
                    "How would you explain the relationship between {concept1} and {concept2}?"
                ]
            },
            
            # Direct instruction templates
            TeachingStyle.DIRECT.value: {
                ErrorType.CALCULATION.value: [
                    "There's a calculation error. {incorrect_step} should equal {correct_step}, not {student_step}.",
                    "Check your arithmetic in step {step_number}. {incorrect_calculation} = {correct_result}."
                ],
                ErrorType.PROCEDURAL.value: [
                    "You need to {correct_procedure} instead of {student_procedure}.",
                    "The correct sequence is: {step1}, then {step2}, then {step3}."
                ],
                ErrorType.CONCEPTUAL.value: [
                    "{concept} means {definition}, so you should apply it by {application}.",
                    "You're confusing {concept1} with {concept2}. Here's the difference: {explanation}"
                ]
            },
            
            # Growth mindset templates
            TeachingStyle.GROWTH_MINDSET.value: {
                ErrorType.CALCULATION.value: [
                    "This calculation is tricky! Many students find this challenging at first. Let's break it down.",
                    "Good effort on tackling this complex calculation. With practice, these will become easier."
                ],
                ErrorType.PROCEDURAL.value: [
                    "You're making progress with this procedure! Let's refine your approach.",
                    "I can see your thinking process here. With a small adjustment to your procedure, you'll master this."
                ],
                ErrorType.CONCEPTUAL.value: [
                    "This concept takes time to fully grasp. Your understanding is developing nicely.",
                    "Building a deep understanding of {concept} is challenging but you're on the right track."
                ]
            },
            
            # Visual learning templates
            TeachingStyle.VISUAL.value: {
                ErrorType.CALCULATION.value: [
                    "Let me show you visually: [DIAGRAM: step-by-step calculation breakdown]",
                    "Picture the numbers on a number line: [VISUAL: number line showing operation]"
                ],
                ErrorType.PROCEDURAL.value: [
                    "Here's a flowchart of the correct procedure: [DIAGRAM: procedure steps]",
                    "This visual model shows the process: [VISUAL: process diagram]"
                ],
                ErrorType.CONCEPTUAL.value: [
                    "This diagram illustrates the concept: [VISUAL: concept visualization]",
                    "Here's how to visualize {concept}: [DIAGRAM: conceptual model]"
                ]
            }
        }
    
    def _select_teaching_style(self, student_id=None, error_type=None, previous_styles=None):
        """Select appropriate teaching style based on student history or error type"""
        if student_id and student_id in self.learning_styles:
            # Use the student's preferred style if known
            return self.learning_styles[student_id]
            
        if previous_styles and len(previous_styles) > 0:
            # Avoid using the same style repeatedly
            available_styles = [style for style in TeachingStyle if style.value not in previous_styles[-2:]]
            if available_styles:
                return random.choice(available_styles)
        
        # Select style based on error type if provided
        if error_type:
            if error_type == ErrorType.CALCULATION.value:
                return random.choice([TeachingStyle.DIRECT, TeachingStyle.VISUAL])
            elif error_type == ErrorType.PROCEDURAL.value:
                return random.choice([TeachingStyle.PROCEDURAL, TeachingStyle.DIRECT])
            elif error_type == ErrorType.CONCEPTUAL.value:
                return random.choice([TeachingStyle.SOCRATIC, TeachingStyle.CONCEPTUAL])
                
        # Default to random selection
        return random.choice(list(TeachingStyle))
    
    def _fill_template(self, template, context):
        """Fill in a template with context-specific values"""
        filled = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in filled:
                filled = filled.replace(placeholder, str(value))
        return filled
        
    def _generate_practice_problems(self, error_type, difficulty, n=3):
        """Generate relevant practice problems based on error type"""
        # This would ideally pull from a large database of problems
        # For now, we'll return some examples
        practice_sets = {
            ErrorType.CALCULATION.value: [
                {"problem": "Calculate: 3(5 - 2) + 4", "answer": "13", "difficulty": "easy"},
                {"problem": "Simplify: 2(x + 3) - 4(x - 1)", "answer": "2x + 10", "difficulty": "medium"},
                {"problem": "Calculate: 5/8 + 3/4", "answer": "11/8", "difficulty": "medium"},
                {"problem": "Solve: 2.5 × (6.4 + 1.2)", "answer": "19", "difficulty": "medium"},
                {"problem": "Calculate: (-3)² × √16", "answer": "36", "difficulty": "hard"}
            ],
            ErrorType.PROCEDURAL.value: [
                {"problem": "Solve for x: 2x + 5 = 11", "answer": "x = 3", "difficulty": "easy"},
                {"problem": "Factor: x² + 5x + 6", "answer": "(x + 2)(x + 3)", "difficulty": "medium"},
                {"problem": "Solve: 3(x-2) + 4 = 2(x+1)", "answer": "x = 0", "difficulty": "medium"},
                {"problem": "Solve the system: 2x + y = 7, 3x - 2y = 4", "answer": "x = 2, y = 3", "difficulty": "hard"}
            ],
            ErrorType.CONCEPTUAL.value: [
                {"problem": "If f(x) = 2x + 3, find f(4)", "answer": "11", "difficulty": "easy"},
                {"problem": "Find the slope of the line through points (2,5) and (4,9)", "answer": "2", "difficulty": "medium"},
                {"problem": "Find the area of a circle with radius 3 units", "answer": "9π", "difficulty": "medium"},
                {"problem": "Find the derivative of f(x) = x³ - 2x + 1", "answer": "f'(x) = 3x² - 2", "difficulty": "hard"}
            ]
        }
        
        # Get problems of the right type and difficulty
        suitable_problems = [p for p in practice_sets.get(error_type, []) 
                           if p["difficulty"] == difficulty]
        
        # If not enough problems of the right difficulty, get problems of any difficulty
        if len(suitable_problems) < n:
            suitable_problems = practice_sets.get(error_type, [])
            
        # Return n problems (or fewer if not enough available)
        return random.sample(suitable_problems, min(n, len(suitable_problems)))
    
    def _determine_difficulty(self, error_severity, student_history=None):
        """Determine appropriate difficulty for practice problems"""
        if student_history and len(student_history) > 2:
            # Analyze recent performance
            recent_scores = [s.get("score", 0) for s in student_history[-3:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if avg_score > 85:
                return "hard"
            elif avg_score > 60:
                return "medium"
            else:
                return "easy"
        
        # Based on error severity if no history
        if error_severity > 0.7:
            return "easy"
        elif error_severity > 0.3:
            return "medium"
        else:
            return "hard"
    
    def generate_feedback(self, analysis, student_id=None):
        """
        Generate personalized feedback based on problem analysis
        
        Args:
            analysis (dict): Analysis of student's work including errors
            student_id (str, optional): Student identifier for personalization
            
        Returns:
            dict: Structured feedback with explanation, tips, and practice problems
        """
        # Extract relevant information from analysis
        errors = analysis.get("errors", [])
        question = analysis.get("question", "")
        student_answer = analysis.get("student_answer", "")
        correct_answer = analysis.get("correct_answer", "")
        
        # If no errors, provide positive feedback
        if not errors:
            return {
                "feedback": f"Great job! Your answer {student_answer} is correct.",
                "practice_recommendation": "You're ready for more challenging problems.",
                "practice_problems": self._generate_practice_problems(
                    random.choice([e.value for e in ErrorType]), 
                    "hard", 
                    n=2
                ),
                "style": TeachingStyle.GROWTH_MINDSET.value
            }
            
        # Get error details
        primary_error = errors[0]
        error_type = primary_error.get("type", "calculation")
        error_desc = primary_error.get("description", "")
        
        # Create context for template filling
        context = {
            "question": question,
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "error_description": error_desc,
            "concept": question.split()[0] if question else "this concept",  # Very simple concept extraction
            "step_number": "1",  # Placeholder
            "incorrect_step": student_answer,
            "correct_step": correct_answer
        }
        
        # Get student history if available
        student_history = self.student_history.get(student_id, []) if student_id else []
        previous_styles = [session.get("style") for session in student_history[-3:] if "style" in session]
        
        # Select teaching style
        style = self._select_teaching_style(student_id, error_type, previous_styles)
        
        # Get templates for this style and error type
        templates = self.templates.get(style.value, {}).get(error_type, [
            "Let's take a look at your work on this problem."
        ])
        
        # Select and fill a template
        selected_template = random.choice(templates) if templates else "Let's review this problem."
        feedback_text = self._fill_template(selected_template, context)
        
        # Add explanation of the correct approach
        if style != TeachingStyle.SOCRATIC:
            feedback_text += f"\n\nThe correct approach is: {correct_answer}."
        
        # Determine appropriate difficulty for practice
        error_severity = len(errors) / 3  # Simple severity metric
        difficulty = self._determine_difficulty(error_severity, student_history)
        
        # Generate practice problems
        practice_problems = self._generate_practice_problems(error_type, difficulty)
        
        # Create structured feedback response
        feedback = {
            "feedback": feedback_text,
            "practice_recommendation": f"Here are some {difficulty} problems to practice {error_type} skills:",
            "practice_problems": practice_problems,
            "style": style.value
        }
        
        # Update student history if ID provided
        if student_id:
            self.student_history.setdefault(student_id, []).append({
                "question": question,
                "error_type": error_type,
                "style": style.value,
                "score": analysis.get("score", 0)
            })
            
        return feedback

# Factory to create specialized feedback generators
class FeedbackGeneratorFactory:
    @staticmethod
    def create_generator(subject="math", level="k12", templates_file=None):
        """Create an appropriate feedback generator based on subject and level"""
        if subject.lower() == "math":
            if level.lower() in ["k12", "school", "elementary", "middle", "high"]:
                return MathFeedbackGenerator(templates_file)
            elif level.lower() in ["college", "university", "higher"]:
                return CollegeMathFeedbackGenerator(templates_file)
        
        # Default to general math feedback
        return MathFeedbackGenerator(templates_file)

# Example usage:
# generator = FeedbackGeneratorFactory.create_generator()
# analysis = {"question": "2x + 3 = 7", "student_answer": "x = 3", "correct_answer": "x = 2", 
#            "errors": [{"type": "calculation", "description": "Calculation error in solving for x"}]}
# feedback = generator.generate_feedback(analysis)
# print(feedback["feedback"])
