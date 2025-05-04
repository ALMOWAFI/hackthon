import json
import os
from pathlib import Path
from typing import Dict, List, Any
from feedback_generator import MathFeedbackGenerator, TeachingStyle, ErrorType
from paper_grading import PaperGradingSystem
import random
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import re

class MathFeedbackTrainer:
    def __init__(self):
        self.feedback_generator = MathFeedbackGenerator()
        self.grading_system = PaperGradingSystem(subject_area="math")
        self.data_path = Path(__file__).parent.parent / "data" / "processed"
        self.seed_templates = self._initialize_seed_templates()
        
    def load_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        datasets = {}
        
        # Load math instruction dataset
        math_instruct_path = self.data_path / "mathinstruct.json"
        if math_instruct_path.exists():
            with open(math_instruct_path, 'r', encoding='utf-8') as f:
                datasets['math_instructions'] = json.load(f)
                
        # Load student performance dataset
        student_perf_path = self.data_path / "student_performance2.json"
        if student_perf_path.exists():
            with open(student_perf_path, 'r', encoding='utf-8') as f:
                datasets['student_performance'] = json.load(f)
        
        return datasets
    
    def preprocess_data(self, datasets: Dict[str, List[Dict[str, Any]]]):
        processed_data = []
        
        # Process math instructions
        if 'math_instructions' in datasets:
            for item in datasets['math_instructions']:
                processed_item = {
                    'question': item.get('question', ''),
                    'solution': item.get('solution', ''),
                    'explanation': item.get('explanation', ''),
                    'difficulty': item.get('difficulty', 'medium'),
                    'concepts': item.get('concepts', []),
                    'error_types': []
                }
                processed_data.append(processed_item)
        
        # Process student performance data
        if 'student_performance' in datasets:
            for item in datasets['student_performance']:
                if 'student_answer' in item and 'correct_answer' in item:
                    error_types = self._identify_error_types(
                        item['student_answer'], 
                        item['correct_answer']
                    )
                    item['error_types'] = error_types
                    processed_data.append(item)
        
        return processed_data
    
    def _identify_error_types(self, student_answer: str, correct_answer: str) -> List[ErrorType]:
        error_types = []
        
        # Normalize answers for comparison
        student = self._normalize_math_expression(student_answer)
        correct = self._normalize_math_expression(correct_answer)
        
        # Calculation error detection
        if self._has_calculation_error(student, correct):
            error_types.append(ErrorType.CALCULATION)
            
        # Procedural error detection
        if self._has_procedural_error(student, correct):
            error_types.append(ErrorType.PROCEDURAL)
            
        # Notation error detection
        if self._has_notation_error(student_answer, correct_answer):
            error_types.append(ErrorType.NOTATION)
            
        # Conceptual error detection
        if self._has_conceptual_error(student, correct):
            error_types.append(ErrorType.CONCEPTUAL)
        
        return error_types
    
    def _normalize_math_expression(self, expr: str) -> str:
        # Remove spaces and convert to lowercase
        expr = expr.lower().replace(' ', '')
        
        # Standardize multiplication symbols
        expr = expr.replace('×', '*').replace('·', '*')
        
        # Standardize division
        expr = expr.replace('÷', '/')
        
        # Standardize equals sign
        expr = expr.replace('==', '=')
        
        return expr
    
    def _has_calculation_error(self, student: str, correct: str) -> bool:
        try:
            # Extract numerical values
            student_nums = [float(n) for n in re.findall(r'-?\d*\.?\d+', student)]
            correct_nums = [float(n) for n in re.findall(r'-?\d*\.?\d+', correct)]
            
            # Check if numbers are present but different
            if student_nums and correct_nums:
                # Allow for small floating point differences
                return any(abs(s - c) > 0.0001 for s, c in zip(student_nums, correct_nums))
            return False
        except ValueError:
            return False
    
    def _has_procedural_error(self, student: str, correct: str) -> bool:
        # Check for missing or extra steps
        student_steps = self._extract_steps(student)
        correct_steps = self._extract_steps(correct)
        
        if len(student_steps) != len(correct_steps):
            return True
            
        # Check operation order
        student_ops = re.findall(r'[+\-*/=]', student)
        correct_ops = re.findall(r'[+\-*/=]', correct)
        
        return student_ops != correct_ops
    
    def _extract_steps(self, expr: str) -> List[str]:
        # Split expression into steps based on equals signs or line breaks
        steps = re.split(r'=|\n', expr)
        return [s.strip() for s in steps if s.strip()]
    
    def _has_notation_error(self, student: str, correct: str) -> bool:
        # Check for common notation errors
        notation_patterns = [
            (r'\d[a-zA-Z]', r'[a-zA-Z]\d'),  # Missing multiplication symbol
            (r'\(\)', r'\(.*\)'),  # Empty parentheses
            (r'[+\-*/]=', r'='),  # Operation signs next to equals
            (r'\d\(', r'\d *\('),  # Missing multiplication before parentheses
        ]
        
        for student_pattern, correct_pattern in notation_patterns:
            student_match = bool(re.search(student_pattern, student))
            correct_match = bool(re.search(correct_pattern, correct))
            if student_match != correct_match:
                return True
        
        return False
    
    def _has_conceptual_error(self, student: str, correct: str) -> bool:
        # Check for fundamental misunderstandings
        conceptual_indicators = [
            # Wrong operation entirely
            ('+' in student and '*' in correct) or ('*' in student and '+' in correct),
            # Division/multiplication confusion
            ('/' in student and '*' in correct) or ('*' in student and '/' in correct),
            # Sign errors (positive vs negative)
            (student.count('-') != correct.count('-')),
            # Variable isolation errors
            (student.count('x') != correct.count('x')),
        ]
        
        return any(conceptual_indicators)
    
    def _initialize_seed_templates(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            TeachingStyle.SOCRATIC.value: {
                "calculation": [
                    "What steps did you take to arrive at {student_answer}? Let's check each step.",
                    "Can you explain how you got from {step1} to {step2}?",
                    "What would happen if we tried a different approach to solve this?"
                ],
                "procedural": [
                    "What rule or method did you use here? Is there another way to approach this?",
                    "Could you break down your solution into smaller steps?",
                    "What is the first step in solving this type of problem?"
                ],
                "conceptual": [
                    "How does this problem relate to {concept}?",
                    "What are the key mathematical ideas we're working with here?",
                    "Can you think of a similar problem we've solved before?"
                ],
                "notation": [
                    "What does this symbol mean in mathematics?",
                    "How else could we write this expression?",
                    "Why do we use this particular notation?"
                ]
            },
            TeachingStyle.DIRECT.value: {
                "calculation": [
                    "Let's solve this step by step. First, {step1}",
                    "The calculation error is here: {error_point}. The correct step is {correction}",
                    "Remember to check your calculations at each step"
                ],
                "procedural": [
                    "Follow these steps: 1. {step1} 2. {step2} 3. {step3}",
                    "The correct procedure is to first {step1}, then {step2}",
                    "You need to apply {rule_name} here"
                ],
                "conceptual": [
                    "This problem is about {concept}. Here's how it works...",
                    "The key principle here is {principle}",
                    "Let's review the definition of {term}"
                ],
                "notation": [
                    "Use {correct_symbol} instead of {incorrect_symbol}",
                    "In math, we write this as {correct_notation}",
                    "The proper way to write this is {example}"
                ]
            }
        }
    
    def train(self):
        print("Loading datasets...")
        datasets = self.load_datasets()
        
        print("Preprocessing data...")
        processed_data = self.preprocess_data(datasets)
        
        # Split data into train/validation/test sets
        train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        print(f"Training on {len(train_data)} samples...")
        print(f"Validation set: {len(val_data)} samples")
        print(f"Test set: {len(test_data)} samples")
        
        # Train feedback templates for different teaching styles
        for style in TeachingStyle:
            print(f"\nTraining {style.value} teaching style...")
            style_templates = self._generate_style_templates(train_data, style)
            
            # Validate templates on validation set
            val_accuracy = self._validate_templates(style_templates, val_data)
            print(f"Validation accuracy for {style.value}: {val_accuracy:.2f}%")
            
            self.feedback_generator.templates[style.value] = style_templates
        
        # Train error type detection
        print("\nTraining error type detection...")
        self._train_error_detection(train_data)
        
        # Evaluate on test set
        print("\nEvaluating model...")
        accuracy = self._evaluate(test_data)
        print(f"Test accuracy: {accuracy:.2f}%")
        
        # Save trained templates
        self._save_templates()
    
    def _validate_templates(self, templates: Dict[str, List[str]], val_data: List[Dict[str, Any]]) -> float:
        correct = 0
        total = 0
        
        for item in val_data:
            if 'error_types' in item:
                for error_type in item['error_types']:
                    if error_type.value in templates and templates[error_type.value]:
                        # Check if generated feedback is appropriate
                        feedback = random.choice(templates[error_type.value])
                        if self._is_valid_feedback(feedback, item):
                            correct += 1
                        total += 1
        
        return (correct / total * 100) if total > 0 else 0
    
    def _is_valid_feedback(self, feedback: str, item: Dict[str, Any]) -> bool:
        # Basic validation checks
        if not feedback or len(feedback) < 10:
            return False
            
        # Check if feedback references key concepts
        if 'concepts' in item:
            return any(concept.lower() in feedback.lower() for concept in item['concepts'])
            
        # Check if feedback matches difficulty level
        if 'difficulty' in item:
            words = len(feedback.split())
            if item['difficulty'] == 'easy' and words > 50:
                return False
            if item['difficulty'] == 'hard' and words < 20:
                return False
                
        return True
    
    def _generate_style_templates(self, train_data: List[Dict[str, Any]], style: TeachingStyle) -> Dict[str, List[str]]:
        templates = {
            'calculation': [],
            'procedural': [],
            'conceptual': [],
            'notation': []
        }
        
        # Start with seed templates
        if style.value in self.seed_templates:
            for error_type in templates:
                templates[error_type].extend(self.seed_templates[style.value][error_type])
        
        # Generate new templates from training data
        for item in train_data:
            if 'explanation' in item and item['explanation'].strip():
                error_types = item.get('error_types', [])
                for error_type in error_types:
                    if error_type.value in templates:
                        # Extract concepts and steps
                        concepts = item.get('concepts', [])
                        steps = self._extract_steps(item['explanation'])
                        
                        # Generate template
                        template = self._create_feedback_template(
                            item['explanation'],
                            style,
                            error_type,
                            concepts=concepts,
                            steps=steps
                        )
                        if template and self._is_valid_feedback(template, item):
                            templates[error_type.value].append(template)
        
        return templates
    
    def _create_feedback_template(self, explanation: str, style: TeachingStyle, 
                                error_type: ErrorType, concepts=None, steps=None) -> str:
        concepts = concepts or []
        steps = steps or []
        
        if style == TeachingStyle.SOCRATIC:
            if error_type == ErrorType.CALCULATION:
                return f"What would happen if we checked {steps[0] if steps else 'this step'} again?"
            elif error_type == ErrorType.PROCEDURAL:
                return f"How would you explain your approach to solving this problem?"
            elif error_type == ErrorType.CONCEPTUAL:
                concept = concepts[0] if concepts else "this concept"
                return f"How does {concept} apply to this problem?"
            else:  # NOTATION
                return "Why did you choose to write it this way?"
                
        elif style == TeachingStyle.DIRECT:
            if error_type == ErrorType.CALCULATION:
                step = steps[0] if steps else "this calculation"
                return f"Check {step} carefully. Make sure to..."
            elif error_type == ErrorType.PROCEDURAL:
                return f"Follow these steps: {' Then '.join(steps)}" if steps else explanation
            elif error_type == ErrorType.CONCEPTUAL:
                concept = concepts[0] if concepts else "this concept"
                return f"Remember that {concept} means..."
            else:  # NOTATION
                return f"The correct notation is..."
                
        elif style == TeachingStyle.GROWTH_MINDSET:
            return f"Let's learn from this! {explanation}"
            
        elif style == TeachingStyle.CONCEPTUAL:
            concept = concepts[0] if concepts else "this concept"
            return f"Let's understand why {concept} works here..."
            
        elif style == TeachingStyle.VISUAL:
            return f"Let's draw this out to see..."
            
        elif style == TeachingStyle.PROCEDURAL:
            steps_text = " Then ".join(steps) if steps else explanation
            return f"Let's break this down into steps: {steps_text}"
            
        elif style == TeachingStyle.ANALOGICAL:
            return f"This is similar to..."
            
        else:  # GAMIFIED
            return f"Let's turn this into a challenge!"
    
    def _train_error_detection(self, train_data: List[Dict[str, Any]]):
        # Train error detection patterns
        self.error_patterns = {
            ErrorType.CALCULATION: set(),
            ErrorType.PROCEDURAL: set(),
            ErrorType.NOTATION: set(),
            ErrorType.CONCEPTUAL: set()
        }
        
        for item in train_data:
            if 'error_types' in item:
                for error_type in item['error_types']:
                    if 'student_answer' in item:
                        self.error_patterns[error_type].add(
                            self._extract_error_pattern(item['student_answer'])
                        )
    
    def _extract_error_pattern(self, answer: str) -> str:
        # Simple pattern extraction
        return ''.join(c if c.isalnum() else '_' for c in answer)
    
    def _evaluate(self, test_data: List[Dict[str, Any]]) -> float:
        correct = 0
        total = 0
        
        for item in test_data:
            if 'student_answer' in item and 'correct_answer' in item:
                predicted_errors = self._identify_error_types(
                    item['student_answer'],
                    item['correct_answer']
                )
                actual_errors = item.get('error_types', [])
                
                if set(predicted_errors) == set(actual_errors):
                    correct += 1
                total += 1
        
        return (correct / total * 100) if total > 0 else 0
    
    def _save_templates(self):
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "trained_templates.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_generator.templates, f, indent=2)
        
        print(f"\nSaved trained templates to {output_path}")

if __name__ == "__main__":
    trainer = MathFeedbackTrainer()
    trainer.train()
