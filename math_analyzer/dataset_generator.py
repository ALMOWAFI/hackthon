import random
import json
import sympy
from sympy import symbols, Eq, solve, simplify, expand, factor
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import datetime

class MathProblemGenerator:
    """Generate diverse math problems across different categories and difficulty levels"""
    
    def __init__(self):
        self.problem_types = {
            "algebra_linear": self._generate_linear_equation,
            "algebra_quadratic": self._generate_quadratic_equation,
            "arithmetic_basic": self._generate_basic_arithmetic,
            "arithmetic_fractions": self._generate_fraction_problem,
            "geometry_area": self._generate_area_problem,
            "geometry_pythagorean": self._generate_pythagorean_problem,
            "calculus_derivative": self._generate_derivative_problem,
            "calculus_integral": self._generate_integral_problem
        }
        
        # Common student errors by problem type
        self.common_errors = {
            "algebra_linear": [
                lambda eq, sol: (eq, sol * -1, "sign error"),
                lambda eq, sol: (eq, sol + random.choice([1, 2]), "arithmetic error"),
                lambda eq, sol: (eq, sol - random.choice([1, 2]), "arithmetic error"),
                lambda eq, sol: (eq, sol * 2, "missed division step")
            ],
            "algebra_quadratic": [
                lambda eq, sols: (eq, [sols[0]], "incomplete factorization"),
                lambda eq, sols: (eq, [-s for s in sols], "sign error in factorization"),
                lambda eq, sols: (eq, [sols[0] + 1, sols[1] - 1], "arithmetic error")
            ],
            "arithmetic_basic": [
                lambda p, a: (p, a + random.choice([1, 2, -1, -2]), "arithmetic error"),
                lambda p, a: (p, a * random.choice([2, 10]), "place value error"),
                lambda p, a: (p, a // 2, "division instead of multiplication")
            ],
            "arithmetic_fractions": [
                lambda p, a: (p, simplify(a + 1), "numerator addition error"),
                lambda p, a: (p, simplify(1/a), "reciprocal confusion"),
                lambda p, a: (p, simplify(a*2), "denominator handling error")
            ]
        }
        
    def _generate_linear_equation(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate a linear equation problem"""
        x = symbols('x')
        
        # Adjust coefficients based on difficulty
        if difficulty == "easy":
            a = random.choice([1, 2, 3])
            b = random.randint(1, 10)
            c = random.randint(1, 20)
        elif difficulty == "medium":
            a = random.randint(2, 5)
            b = random.randint(-10, 10)
            c = random.randint(-20, 20)
        else:  # hard
            a = random.randint(2, 10) * random.choice([1, -1])
            b = random.randint(-15, 15)
            c = random.randint(-30, 30)
            
        # Generate the equation
        left_side = a*x + b
        right_side = c
        equation = Eq(left_side, right_side)
        
        # Solve
        solution = solve(equation, x)[0]
        
        # Format as string
        equation_str = f"{a}x + {b} = {c}" if b >= 0 else f"{a}x - {abs(b)} = {c}"
        
        return (f"Solve for x: {equation_str}", solution)
    
    def _generate_quadratic_equation(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate a quadratic equation problem"""
        x = symbols('x')
        
        # Generate roots based on difficulty
        if difficulty == "easy":
            # Simple integer roots
            root1 = random.randint(1, 5)
            root2 = random.randint(1, 5)
        elif difficulty == "medium":
            # One or both roots could be negative
            root1 = random.randint(-5, 5)
            while root1 == 0:  # Avoid zero for cleaner problems
                root1 = random.randint(-5, 5)
            root2 = random.randint(-5, 5)
        else:  # hard
            # Potentially fractional roots
            root1 = random.randint(-10, 10) / random.choice([1, 2, 5])
            root2 = random.randint(-10, 10) / random.choice([1, 2, 5])
            
        # Create the quadratic from roots
        expr = expand((x - root1) * (x - root2))
        
        # Leading coefficient variation for harder problems
        if difficulty == "hard":
            a = random.randint(2, 5)
            expr = expand(a * expr)
            
        # Format equation
        equation_str = str(expr) + " = 0"
        equation_str = equation_str.replace("**", "^")
        
        return (f"Solve for x: {equation_str}", [root1, root2])
    
    def _generate_basic_arithmetic(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate basic arithmetic problems"""
        if difficulty == "easy":
            # Single-digit addition or subtraction
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            op = random.choice(["+", "-"])
        elif difficulty == "medium":
            # Two-digit with all operations
            a = random.randint(10, 99)
            b = random.randint(1, 20)
            op = random.choice(["+", "-", "*", "/"])
            if op == "/":  # Ensure clean division
                a = b * random.randint(1, 10)
        else:  # hard
            # Multi-step operations
            a = random.randint(10, 50)
            b = random.randint(5, 20)
            c = random.randint(1, 10)
            op1 = random.choice(["+", "-", "*"])
            op2 = random.choice(["+", "-", "*"])
            return (f"Calculate: {a} {op1} {b} {op2} {c}", eval(f"{a} {op1} {b} {op2} {c}"))
        
        # Single operation result    
        result = eval(f"{a} {op} {b}")
        return (f"Calculate: {a} {op} {b}", result)
    
    def _generate_fraction_problem(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate problems involving fractions"""
        if difficulty == "easy":
            # Simple fractions with same denominator
            denom = random.randint(2, 10)
            num1 = random.randint(1, denom-1)
            num2 = random.randint(1, denom-1)
            op = random.choice(["+", "-"])
            expr = f"{num1}/{denom} {op} {num2}/{denom}"
        elif difficulty == "medium":
            # Different denominators
            denom1 = random.randint(2, 10)
            denom2 = random.randint(2, 10)
            while denom1 == denom2:
                denom2 = random.randint(2, 10)
            num1 = random.randint(1, denom1-1)
            num2 = random.randint(1, denom2-1)
            op = random.choice(["+", "-"])
            expr = f"{num1}/{denom1} {op} {num2}/{denom2}"
        else:  # hard
            # Mixed operations with fractions
            denom1 = random.randint(2, 12)
            denom2 = random.randint(2, 12)
            num1 = random.randint(1, denom1-1)
            num2 = random.randint(1, denom2-1)
            op = random.choice(["+", "-", "*", "/"])
            expr = f"{num1}/{denom1} {op} {num2}/{denom2}"
            
        result = sympy.sympify(expr.replace("/", "*1.0/"))
        return (f"Calculate and simplify: {expr}", result)
    
    def _generate_area_problem(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate geometric area problems"""
        if difficulty == "easy":
            # Rectangle or square
            if random.choice([True, False]):
                # Square
                side = random.randint(2, 10)
                return (f"Find the area of a square with side length {side} units.", side**2)
            else:
                # Rectangle
                length = random.randint(2, 10)
                width = random.randint(2, 10)
                return (f"Find the area of a rectangle with length {length} units and width {width} units.", length*width)
        elif difficulty == "medium":
            # Circle or triangle
            if random.choice([True, False]):
                # Circle
                radius = random.randint(1, 10)
                return (f"Find the area of a circle with radius {radius} units. Use π in your answer.", f"{radius**2}π")
            else:
                # Triangle
                base = random.randint(2, 10)
                height = random.randint(2, 10)
                return (f"Find the area of a triangle with base {base} units and height {height} units.", 0.5*base*height)
        else:  # hard
            # Composite shapes
            shape = random.choice(["trapezoid", "composite rectangle", "sector"])
            if shape == "trapezoid":
                a = random.randint(3, 10)
                b = random.randint(3, 10)
                h = random.randint(2, 8)
                return (f"Find the area of a trapezoid with parallel sides {a} and {b} units and height {h} units.", 0.5*(a+b)*h)
            elif shape == "composite rectangle":
                l1 = random.randint(2, 8)
                w1 = random.randint(2, 8)
                l2 = random.randint(2, 8)
                w2 = random.randint(2, 8)
                return (f"Find the total area of two rectangles with dimensions {l1}×{w1} and {l2}×{w2} units.", l1*w1 + l2*w2)
            else:  # sector
                radius = random.randint(3, 10)
                angle = random.choice([30, 45, 60, 90, 120, 180])
                return (f"Find the area of a sector with radius {radius} units and central angle {angle}°. Use π in your answer.", 
                       f"{radius**2 * angle/360}π")
    
    def _generate_pythagorean_problem(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate problems using the Pythagorean theorem"""
        if difficulty == "easy":
            # Find the hypotenuse with clean values
            a = random.choice([3, 4, 5, 6, 8, 10])
            b = random.choice([3, 4, 5, 6, 8, 10])
            c = (a**2 + b**2)**0.5
            if c == int(c):  # Only use if it's a clean value
                c = int(c)
                return (f"In a right triangle, if the legs have lengths {a} and {b} units, find the hypotenuse.", c)
            else:
                # Fallback to Pythagorean triple
                return self._generate_pythagorean_problem(difficulty)
        elif difficulty == "medium":
            # Find a leg
            a = random.choice([3, 4, 5, 6, 8, 10, 12])
            c = random.choice([5, 10, 13, 15, 17])
            if c > a:  # Ensure c is larger than a
                b_squared = c**2 - a**2
                b = b_squared**0.5
                if b == int(b):  # Only use if it's a clean value
                    b = int(b)
                    return (f"In a right triangle, if one leg is {a} units and the hypotenuse is {c} units, find the other leg.", b)
            # If values don't work, try again
            return self._generate_pythagorean_problem(difficulty)
        else:  # hard
            # Application problem
            scenario = random.choice([
                f"A ladder of length {random.randint(10, 20)} feet reaches {random.randint(6, 15)} feet up a wall. How far is the base of the ladder from the wall?",
                f"A rectangular garden has a diagonal of {random.randint(10, 20)} meters and a width of {random.randint(4, 8)} meters. Find its length.",
                f"A ship is {random.randint(8, 15)} miles north and {random.randint(5, 12)} miles east of port. How far is the ship from port in a direct line?"
            ])
            # For simplicity, we'll return a placeholder answer - in a real implementation, this would calculate the specific value
            return (scenario, "Calculation based on specific values")
    
    def _generate_derivative_problem(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate derivative problems (calculus)"""
        x = symbols('x')
        
        if difficulty == "easy":
            # Simple polynomial
            degree = random.randint(1, 3)
            coeffs = [random.randint(1, 5) for _ in range(degree+1)]
            expr = sum(coeffs[i] * x**i for i in range(degree+1))
            derivative = sympy.diff(expr, x)
            return (f"Find the derivative of f(x) = {expr}", derivative)
        elif difficulty == "medium":
            # Trigonometric or exponential functions
            func_type = random.choice(["trig", "exp", "log", "poly"])
            if func_type == "trig":
                a = random.randint(1, 5)
                expr = a * sympy.sin(x) if random.choice([True, False]) else a * sympy.cos(x)
            elif func_type == "exp":
                a = random.randint(1, 3)
                expr = a * sympy.exp(x)
            elif func_type == "log":
                expr = sympy.log(x)
            else:  # poly with negative or fractional exponents
                expr = x**random.choice([2, 3, -1, -2, 0.5, 0.25])
            derivative = sympy.diff(expr, x)
            return (f"Find the derivative of f(x) = {expr}", derivative)
        else:  # hard
            # Product, quotient, or chain rule problems
            rule_type = random.choice(["product", "quotient", "chain"])
            if rule_type == "product":
                f = x**random.randint(1, 3)
                g = sympy.sin(x) if random.choice([True, False]) else sympy.exp(x)
                expr = f * g
            elif rule_type == "quotient":
                f = x**random.randint(1, 3)
                g = x**random.randint(1, 2) + random.randint(1, 5)
                expr = f / g
            else:  # chain
                inner = random.randint(1, 3) * x + random.randint(0, 5)
                outer_type = random.choice(["sin", "exp", "sqrt"])
                if outer_type == "sin":
                    expr = sympy.sin(inner)
                elif outer_type == "exp":
                    expr = sympy.exp(inner)
                else:
                    expr = sympy.sqrt(inner)
            derivative = sympy.diff(expr, x)
            return (f"Find the derivative of f(x) = {expr}", derivative)
    
    def _generate_integral_problem(self, difficulty="medium") -> Tuple[str, Any]:
        """Generate integral problems (calculus)"""
        x = symbols('x')
        
        if difficulty == "easy":
            # Simple polynomial
            degree = random.randint(1, 3)
            coeffs = [random.randint(1, 5) for _ in range(degree+1)]
            expr = sum(coeffs[i] * x**i for i in range(degree+1))
            integral = sympy.integrate(expr, x)
            return (f"Find the indefinite integral of f(x) = {expr}", integral)
        elif difficulty == "medium":
            # Trigonometric, exponential, or 1/x type
            func_type = random.choice(["trig", "exp", "rational"])
            if func_type == "trig":
                a = random.randint(1, 5)
                expr = a * sympy.sin(x) if random.choice([True, False]) else a * sympy.cos(x)
            elif func_type == "exp":
                a = random.randint(1, 3)
                expr = a * sympy.exp(x)
            else:  # rational
                expr = 1/x
            integral = sympy.integrate(expr, x)
            return (f"Find the indefinite integral of f(x) = {expr}", integral)
        else:  # hard
            # Definite integrals or u-substitution problems
            int_type = random.choice(["definite", "u-sub"])
            if int_type == "definite":
                a = random.randint(0, 3)
                b = random.randint(a+1, 5)
                expr = x**random.randint(1, 3)
                integral = sympy.integrate(expr, (x, a, b))
                return (f"Evaluate the definite integral of f(x) = {expr} from x = {a} to x = {b}", integral)
            else:  # u-sub
                # Create an expression that would require u-substitution
                u = random.randint(1, 3) * x + random.randint(0, 3)
                du = sympy.diff(u, x)
                expr = sympy.sin(u) * du if random.choice([True, False]) else u**random.randint(1, 3) * du
                integral = sympy.integrate(expr, x)
                return (f"Find the indefinite integral of f(x) = {expr}", integral)

    def generate_problem(self, problem_type=None, difficulty="medium"):
        """
        Generate a math problem of the specified type and difficulty
        
        Args:
            problem_type: Type of problem to generate (if None, randomly chosen)
            difficulty: "easy", "medium", or "hard"
            
        Returns:
            tuple: (problem_text, correct_answer)
        """
        # If no type specified, choose randomly
        if problem_type is None:
            problem_type = random.choice(list(self.problem_types.keys()))
            
        # If type not recognized, default to algebra
        if problem_type not in self.problem_types:
            problem_type = "algebra_linear"
            
        # Generate the problem
        generator = self.problem_types[problem_type]
        return generator(difficulty)
        
    def generate_incorrect_solution(self, problem_type, problem, correct_answer):
        """
        Generate common incorrect solutions for a given problem
        
        Args:
            problem_type: Type of the problem
            problem: The problem text
            correct_answer: The correct answer
            
        Returns:
            tuple: (problem, incorrect_answer, error_description)
        """
        # If errors defined for this type, use them
        if problem_type in self.common_errors:
            error_generator = random.choice(self.common_errors[problem_type])
            return error_generator(problem, correct_answer)
            
        # Default fallback if no specific errors defined
        if isinstance(correct_answer, (int, float)):
            incorrect = correct_answer + random.choice([1, -1, 2, -2])
            return (problem, incorrect, "arithmetic error")
        elif isinstance(correct_answer, list):
            if len(correct_answer) > 0:
                incorrect = correct_answer.copy()
                incorrect[0] += random.choice([1, -1])
                return (problem, incorrect, "calculation error")
        
        # For complex answers, just return a generic error
        return (problem, "incorrect answer", "conceptual error")
        
class DatasetGenerator:
    """Generate datasets for training math feedback models"""
    
    def __init__(self, problem_generator=None):
        self.problem_generator = problem_generator or MathProblemGenerator()
        
    def generate_examples(self, n=100, problem_types=None, 
                         error_rate=0.7, output_format="json"):
        """
        Generate n examples for model training
        
        Args:
            n: Number of examples to generate
            problem_types: List of problem types to include (None for all)
            error_rate: Fraction of examples that should contain errors
            output_format: "json" or "gemini" (for Gemini fine-tuning)
            
        Returns:
            List of examples in the specified format
        """
        examples = []
        
        # If no types specified, use all available
        if problem_types is None:
            problem_types = list(self.problem_generator.problem_types.keys())
        
        # Generate the specified number of examples
        for i in range(n):
            # Randomly select problem type from available options
            problem_type = random.choice(problem_types)
            
            # Randomly select difficulty
            difficulty = random.choice(["easy", "medium", "hard"])
            
            # Generate a problem and correct answer
            problem, correct_answer = self.problem_generator.generate_problem(
                problem_type, difficulty)
            
            # Decide if this example should have an error
            has_error = random.random() < error_rate
            
            if has_error:
                # Generate incorrect solution
                _, student_answer, error_description = self.problem_generator.generate_incorrect_solution(
                    problem_type, problem, correct_answer)
                score = random.randint(0, 80)  # Lower score for incorrect answers
            else:
                # Use correct solution
                student_answer = correct_answer
                error_description = None
                score = random.randint(80, 100)  # Higher score for correct answers
            
            # Format the student solution as a string
            if isinstance(student_answer, (list, tuple)):
                student_solution = ", ".join(str(x) for x in student_answer)
            else:
                student_solution = str(student_answer)
                
            # Format the correct answer as a string
            if isinstance(correct_answer, (list, tuple)):
                correct_solution = ", ".join(str(x) for x in correct_answer)
            else:
                correct_solution = str(correct_answer)
            
            # Create the example
            example = {
                "problem_type": problem_type,
                "difficulty": difficulty,
                "problem": problem,
                "student_solution": student_solution,
                "correct_solution": correct_solution,
                "score": score
            }
            
            if error_description:
                example["error_description"] = error_description
                
            examples.append(example)
        
        # Format examples according to output format
        if output_format.lower() == "gemini":
            return self._format_for_gemini(examples)
        else:
            return examples
    
    def _format_for_gemini(self, examples):
        """Format examples for Gemini fine-tuning"""
        gemini_examples = []
        
        for example in examples:
            # Create input content with the problem and student solution
            input_content = f"Please grade this math problem:\n\n{example['problem']}\n\nStudent solution: {example['student_solution']}"
            
            # Create appropriate feedback based on correctness
            if example.get('error_description'):
                # Create detailed feedback for incorrect solutions
                feedback = self._generate_feedback(example)
            else:
                # Positive feedback for correct solutions
                feedback = (f"# Homework Evaluation\n\n"
                           f"## Problem\n{example['problem']}\n\n"
                           f"## Student Solution\n{example['student_solution']}\n\n"
                           f"## Assessment\n✓ Correct solution\n\n"
                           f"The student has solved this problem correctly. The answer {example['student_solution']} is accurate.\n\n"
                           f"## Practice Recommendations\n"
                           f"Since the student has mastered this type of problem, they can move on to more challenging problems.")
            
            # Format in Gemini's expected structure
            gemini_example = {
                "input": {
                    "role": "user",
                    "content": input_content
                },
                "output": {
                    "role": "assistant",
                    "content": feedback
                }
            }
            
            gemini_examples.append(gemini_example)
            
        return {"examples": gemini_examples}
    
    def _generate_feedback(self, example):
        """Generate structured feedback for an incorrect solution"""
        # Extract information
        problem = example['problem']
        student_solution = example['student_solution']
        correct_solution = example['correct_solution']
        error_description = example.get('error_description', 'calculation error')
        problem_type = example['problem_type']
        
        # Generate practice problems
        practice_problems = []
        for _ in range(3):
            p, a = self.problem_generator.generate_problem(problem_type, example['difficulty'])
            practice_problems.append(p)
        
        # Create structured feedback
        feedback = f"# Homework Evaluation\n\n"
        feedback += f"## Problem\n{problem}\n\n"
        feedback += f"## Student Solution\n{student_solution}\n\n"
        feedback += f"## Assessment\n"
        
        # Add error markers
        if "arithmetic" in error_description:
            feedback += f"❌ Arithmetic error found\n"
        elif "procedure" in error_description:
            feedback += f"❌ Error in problem-solving procedure\n"
        elif "conceptual" in error_description:
            feedback += f"❌ Conceptual misunderstanding\n"
        else:
            feedback += f"❌ Incorrect answer\n"
        
        # Add detailed explanation
        feedback += f"\nThe student made a {error_description}. "
        
        if "arithmetic" in error_description:
            feedback += f"The calculations should be checked more carefully. "
        elif "procedure" in error_description:
            feedback += f"The approach has a flaw in the steps taken. "
        elif "sign" in error_description:
            feedback += f"There is a sign error in the work. "
        
        feedback += f"The correct answer is {correct_solution}.\n\n"
        
        # Add practice recommendations
        feedback += f"## Practice Recommendations\n"
        feedback += f"The student needs more practice with:\n"
        
        if "arithmetic" in error_description:
            feedback += f"1. Careful calculation without arithmetic errors\n"
        elif "procedure" in error_description:
            feedback += f"1. Step-by-step problem-solving procedures\n"
        elif "conceptual" in error_description:
            feedback += f"1. Understanding the core concepts of {problem_type.replace('_', ' ')}\n"
        
        feedback += f"\nHere are 3 practice problems:\n"
        for i, problem in enumerate(practice_problems):
            feedback += f"{i+1}. {problem}\n"
        
        return feedback
    
    def save_dataset(self, examples, filename=None):
        """Save generated examples to a file"""
        if filename is None:
            # Create a filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"math_examples_{timestamp}.json"
            
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save to file
        with open(filename, 'w') as f:
            json.dump(examples, f, indent=2)
            
        return filename
        
# Example usage:
# generator = DatasetGenerator()
# examples = generator.generate_examples(5, output_format="gemini")
# generator.save_dataset(examples, "examples/gemini_format_examples.json")
