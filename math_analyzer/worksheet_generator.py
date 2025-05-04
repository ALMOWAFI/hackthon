"""
Practice Worksheet Generator for Math Feedback System

This module generates customized practice worksheets based on student performance,
targeting specific skills and concepts where improvement is needed.
"""

import os
import json
from datetime import datetime
import re
import random
from pathlib import Path
from typing import List, Dict, Any
from .vllm_client import VLLMClient

class WorksheetGenerator:
    """Generates practice worksheets tailored to student needs"""
    
    def __init__(self, vllm_url=None, vllm_key=None):
        """
        Initialize the worksheet generator
        
        Args:
            vllm_url: URL of the vLLM API endpoint
            vllm_key: API key for vLLM authentication
        """
        self.vllm_client = VLLMClient(api_url=vllm_url, api_key=vllm_key)
        
        # Problem difficulty progression
        self.difficulty_levels = ["easy", "medium", "hard"]
        
        # Mapping of error types to skills to practice
        self.error_to_skill_map = {
            "CALCULATION": "basic arithmetic operations",
            "PROCEDURAL": "step-by-step problem solving",
            "CONCEPTUAL": "understanding mathematical concepts",
            "SYNTAX": "mathematical notation and symbols",
            "ALGEBRAIC": "algebraic manipulation",
            "GEOMETRIC": "geometric properties and relationships",
            "ARITHMETIC": "arithmetic operations and number properties",
            "CONCEPTUAL_MISUNDERSTANDING": "conceptual foundations",
            "PROCESS_ERROR": "mathematical processes and procedures",
            "LOGICAL_ERROR": "logical reasoning in mathematics",
            "NOTATION_ERROR": "correct mathematical notation",
            "APPLICATION_ERROR": "applying concepts to problems"
        }
        
        # Mapping of problem types to topics
        self.type_to_topic_map = {
            "LINEAR_EQUATION": "linear equations",
            "QUADRATIC_EQUATION": "quadratic equations",
            "FRACTION": "fractions",
            "PYTHAGOREAN": "Pythagorean theorem",
            "SYSTEM_OF_EQUATIONS": "systems of equations",
            "ALGEBRAIC_EXPRESSION": "algebraic expressions",
            "CALCULUS_DERIVATIVE": "derivatives",
            "CALCULUS_INTEGRAL": "integrals",
            "PROBABILITY": "probability",
            "STATISTICS": "statistics",
            "UNKNOWN": "general mathematics"
        }
    
    def generate_worksheet(self, feedback_data, output_dir="results", filename=None):
        """
        Generate a practice worksheet based on feedback data
        
        Args:
            feedback_data: Feedback data from the math analysis system
            output_dir: Directory to save the worksheet
            filename: Optional filename for the worksheet
            
        Returns:
            Path to the generated worksheet file
        """
        # Extract problem types and errors from feedback
        problem_types = {}
        error_types = {}
        
        # Process each problem in the feedback data
        for problem in feedback_data.get('problems', []):
            # Count problem types
            problem_type = problem.get('math_type', 'UNKNOWN')
            if problem_type in problem_types:
                problem_types[problem_type] += 1
            else:
                problem_types[problem_type] = 1
                
            # Count error types
            for error in problem.get('analysis', {}).get('errors', []):
                error_type = error.get('type', 'UNKNOWN')
                if error_type in error_types:
                    error_types[error_type] += 1
                else:
                    error_types[error_type] = 1
        
        # Determine the primary areas for practice
        primary_problem_type = max(problem_types.items(), key=lambda x: x[1])[0] if problem_types else "UNKNOWN"
        primary_error_type = max(error_types.items(), key=lambda x: x[1])[0] if error_types else "UNKNOWN"
        
        # Map to topics and skills
        topic = self.type_to_topic_map.get(primary_problem_type, "general mathematics")
        skill = self.error_to_skill_map.get(primary_error_type, "problem solving")
        
        # Generate practice problems using vLLM
        worksheet_data = {
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'topic': topic,
            'skill': skill,
            'based_on': {
                'primary_problem_type': primary_problem_type,
                'primary_error_type': primary_error_type
            },
            'sections': []
        }
        
        # Create easy problems section
        easy_problems = self.vllm_client.generate_practice_worksheet(
            topic=topic,
            skill=skill,
            count=3,
            difficulty="easy"
        )
        worksheet_data['sections'].append({
            'title': f"Basic {topic.title()} Problems",
            'description': f"These problems focus on building fundamental skills in {topic}.",
            'problems': easy_problems
        })
        
        # Create medium problems section
        medium_problems = self.vllm_client.generate_practice_worksheet(
            topic=topic,
            skill=skill,
            count=2,
            difficulty="medium"
        )
        worksheet_data['sections'].append({
            'title': f"Intermediate {topic.title()} Problems",
            'description': f"These problems will help you build confidence with {topic}.",
            'problems': medium_problems
        })
        
        # Create challenge problem section
        challenge_problem = self.vllm_client.generate_practice_worksheet(
            topic=topic,
            skill=skill,
            count=1,
            difficulty="hard"
        )
        worksheet_data['sections'].append({
            'title': f"Challenge Problem",
            'description': "This problem will test your understanding at a deeper level.",
            'problems': challenge_problem
        })
        
        # Add tips and study recommendations
        worksheet_data['study_tips'] = [
            f"Focus on understanding the core concepts of {topic}.",
            f"Practice step-by-step problem-solving to avoid {primary_error_type.lower().replace('_', ' ')} errors.",
            "Check your work carefully after solving each problem.",
            "Try explaining the solution process out loud to reinforce your understanding."
        ]
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"practice_worksheet_{timestamp}.json"
        
        # Save the worksheet data as JSON
        worksheet_path = os.path.join(output_dir, filename)
        with open(worksheet_path, 'w', encoding='utf-8') as f:
            json.dump(worksheet_data, f, indent=2)
        
        return worksheet_path
    
    def generate_html_worksheet(self, worksheet_data, output_dir="results", filename=None):
        """
        Generate an HTML version of the practice worksheet
        
        Args:
            worksheet_data: The worksheet data (can be path to JSON or dictionary)
            output_dir: Directory to save the HTML worksheet
            filename: Optional filename for the HTML worksheet
            
        Returns:
            Path to the HTML worksheet
        """
        # Load the worksheet data if a path is provided
        if isinstance(worksheet_data, (str, Path)) and os.path.exists(worksheet_data):
            with open(worksheet_data, 'r', encoding='utf-8') as f:
                worksheet_data = json.load(f)
        
        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Practice Worksheet: {worksheet_data['topic'].title()}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .section {{
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        .problem {{
            background-color: white;
            border-left: 4px solid #6e8efb;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }}
        .solution {{
            background-color: #f0f8ff;
            border-left: 4px solid #a777e3;
            padding: 15px;
            margin-top: 10px;
            border-radius: 4px;
            display: none;
        }}
        .problem-number {{
            font-weight: bold;
            color: #6e8efb;
        }}
        .show-solution {{
            background-color: #6e8efb;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }}
        .show-solution:hover {{
            background-color: #5d7ce0;
        }}
        .tips {{
            background-color: #fff8e1;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        h1, h2, h3 {{
            color: #444;
        }}
        ul {{
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Practice Worksheet: {worksheet_data['topic'].title()}</h1>
        <p>Focus Skill: {worksheet_data['skill'].title()}</p>
        <p>Generated: {worksheet_data['generated_at']}</p>
    </div>
"""
        
        # Add each section and its problems
        problem_counter = 1
        for section in worksheet_data['sections']:
            html_content += f"""
    <div class="section">
        <h2>{section['title']}</h2>
        <p>{section['description']}</p>
"""
            
            for problem in section['problems']:
                html_content += f"""
        <div class="problem">
            <p><span class="problem-number">Problem {problem_counter}:</span> {problem['problem']}</p>
            <button class="show-solution" onclick="toggleSolution('solution-{problem_counter}')">Show Solution</button>
            <div class="solution" id="solution-{problem_counter}">
                <h3>Solution:</h3>
                <p>{problem['steps'].replace('\n', '<br>')}</p>
                <h3>Answer:</h3>
                <p>{problem['answer']}</p>
            </div>
        </div>
"""
                problem_counter += 1
                
            html_content += "    </div>\n"
        
        # Add study tips section
        html_content += """
    <div class="tips">
        <h2>Study Tips</h2>
        <ul>
"""
        
        for tip in worksheet_data['study_tips']:
            html_content += f"            <li>{tip}</li>\n"
            
        html_content += """
        </ul>
    </div>
    
    <script>
        function toggleSolution(id) {
            const solution = document.getElementById(id);
            if (solution.style.display === 'block') {
                solution.style.display = 'none';
            } else {
                solution.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            if isinstance(worksheet_data, dict) and 'generated_at' in worksheet_data:
                timestamp = worksheet_data['generated_at'].replace(' ', '_').replace(':', '')
                filename = f"practice_worksheet_{timestamp}.html"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"practice_worksheet_{timestamp}.html"
        
        # Ensure the filename has the .html extension
        if not filename.lower().endswith('.html'):
            filename += '.html'
        
        # Save the HTML content
        html_path = os.path.join(output_dir, filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path


# Example usage
if __name__ == "__main__":
    generator = WorksheetGenerator()
    
    # Example feedback data
    feedback_data = {
        'problems': [
            {
                'math_type': 'LINEAR_EQUATION',
                'analysis': {
                    'errors': [
                        {'type': 'ALGEBRAIC', 'description': 'Error in solving for the variable'}
                    ]
                }
            },
            {
                'math_type': 'LINEAR_EQUATION',
                'analysis': {
                    'errors': [
                        {'type': 'CALCULATION', 'description': 'Error in arithmetic calculation'}
                    ]
                }
            }
        ]
    }
    
    # Generate the worksheet
    worksheet_path = generator.generate_worksheet(feedback_data)
    print(f"Generated worksheet: {worksheet_path}")
    
    # Generate HTML version
    html_path = generator.generate_html_worksheet(worksheet_path)
    print(f"Generated HTML worksheet: {html_path}")
