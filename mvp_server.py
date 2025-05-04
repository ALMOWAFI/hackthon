#!/usr/bin/env python3
"""
MVP Server for Math Homework Analysis System

This is a simple Flask server that serves as an MVP integration point
between the web interface and the existing Azure-powered math homework analysis.
"""

import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

# Import the analysis modules
sys.path.append('.')
try:
    from math_analyzer.improved_error_localization import MathErrorDetector
    from hackthon.hackthon.azure_math_integration import AzureMathRecognizer, AzureEnhancedTutor
    AZURE_AVAILABLE = True
except ImportError:
    print("Warning: Azure integration modules not found. Running in demo mode.")
    AZURE_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Allow cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize the analyzers
if AZURE_AVAILABLE:
    azure_math_recognizer = AzureMathRecognizer()
    azure_tutor = AzureEnhancedTutor()
math_detector = MathErrorDetector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'math_analyzer_mvp.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_homework():
    """API endpoint to analyze a homework image"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
            
        # Save the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, 'temp_homework.jpg')
        image_file.save(image_path)
        
        # Determine if we're using Azure or demo mode
        use_azure = AZURE_AVAILABLE and request.form.get('use_azure', 'true').lower() == 'true'
        
        # Process with Azure if available and requested
        if use_azure:
            logger.info("Processing with Azure integration...")
            try:
                # Analyze with Azure Form Recognizer
                analysis_result = azure_math_recognizer.analyze_math_document(image_path)
                
                # Generate feedback with Azure OpenAI
                if 'equations' in analysis_result:
                    for eq in analysis_result['equations']:
                        eq['feedback'] = azure_tutor.generate_feedback(
                            eq['text'], 
                            eq['confidence'],
                            'encouraging'
                        )
                
                # Transform to three-stage format
                result = transform_to_template(analysis_result)
                
            except Exception as e:
                logger.error(f"Azure processing error: {str(e)}")
                # Fall back to demo mode
                result = get_demo_result()
        else:
            # Use demo mode with template
            logger.info("Using demo mode...")
            result = get_demo_result()
        
        # Save the result
        result_path = os.path.join(RESULTS_FOLDER, 'latest_analysis.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Error in analyze_homework endpoint")
        return jsonify({"error": str(e)}), 500

def transform_to_template(azure_result):
    """Transform Azure analysis result to the three-stage template format"""
    # Start with a template structure
    template = {
        "student_info": {
            "name": "Student",
            "grade_level": "Elementary"
        },
        "assignment_info": {
            "title": "Math Homework",
            "date": "2025-05-04",
            "total_problems": 0,
            "total_correct": 0,
            "score_percentage": 0
        },
        "problems": [],
        "summary": {
            "strengths": [],
            "areas_for_improvement": [],
            "teacher_recommendations": [],
            "next_steps": []
        },
        "pedagogical_approach": {
            "teaching_style": "Multi-sensory",
            "differentiation_suggestions": []
        }
    }
    
    # Process equations from Azure result
    if 'equations' in azure_result:
        template['assignment_info']['total_problems'] = len(azure_result['equations'])
        
        for i, eq in enumerate(azure_result['equations']):
            is_correct = eq.get('is_correct', False)
            if is_correct:
                template['assignment_info']['total_correct'] += 1
                
            # Create problem entry
            problem = {
                "id": i + 1,
                "text": eq['text'],
                "is_correct": is_correct,
                "parsed": {}
            }
            
            # Add error details if not correct
            if not is_correct:
                problem["error_details"] = {
                    "type": eq.get('error_type', 'unknown_error'),
                    "description": eq.get('explanation', 'Error in calculation'),
                    "common_mistake": eq.get('common_mistake', 'calculation_error')
                }
                
                # Add feedback information
                problem["feedback"] = {
                    "concept_explanation": "This concept involves proper application of math rules.",
                    "learning_strategy": "Review the steps of the calculation carefully.",
                    "socratic_questions": [
                        f"What did you notice about {eq['text']}?",
                        "How could you verify your answer?",
                        "What rule applies in this situation?"
                    ],
                    "direct_instruction": [
                        f"The correct answer should be {eq.get('correct_form', 'different')}.",
                        "Remember to follow the order of operations."
                    ],
                    "visual_aids": {
                        "number_line": "Use a number line to visualize the operation.",
                        "manipulatives": "Count using manipulatives to understand the concept."
                    }
                }
            
            template['problems'].append(problem)
    
    # Calculate score percentage
    if template['assignment_info']['total_problems'] > 0:
        template['assignment_info']['score_percentage'] = round(
            (template['assignment_info']['total_correct'] / 
             template['assignment_info']['total_problems']) * 100
        )
    
    # Generate summary based on problems
    if template['problems']:
        # Add strengths
        if template['assignment_info']['total_correct'] > 0:
            template['summary']['strengths'].append(
                "Student correctly solved some math problems."
            )
            
        # Add areas for improvement
        if template['assignment_info']['total_correct'] < template['assignment_info']['total_problems']:
            error_types = set()
            for problem in template['problems']:
                if not problem['is_correct'] and 'error_details' in problem:
                    error_types.add(problem['error_details']['type'])
                    
            for error_type in error_types:
                template['summary']['areas_for_improvement'].append(
                    f"Improve understanding of concepts related to {error_type.replace('_', ' ')}."
                )
                
        # Add recommendations
        template['summary']['teacher_recommendations'] = [
            "Practice with additional examples",
            "Review fundamental concepts",
            "Use visual aids to enhance understanding"
        ]
        
        # Add next steps
        template['summary']['next_steps'] = [
            "Complete practice exercises focusing on problem areas",
            "Review correct solutions and understand why they work",
            "Apply concepts to real-world scenarios"
        ]
        
        # Add differentiation suggestions
        template['pedagogical_approach']['differentiation_suggestions'] = [
            "Use concrete manipulatives for visual learners",
            "Provide step-by-step written instructions for sequential learners",
            "Incorporate math games for kinesthetic learners"
        ]
    
    return template

def get_demo_result():
    """Get a demo result from the template file"""
    try:
        template_path = os.path.join('results', 'advanced', 'math_homework_grading_template.json')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return json.load(f)
        else:
            # Create a simple demo result
            return {
                "student_info": {
                    "name": "Demo Student",
                    "grade_level": "Elementary"
                },
                "assignment_info": {
                    "title": "Demo Math Homework",
                    "date": "2025-05-04",
                    "total_problems": 2,
                    "total_correct": 1,
                    "score_percentage": 50
                },
                "problems": [
                    {
                        "id": 1,
                        "text": "2 + 2 = 4",
                        "is_correct": True
                    },
                    {
                        "id": 2,
                        "text": "3 + 5 = 7",
                        "is_correct": False,
                        "error_details": {
                            "type": "arithmetic_error",
                            "description": "The sum of 3 and 5 should be 8, not 7.",
                            "common_mistake": "off_by_one_low"
                        },
                        "feedback": {
                            "concept_explanation": "Addition means combining quantities to find their sum.",
                            "learning_strategy": "Count all objects carefully, one by one.",
                            "socratic_questions": [
                                "How many objects would you have if you combined 3 and 5?",
                                "Can you show me how you counted?"
                            ],
                            "direct_instruction": [
                                "The correct answer is 3 + 5 = 8.",
                                "When adding, count all objects in both groups."
                            ],
                            "visual_aids": {
                                "number_line": "Use a number line to count forward.",
                                "counters": "Use counters of different colors to represent each number."
                            }
                        }
                    }
                ],
                "summary": {
                    "strengths": ["Good understanding of some addition problems."],
                    "areas_for_improvement": ["Need more practice with basic addition."],
                    "teacher_recommendations": [
                        "Practice counting objects",
                        "Use visual aids like number lines"
                    ],
                    "next_steps": [
                        "Complete additional practice problems",
                        "Use manipulatives to build understanding"
                    ]
                },
                "pedagogical_approach": {
                    "teaching_style": "Multi-sensory",
                    "differentiation_suggestions": [
                        "Use manipulatives for hands-on learning",
                        "Practice through games"
                    ]
                }
            }
    except Exception as e:
        logger.exception("Error getting demo result")
        return {"error": str(e)}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
