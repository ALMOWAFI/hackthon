import os
import cv2
import numpy as np
import requests
import json
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from math_analyzer.improved_error_localization import MathErrorDetector

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# vLLM API URL (use environment variable or default to localhost)
VLLM_API_URL = os.environ.get('VLLM_API_URL', 'http://localhost:8000')
AVAILABLE_TEACHING_STYLES = [
    'detailed', 
    'encouraging', 
    'historical_mathematician',
    'quantum_professor', 
    'renaissance_artist'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', teaching_styles=AVAILABLE_TEACHING_STYLES)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/teaching-styles')
def teaching_styles():
    """Return the available teaching styles based on vLLM models"""
    try:
        response = requests.get(f"{VLLM_API_URL}/models")
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Filter for teaching style models
            teaching_styles = [model for model in models 
                              if model in AVAILABLE_TEACHING_STYLES]
            return jsonify({'styles': teaching_styles})
        return jsonify({'styles': AVAILABLE_TEACHING_STYLES})
    except Exception as e:
        print(f"Error fetching models: {e}")
        return jsonify({'styles': AVAILABLE_TEACHING_STYLES})

@app.route('/detect', methods=['POST'])
def detect_errors():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'})
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Get student work and correct solution from form
    student_work = request.form.get('student_work', '')
    correct_solution = request.form.get('correct_solution', '')
    teaching_style = request.form.get('teaching_style', 'detailed')
    
    # Read image
    image = cv2.imread(file_path)
    if image is None:
        return jsonify({'error': 'Could not read image'})
    
    # Process the image
    try:
        # Create error detector
        detector = MathErrorDetector()
        
        # Detect errors
        result = detector.detect_errors(student_work, correct_solution, image)
        
        # Save the marked image
        result_filename = f"{os.path.splitext(filename)[0]}_marked.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        if result.marked_image is not None:
            cv2.imwrite(result_path, result.marked_image)
        
        # Format errors for display
        errors = []
        for error in result.errors:
            errors.append({
                'line_number': error.line_number,
                'error_text': error.error_text,
                'error_type': error.error_type,
                'correction': error.correction,
                'explanation': error.explanation,
                'position': {
                    'top_left_x': error.top_left_x,
                    'top_left_y': error.top_left_y,
                    'bottom_right_x': error.bottom_right_x,
                    'bottom_right_y': error.bottom_right_y
                }
            })
        
        # Generate vLLM feedback with selected teaching style
        expert_feedback = generate_vllm_feedback(student_work, result.errors, teaching_style)
        
        # Generate practice sheet based on errors
        practice_sheet = generate_practice_sheet(result.errors)
        
        return jsonify({
            'success': True,
            'original_image': f"/uploads/{filename}",
            'marked_image': f"/results/{result_filename}",
            'errors': errors,
            'error_count': len(errors),
            'expert_feedback': expert_feedback,
            'practice_sheet': practice_sheet,
            'teaching_style': teaching_style
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing image: {str(e)}'})

def generate_vllm_feedback(student_work, errors, style="detailed"):
    """Generate pedagogical feedback based on detected errors using vLLM"""
    if not errors:
        return "Great job! All your answers are correct."
    
    # Create prompt for vLLM
    error_descriptions = []
    for i, error in enumerate(errors):
        error_descriptions.append(
            f"Error {i+1}: {error.error_type} in '{error.error_text}'. "
            f"Correct form: {error.correction}. "
            f"Explanation: {error.explanation}"
        )
    
    error_text = "\n".join(error_descriptions)
    
    prompt = f"""
As a {style} math teacher, provide helpful feedback for a student who made the following errors:

Student's work:
{student_work}

Detected errors:
{error_text}

Please provide tailored feedback to help the student understand their mistakes and improve.
"""
    
    try:
        # Call vLLM API
        response = requests.post(
            f"{VLLM_API_URL}/generate",
            json={
                "prompt": prompt,
                "model": style,  # Use teaching style as model name
                "max_tokens": 512,
                "temperature": 0.2
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("text", "").strip()
        else:
            print(f"vLLM API error: {response.status_code} - {response.text}")
            # Fallback to basic feedback
            return generate_basic_feedback(student_work, errors)
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        # Fallback to basic feedback
        return generate_basic_feedback(student_work, errors)

def generate_basic_feedback(student_work, errors):
    """Fallback feedback generator when vLLM is not available"""
    if not errors:
        return "Great job! All your answers are correct."
    
    lines = student_work.strip().split('\n')
    
    # Group errors by line
    errors_by_line = {}
    for error in errors:
        line_idx = error.line_number - 1  # Convert to 0-indexed
        if 0 <= line_idx < len(lines):
            if line_idx not in errors_by_line:
                errors_by_line[line_idx] = []
            errors_by_line[line_idx].append(error)
    
    feedback = ["I noticed some issues in your work:"]
    
    # Generate feedback for each line with errors
    for line_idx, line_errors in sorted(errors_by_line.items()):
        line = lines[line_idx]
        feedback.append(f"\nProblem {line_idx + 1}: {line}")
        
        for error in line_errors:
            feedback.append(f"• {error.explanation} The correct form is {error.correction}.")
                
    # Add overall recommendation
    feedback.append("\nRemember to carefully check your calculations and apply mathematical rules correctly.")
    
    return "\n".join(feedback)

def generate_practice_sheet(errors):
    """Generate practice problems based on detected error types"""
    if not errors:
        return "No errors detected. Keep up the good work!"
    
    # Get unique error types
    error_types = set()
    for error in errors:
        error_types.add(error.error_type.lower())
    
    practice_problems = ["# Practice Sheet\n## Based on your specific error patterns\n"]
    
    if "arithmetic" in error_types:
        practice_problems.append("""
### Arithmetic Practice
1. Evaluate: 3/4 + 2/3
2. Simplify: (7 × 9) ÷ 3 - 4^2
3. Calculate: 12.5 × 0.8
""")
    
    if "sign" in error_types:
        practice_problems.append("""
### Sign Handling Practice
1. Solve: 3x - 7 = -10
2. Simplify: -2(3x - 4) + 5
3. Solve: -5x > 15
""")
    
    if "exponent" in error_types:
        practice_problems.append("""
### Exponent Rules Practice
1. Simplify: (x^2)^3 × x^4
2. Expand: (2a)^3
3. Solve: 2^x = 32
""")
        
    if "distribution" in error_types:
        practice_problems.append("""
### Distribution Practice
1. Expand: 3(x + 2y - 5)
2. Expand and simplify: (x + 5)(x - 2)
3. Factor: 3x^2 - 12
""")
    
    if "factoring" in error_types:
        practice_problems.append("""
### Factoring Practice
1. Factor completely: x^2 - 9
2. Factor: 6x^2 + 13x - 5
3. Solve by factoring: x^2 - 7x + 12 = 0
""")
    
    # Create a general practice section for any other error types
    practice_problems.append("""
### General Review Practice
1. Solve the equation: 2x/3 - 5 = x/6 + 2
2. Simplify the expression: (3x^2 - 5x + 2) - (x^2 - 3x - 7)
3. Find the domain of f(x) = √(x + 3)
""")
    
    return "\n".join(practice_problems)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'ok'})

@app.route('/api/models')
def get_models():
    """Get available models from vLLM server"""
    try:
        response = requests.get(f"{VLLM_API_URL}/models")
        if response.status_code == 200:
            return response.json()
        return jsonify({'error': f"Error fetching models: {response.status_code}"})
    except Exception as e:
        return jsonify({'error': f"Error connecting to vLLM server: {str(e)}"})

if __name__ == '__main__':
    # Check vLLM server availability
    try:
        response = requests.get(f"{VLLM_API_URL}/health", timeout=2)
        if response.status_code == 200:
            print(f"Successfully connected to vLLM server at {VLLM_API_URL}")
        else:
            print(f"Warning: vLLM server health check failed with status {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not connect to vLLM server: {e}")
        print("Application will use fallback feedback generation.")
        
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
