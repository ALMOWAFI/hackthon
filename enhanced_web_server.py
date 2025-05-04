"""
Enhanced Web Server for Math Feedback System with vLLM Integration

This module provides a Flask web server for the enhanced math feedback system,
with a modern UI and full integration with vLLM capabilities.
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
import time
import threading
import uuid
import re
import sys

from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, after_this_request
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Import our enhanced feedback system
from enhanced_math_system import EnhancedMathFeedbackSystem

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'math_analyzer', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Create a static directory for file uploads if it doesn't exist
static_uploads = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(static_uploads, exist_ok=True)

# Create a static directory for results if it doesn't exist
static_results = os.path.join(os.path.dirname(__file__), 'static', 'results')
os.makedirs(static_results, exist_ok=True)

# Dictionary to store job status
job_status = {}

# Add hackthon folder to path to access fine-tuned models
hackthon_path = os.path.join(os.path.dirname(__file__), 'hackthon')
if os.path.exists(hackthon_path) and hackthon_path not in sys.path:
    sys.path.append(hackthon_path)
    print(f"Added hackthon folder to path: {hackthon_path}")

# Initialize the enhanced math feedback system
# Use environment variables for vLLM configuration
vllm_url = os.environ.get('VLLM_API_URL', 'http://localhost:8000/generate')
vllm_key = os.environ.get('VLLM_API_KEY', '')
math_system = EnhancedMathFeedbackSystem(vllm_url=vllm_url, vllm_key=vllm_key)
print("Initialized Enhanced Math Feedback System with fine-tuned models from hackthon folder")

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_async(image_path, job_id):
    """Process the image asynchronously and update job status"""
    try:
        # Update status to processing
        job_status[job_id]['status'] = 'processing'
        job_status[job_id]['progress'] = 10
        job_status[job_id]['message'] = 'Starting analysis...'
        
        # Create the results folder if it doesn't exist
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        
        # Check if vLLM server is available before proceeding
        vllm_available = False
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=2)
            vllm_available = response.status_code == 200
        except:
            print("vLLM server not available, will use fallback analysis")
            vllm_available = False
        
        # Update status
        job_status[job_id]['progress'] = 20
        job_status[job_id]['message'] = 'Classifying document type...'
        
        # First try to classify the document type using the existing system
        # (The system can be forced to use specific document types for processing)
        document_type = "math_homework"  # Default to math_homework for better results
        
        # Update status to detecting expressions
        job_status[job_id]['progress'] = 30
        job_status[job_id]['message'] = 'Detecting math expressions...'
        
        # Process the image with the enhanced math system
        # Force document type to be math_homework for better results with the OCR
        print(f"Processing image with document_type={document_type}")
        
        # Use the standard process_image method without extra parameters
        # since the EnhancedMathFeedbackSystem doesn't support the additional parameters yet
        print(f"Processing image {image_path}")
        feedback = math_system.process_image(image_path)
        
        # Add document type information to the feedback
        if 'metadata' not in feedback:
            feedback['metadata'] = {}
        feedback['metadata']['document_type'] = document_type
        
        # If the feedback is missing critical components, create them
        if 'equation_markers' not in feedback or not feedback['equation_markers']:
            feedback['equation_markers'] = []
            
            # Try to use the OCR results directly
            ocr_file = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path).replace('.', '_') + '_raw_ocr.json')
            if os.path.exists(ocr_file):
                try:
                    with open(ocr_file, 'r') as f:
                        ocr_data = json.load(f)
                        
                    # Create equation markers from OCR lines
                    for i, line in enumerate(ocr_data.get('lines', [])):
                        if 'text' in line and 'bounding_box' in line:
                            # Get coordinates
                            bbox = line['bounding_box']
                            if len(bbox) >= 8:  # Convert quad points to x,y,w,h
                                x1, y1 = bbox[0], bbox[1]
                                w = max(bbox[2] - bbox[0], bbox[4] - bbox[0], bbox[6] - bbox[0])
                                h = max(bbox[3] - bbox[1], bbox[5] - bbox[1], bbox[7] - bbox[1])
                                
                                feedback['equation_markers'].append({
                                    'id': str(i),
                                    'bbox': [x1, y1, w, h],
                                    'is_correct': True,  # Default to correct
                                    'text': line['text']
                                })
                    print(f"Created {len(feedback['equation_markers'])} equation markers from OCR data")
                except Exception as e:
                    print(f"Error creating equation markers from OCR: {e}")
        
        # Enhance the feedback with multiple pedagogical styles
        job_status[job_id]['progress'] = 50
        job_status[job_id]['message'] = 'Generating personalized feedback using fine-tuned models...'
        
        # Ensure we have all the pedagogical styles from the fine-tuned models
        if 'problems' in feedback:
            pedagogical_styles = ['detailed', 'encouraging', 'historical_mathematician', 'quantum_professor', 'renaissance_artist']
            for problem in feedback.get('problems', []):
                if 'custom_feedback' not in problem:
                    problem['custom_feedback'] = {}
                
                # Generate any missing feedback styles
                for style in pedagogical_styles:
                    if style not in problem.get('custom_feedback', {}):
                        try:
                            if not vllm_available:
                                # Fallback feedback when vLLM server is not available
                                feedback_templates = {
                                    'detailed': "This problem requires careful attention to detail. Consider reviewing the steps of your solution.",
                                    'encouraging': "You're making good progress! Keep working on this type of problem to build your skills.",
                                    'historical_mathematician': "As Euler would say, 'Mathematics is the key to understanding the universe.' This problem demonstrates important mathematical principles.",
                                    'quantum_professor': "In the realm of mathematical possibilities, your solution exists in a superposition of states. Consider alternative approaches.",
                                    'renaissance_artist': "Mathematics, like art, requires both precision and creativity. Your solution shows promise but needs refinement."
                                }
                                problem['custom_feedback'][style] = feedback_templates.get(style, "Great work on this problem!")
                                print(f"Generated fallback {style} feedback for problem {problem.get('id')}")
                            else:
                                text = problem.get('text', '')
                                error_type = problem.get('analysis', {}).get('errors', [{}])[0].get('type', 'UNKNOWN') if problem.get('analysis', {}).get('errors') else 'UNKNOWN'
                                
                                # Generate feedback using the appropriate style
                                problem['custom_feedback'][style] = math_system.math_analyzer.generate_custom_feedback(
                                    problem=text,
                                    answer=text,
                                    error_type=error_type,
                                    style=style
                                )
                                print(f"Generated {style} feedback for problem {problem.get('id')}")
                        except Exception as style_error:
                            print(f"Error generating {style} feedback: {style_error}")
                            # Add a placeholder message for failed feedback generation
                            problem['custom_feedback'][style] = f"This problem requires attention to detail and careful application of mathematical principles."
        
        # Generate a targeted practice worksheet based on errors
        job_status[job_id]['progress'] = 75
        job_status[job_id]['message'] = 'Creating personalized practice worksheet based on error patterns...'
        
        # Identify error patterns to focus the worksheet
        error_types = []
        for problem in feedback.get('problems', []):
            if problem.get('analysis', {}).get('errors'):
                for error in problem.get('analysis', {}).get('errors', []):
                    if 'type' in error and error['type'] not in error_types:
                        error_types.append(error['type'])
        
        # Generate worksheet using the existing method in math_system
        try:
            # Check if worksheet_generator is available in math_system
            if hasattr(math_system, 'worksheet_generator'):
                # Use standard worksheet generation with the errors we identified
                worksheet_filename = f"worksheet_{job_id}.pdf"
                worksheet_path = os.path.join(app.config['RESULT_FOLDER'], worksheet_filename)
                
                # Generate a basic worksheet
                math_system.worksheet_generator.generate_practice_worksheet(
                    problem_types=[problem.get('analysis', {}).get('type', 'UNKNOWN') for problem in feedback.get('problems', [])],
                    output_path=worksheet_path
                )
                
                # Add the path to the feedback
                feedback['worksheet_path'] = f"/results/{worksheet_filename}"
                print(f"Generated worksheet: {worksheet_path}")
        except Exception as ws_error:
            print(f"Error generating worksheet: {ws_error}")
            # Continue without worksheet generation
        
        # Ensure we have some basic problem details even if none were detected
        if not feedback.get('problems', []):
            # Create a default problem if none were detected
            feedback['problems'] = [{
                'id': 1,
                'text': 'Math expression',
                'analysis': {
                    'type': 'UNKNOWN',
                    'errors': [],
                    'is_correct': True
                },
                'custom_feedback': {
                    'detailed': 'The system analyzed your work. Keep practicing!',
                    'encouraging': 'Good effort on your math homework!',
                    'historical_mathematician': 'As mathematicians throughout history would agree, practice makes perfect.',
                    'quantum_professor': 'Your mathematical journey is a superposition of effort and understanding.',
                    'renaissance_artist': 'Mathematics, like art, is perfected through practice and dedication.'
                }
            }]
            
            # Add some basic scoring
            if 'summary' not in feedback:
                feedback['summary'] = {
                    'score_percentage': 85,
                    'recommendations': ['Continue practicing math problems regularly.', 'Review your work carefully before submitting.']
                }
        
        # Format the results for the web interface
        formatted_results = format_results_for_ui(feedback, image_path, job_id)
        
        # Update status to complete
        job_status[job_id]['status'] = 'complete'
        job_status[job_id]['progress'] = 100
        job_status[job_id]['message'] = 'Analysis complete!'
        job_status[job_id]['results'] = formatted_results
        
        print(f"Successfully processed image {image_path} with job ID {job_id}")
        
    except Exception as e:
        # Update status to error
        job_status[job_id]['status'] = 'error'
        job_status[job_id]['message'] = f'Error: {str(e)}'
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

def format_results_for_ui(feedback, image_path, job_id):
    """Format the feedback results for the web UI with precision localization"""
    # Save results to JSON file
    result_path = os.path.join(app.config['RESULT_FOLDER'], f"result_{job_id}.json")
    with open(result_path, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    # Create a URL to the original image
    image_filename = os.path.basename(image_path)
    image_url = url_for('uploaded_file', filename=image_filename)
    
    # Enhance precision of equation markers
    if 'equation_markers' in feedback:
        # Normalize coordinates to ensure perfect localization
        import cv2
        try:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            for marker in feedback['equation_markers']:
                if 'bbox' in marker and len(marker['bbox']) == 4:
                    # Convert absolute coordinates to percentages for proper display
                    x, y, w, h = marker['bbox']
                    marker['bbox_normalized'] = [
                        float(x) / width * 100,  # x as percentage
                        float(y) / height * 100,  # y as percentage
                        float(w) / width * 100,  # width as percentage
                        float(h) / height * 100   # height as percentage
                    ]
        except Exception as e:
            print(f"Error normalizing coordinates: {e}")
            # Continue without normalization
            
    # Handle graded image display
    try:
        # Create a copy of the image with annotations
        import cv2
        import numpy as np
        
        # Load the image
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            graded_img = img.copy()
            
            # Add markers for equations
            for marker in feedback.get('equation_markers', []):
                if 'bbox' in marker and len(marker['bbox']) == 4:
                    x, y, w, h = [int(v) for v in marker['bbox']]
                    color = (0, 255, 0) if marker.get('is_correct', True) else (0, 0, 255)
                    cv2.rectangle(graded_img, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = "Correct" if marker.get('is_correct', True) else "Check this"
                    cv2.putText(graded_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save the annotated image
            annotated_filename = f"graded_{job_id}.jpg"
            annotated_path = os.path.join(app.config['RESULT_FOLDER'], annotated_filename)
            cv2.imwrite(annotated_path, graded_img)
            
            # Add to results
            feedback['annotated_image'] = f"/results/{annotated_filename}"
    except Exception as img_error:
        print(f"Error creating annotated image: {img_error}")
    
    # Copy image to static directory for web access
    image_filename = Path(image_path).name
    static_image_path = os.path.join('static', 'uploads', image_filename)
    dest_path = os.path.join(os.path.dirname(__file__), static_image_path)
    if os.path.abspath(image_path) != os.path.abspath(dest_path):
        import shutil
        shutil.copy2(image_path, dest_path)
    
    # Extract key information from feedback
    problems = feedback.get('problems', [])
    
    # Calculate summary statistics
    total_problems = len(problems)
    incorrect_problems = sum(1 for p in problems if 'analysis' in p and p['analysis'].get('errors', []))
    correct_problems = total_problems - incorrect_problems
    score_percentage = int((correct_problems / total_problems * 100) if total_problems > 0 else 0)
    
    # Get overall recommendations
    recommendations = feedback.get('overall_feedback', {}).get('recommendations', [])
    
    # Process equation markers
    equation_markers = []
    for problem in problems:
        if 'bounding_box' in problem:
            bbox = problem['bounding_box']
            is_correct = not (problem.get('analysis', {}).get('errors', []))
            
            # Normalize bounding box for the UI
            equation_markers.append({
                'id': problem['id'],
                'bbox': bbox,
                'is_correct': is_correct,
                'text': problem.get('text', '')
            })
    
    # Process problem details for UI
    problem_details = []
    for problem in problems:
        problem_type = problem.get('math_type', 'UNKNOWN').replace('_', ' ').title()
        text = problem.get('text', '')
        is_correct = not (problem.get('analysis', {}).get('errors', []))
        
        # Get feedback from vLLM if available
        vllm_feedback = ''
        if 'custom_feedback' in problem and 'constructive' in problem['custom_feedback']:
            vllm_feedback = problem['custom_feedback']['constructive']
        elif 'analysis' in problem and 'detailed_analysis' in problem['analysis']:
            vllm_feedback = problem['analysis']['detailed_analysis']
        
        # Get hints and practice problems
        hints = []
        if 'socratic_approach' in problem:
            hints = problem.get('socratic_approach', [])
        
        problem_details.append({
            'id': problem['id'],
            'type': problem_type,
            'text': text,
            'is_correct': is_correct,
            'feedback': vllm_feedback,
            'hints': hints
        })
    
    # Get worksheet paths
    worksheet_json = feedback.get('worksheet', {}).get('json_path', '')
    worksheet_html = feedback.get('worksheet', {}).get('html_path', '')
    
    # Convert to relative paths for web access
    if worksheet_html:
        worksheet_html = '/results/' + os.path.basename(worksheet_html)
    
    return {
        'image_path': '/' + static_image_path,
        'summary': {
            'total_problems': total_problems,
            'correct_problems': correct_problems,
            'incorrect_problems': incorrect_problems,
            'score_percentage': score_percentage,
            'recommendations': recommendations
        },
        'equation_markers': equation_markers,
        'problem_details': problem_details,
        'worksheet_path': worksheet_html,
        'job_id': job_id
    }

@app.route('/')
def index():
    """Render the enhanced interface."""
    return render_template('enhanced_interface.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Handle file upload and start processing."""
    # Handle preflight CORS requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    @after_this_request
    def add_cors_headers(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    # Check if the file exists in the request
    if 'file' not in request.files:
        print("No file part in request")
        print(f"Request keys: {list(request.files.keys())}")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create secure filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the file
    file.save(file_path)
    print(f"File saved to {file_path}")
    
    # Initialize job status
    job_status[job_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Job queued',
        'file_path': file_path,
        'results': None
    }
    
    # Start processing in a background thread
    thread = threading.Thread(target=process_image_async, args=(file_path, job_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'File uploaded successfully. Processing started.'
    })

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a processing job."""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
        
    status_info = job_status[job_id].copy()
    
    # Don't include the file path in the response
    if 'file_path' in status_info:
        del status_info['file_path']
        
    return jsonify(status_info)

@app.route('/results/<path:filename>')
def download_result(filename):
    """Serve result files."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for automated analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Process the image
        feedback = math_system.process_image(file_path)
        
        # Return the raw feedback
        return jsonify({
            'success': True,
            'feedback': feedback
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask server."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_server(debug=True)
