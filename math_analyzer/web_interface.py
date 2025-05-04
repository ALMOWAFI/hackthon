import os
import json
import base64
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from math_analyzer import MathHomeworkAnalyzer
from .document_classifier import DocumentClassifier

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['TEMPLATE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'templates')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize components
analyzer = MathHomeworkAnalyzer()
document_classifier = DocumentClassifier()

@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document upload and process it."""
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf'}
    if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        try:
            # Process the document
            results = analyzer.analyze_homework(file_path)
            
            # Save results
            result_json_path = os.path.join(app.config['RESULT_FOLDER'], f"{os.path.splitext(filename)[0]}_results.json")
            with open(result_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Get result image paths
            marked_image_path = os.path.join(app.config['RESULT_FOLDER'], f"marked_{filename}")
            feedback_image_path = os.path.join(app.config['RESULT_FOLDER'], f"feedback_{filename}")
        except Exception as e:
            # If analysis fails, return error
            return jsonify({'error': f'Error analyzing document: {str(e)}'}), 500
        
        # Convert images to base64 for display
        def img_to_base64(img_path):
            if os.path.exists(img_path):
                with open(img_path, 'rb') as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            return None
            
        marked_img_b64 = img_to_base64(marked_image_path)
        feedback_img_b64 = img_to_base64(feedback_image_path)
        
        # Prepare response
        response = {
            'success': True,
            'results': results,
            'marked_image': marked_img_b64,
            'feedback_image': feedback_img_b64,
            'download_url': f"/results/{os.path.splitext(filename)[0]}_results.json"
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
def download_results(filename):
    """Download result files."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/privacy')
def privacy_policy():
    """Render the privacy policy page."""
    return render_template('privacy.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for automated document analysis."""
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Process the document
        results = analyzer.analyze_homework(file_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask server."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_server(debug=True)
