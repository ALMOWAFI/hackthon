import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from math_analyzer.improved_error_localization import MathErrorDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

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
        
        # Generate feedback
        socratic_feedback = generate_feedback(student_work, result.errors, "socratic")
        direct_feedback = generate_feedback(student_work, result.errors, "direct")
        
        return jsonify({
            'success': True,
            'original_image': f"/uploads/{filename}",
            'marked_image': f"/results/{result_filename}",
            'errors': errors,
            'error_count': len(errors),
            'socratic_feedback': socratic_feedback,
            'direct_feedback': direct_feedback
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing image: {str(e)}'})

def generate_feedback(student_work, errors, style="direct"):
    """Generate pedagogical feedback based on detected errors"""
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
    
    feedback = []
    
    # Generate feedback for each line with errors
    for line_idx, line_errors in sorted(errors_by_line.items()):
        line = lines[line_idx]
        
        if style == "socratic":
            # Socratic style uses questions to guide the student
            feedback.append(f"Question {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• I see you wrote {error.error_text}. Can you double-check your calculation?")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• What happens to the sign when you move a term across the equals sign?")
                elif "exponent" in error.error_type.lower():
                    feedback.append(f"• Look at how you're handling the exponents in {error.error_text}. What's the rule for exponents when multiplying with the same base?")
                elif "distribution" in error.error_type.lower():
                    feedback.append(f"• When distributing in {error.error_text}, what do you need to do to each term inside the parentheses?")
                elif "factoring" in error.error_type.lower():
                    feedback.append(f"• What factors should you look for when factoring {error.error_text}?")
                else:
                    feedback.append(f"• Take another look at {error.error_text}. What might be wrong here?")
                
            feedback.append("What approach might work better here?")
            
        else:  # Direct instruction style
            # Direct style provides explicit corrections
            feedback.append(f"Problem {line_idx + 1}: {line}")
            
            for error in line_errors:
                if "arithmetic" in error.error_type.lower():
                    feedback.append(f"• There's a calculation error with {error.error_text}. The correct value is {error.correction}.")
                elif "sign" in error.error_type.lower():
                    feedback.append(f"• When moving terms across the equals sign, you need to change the sign. {error.error_text} should be {error.correction}.")
                elif "exponent" in error.error_type.lower():
                    feedback.append(f"• {error.explanation} The correct expression is {error.correction}.")
                elif "distribution" in error.error_type.lower():
                    feedback.append(f"• {error.explanation} The correct distribution is {error.correction}.")
                elif "factoring" in error.error_type.lower():
                    feedback.append(f"• The factoring in {error.error_text} is incorrect. {error.explanation} The correct factorization is {error.correction}.")
                else:
                    feedback.append(f"• {error.explanation} The correct form is {error.correction}.")
                    
        feedback.append("")  # Add a blank line between problems
    
    # Add overall recommendation
    if style == "socratic":
        feedback.append("What patterns do you notice in these errors? How might you avoid similar mistakes in the future?")
    else:
        feedback.append("Remember to carefully check your calculations and apply mathematical rules correctly.")
    
    return "\n".join(feedback)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
