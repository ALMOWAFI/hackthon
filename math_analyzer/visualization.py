import cv2
import numpy as np
from .config import VISUALIZATION

class Visualizer:
    def __init__(self):
        self.colors = VISUALIZATION['colors']
        self.font_scale = VISUALIZATION['font_scale']
        self.thickness = VISUALIZATION['thickness']
        
    def draw_circle(self, image, center, radius, color):
        """Draw a circle on the image."""
        cv2.circle(image, center, radius, color, self.thickness)
        
    def draw_checkmark(self, image, position, size=20):
        """Draw a checkmark on the image."""
        x, y = position
        points = np.array([
            [x - size, y],
            [x - size//2, y + size],
            [x + size, y - size//2]
        ], np.int32)
        cv2.polylines(image, [points], False, self.colors['correct'], self.thickness)
        
    def draw_text(self, image, text, position, color=None):
        """Draw text on the image."""
        if color is None:
            color = self.colors['text']
            
        cv2.putText(image, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   color, self.thickness)
                   
    def annotate_region(self, region, analysis, question_number=0):
        """
        Add annotations to a question region based on analysis.
        
        Args:
            region (numpy.ndarray): Image region
            analysis (dict): Analysis results
            
        Returns:
            numpy.ndarray: Annotated image
        """
        annotated = region.copy()
        
        # Get image dimensions
        h, w = annotated.shape[:2]
        
        # Get information from analysis
        score = analysis.get('score', 0)
        student_answer = analysis.get('student_answer', '')
        correct_answer = analysis.get('correct_answer', '')
        question_text = analysis.get('question', '')
        
        # Mark specific mistakes directly on the equation
        # Look for the equals sign to determine where student answer is
        if '=' in question_text:
            # Try to find position of equals sign in the image
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            equals_template = np.zeros((30, 30), dtype=np.uint8)
            cv2.line(equals_template, (5, 15), (25, 15), 255, 2)
            cv2.line(equals_template, (5, 20), (25, 20), 255, 2)
            
            # Use template matching to find equals sign
            result = cv2.matchTemplate(gray, equals_template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Mark answer position (right of equals sign)
            if max_loc[0] > 0:
                # Draw circle around the answer part
                answer_x = max_loc[0] + 30  # Right of equals sign
                answer_y = max_loc[1]
                if analysis.get('errors', []):
                    # Mark incorrect answer with red circle
                    cv2.circle(annotated, (answer_x, answer_y), 30, self.colors['incorrect'], 2)
                else:
                    # Mark correct answer with green circle
                    cv2.circle(annotated, (answer_x, answer_y), 30, self.colors['correct'], 2)
        
        # Draw score in top corner
        score_text = f"Score: {score}%"
        self.draw_text(annotated, score_text, (10, 30), 
                       self.colors['correct'] if score > 50 else self.colors['incorrect'])
        
        # Show correct answer below for incorrect problems
        if analysis.get('errors', []):
            if correct_answer:
                correct_text = f"Correct: {correct_answer}"
                self.draw_text(annotated, correct_text, (10, h-20), self.colors['correct'])
            
        return annotated
        
    def create_final_image(self, results):
        """
        Create a final annotated image with all questions and overall results.
        
        Args:
            results (dict): Complete analysis results
            
        Returns:
            numpy.ndarray: Final annotated image
        """
        # Create a background image
        h = 800
        w = 600
        final_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw title with better formatting
        self.draw_text(final_img, "Math Homework Analysis", (w//2 - 150, 50), 
                       color=(0, 0, 128))  # Dark blue title
        
        # Draw overall score with better color coding
        final_score = results.get('final_score', 0)
        score_text = f"Overall Score: {final_score:.1f}%"
        score_y = 100
        score_color = (0, 180, 0) if final_score >= 60 else \
                     (180, 180, 0) if final_score >= 40 else \
                     (180, 0, 0)
        self.draw_text(final_img, score_text, (w//2 - 100, score_y), color=score_color)
        
        # Detailed feedback section
        self.draw_text(final_img, "Detailed Feedback:", (50, 150), color=(100, 50, 0))
        
        # Draw detailed summary for each question
        y_offset = 200
        for question in results.get('questions', []):
            q_num = question.get('question_number', 0)
            q_score = question.get('analysis', {}).get('score', 0)
            q_text = question.get('text', '').replace('\n', ' ').strip()
            
            # Highlight the question number
            self.draw_text(final_img, f"Question {q_num}:", (50, y_offset), 
                           color=(50, 50, 100))
            
            # Show the equation that was detected
            if q_text:
                self.draw_text(final_img, f"Detected: '{q_text}'", (150, y_offset), 
                               color=(0, 0, 0))
            
            y_offset += 40
            
            # Show the score
            score_color = (0, 150, 0) if q_score >= 60 else \
                          (150, 150, 0) if q_score >= 40 else \
                          (150, 0, 0)
            self.draw_text(final_img, f"Score: {q_score}%", (70, y_offset), 
                           color=score_color)
            
            # Display feedback
            y_offset += 40
            feedback = question.get('feedback', {}).get('feedback', '')
            if feedback:
                # Split feedback into multiple lines
                words = feedback.split()
                line = ''
                for word in words:
                    if len(line) + len(word) + 1 > 60:  # Limit line length
                        self.draw_text(final_img, line, (70, y_offset), color=(80, 40, 80))
                        y_offset += 30
                        line = word
                    else:
                        line += ' ' + word if line else word
                if line:
                    self.draw_text(final_img, line, (70, y_offset), color=(80, 40, 80))
                    y_offset += 30
            
            y_offset += 20  # Add space between questions
            
        return final_img 