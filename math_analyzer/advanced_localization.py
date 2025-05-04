"""
Advanced localization module for precisely identifying and positioning math errors in handwritten work
Specifically designed to handle messy children's handwriting and provide accurate bounding boxes
"""

import cv2
import numpy as np
import math
import os
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy import ndimage

class AdvancedMathLocalization:
    """
    Advanced system for localizing and identifying handwritten math work, 
    with specific focus on messy children's handwriting
    """
    
    def __init__(self):
        """Initialize the advanced localization system"""
        # Constants for detection
        self.MIN_CONTOUR_AREA = 50  # Minimum area to consider as meaningful content
        self.LINE_DISTANCE_FACTOR = 1.5  # Factor for determining line spacing
        self.CHAR_DISTANCE_FACTOR = 0.8  # Factor for character spacing
        self.EQUATION_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for equation detection
        
        # Store detected components
        self.detected_lines = []
        self.detected_equations = []
        self.detected_symbols = []
        self.connected_components = []
        
        # Store processing results
        self.debug_images = {}
    
    def process_image(self, image_path, debug=False):
        """
        Process an image to detect and localize math content
        
        Args:
            image_path: Path to the image
            debug: Whether to save debug visualization images
            
        Returns:
            Structured data about detected math content including precise coordinates
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Store original image
        self.original_image = image.copy()
        
        # Preprocess image
        preprocessed = self._preprocess_image(image)
        
        # Detect content structure
        self._detect_content_structure(preprocessed)
        
        # Extract equations based on structure
        equations = self._extract_equations(preprocessed)
        
        # Analyze each equation
        equation_data = self._analyze_equations(equations, image)
        
        # Create debug visualizations if requested
        if debug:
            self._create_debug_visualizations(image_path)
        
        return {
            "equations": equation_data,
            "structure": {
                "line_count": len(self.detected_lines),
                "equation_count": len(self.detected_equations),
                "symbol_count": len(self.detected_symbols)
            }
        }
    
    def _preprocess_image(self, image):
        """
        Preprocess image for better contour detection
        Specifically tuned for handwritten content on paper
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying illumination
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate slightly to connect broken character parts
        dilation = cv2.dilate(opening, kernel, iterations=1)
        
        self.debug_images['preprocessed'] = dilation
        return dilation
    
    def _detect_content_structure(self, preprocessed):
        """
        Detect the structure of handwritten content: lines, equations, symbols
        Uses advanced techniques to handle messy writing and varying styles
        """
        # Find all contours
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter out noise (too small components)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]
        
        # Extract bounding boxes
        all_boxes = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            all_boxes.append((x, y, w, h, contour))
        
        # Group into horizontal lines using y-coordinate clustering
        y_centers = np.array([[y + h/2] for x, y, w, h, _ in all_boxes])
        
        # Use DBSCAN clustering to group by Y-coordinate (handles varying line heights)
        # Automatically determine eps based on image height
        eps = self.original_image.shape[0] * 0.03  # 3% of image height
        clustering = DBSCAN(eps=eps, min_samples=1).fit(y_centers)
        
        # Group boxes by line
        lines = defaultdict(list)
        for i, (x, y, w, h, contour) in enumerate(all_boxes):
            lines[clustering.labels_[i]].append((x, y, w, h, contour))
        
        # Sort lines by y-coordinate
        self.detected_lines = []
        for label in sorted(lines.keys()):
            line_boxes = lines[label]
            
            # Sort boxes within line by x-coordinate (left to right)
            line_boxes.sort(key=lambda box: box[0])
            
            # Calculate line metrics
            y_values = [box[1] for box in line_boxes]
            line_top = min(y_values)
            line_bottom = max([box[1] + box[3] for box in line_boxes])
            
            # Store line data
            self.detected_lines.append({
                "top": line_top,
                "bottom": line_bottom,
                "height": line_bottom - line_top,
                "boxes": line_boxes
            })
        
        # Merge components into connected character groups
        self._merge_connected_components(all_boxes)
        
        # Store contours for visualization
        self.debug_images['structure'] = self.original_image.copy()
        for line in self.detected_lines:
            for x, y, w, h, _ in line["boxes"]:
                cv2.rectangle(self.debug_images['structure'], (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    def _merge_connected_components(self, all_boxes):
        """
        Merge contours that likely belong to the same character or symbol
        Critical for accurate character segmentation in messy handwriting
        """
        self.connected_components = []
        
        # Extract just the bounding boxes
        boxes = [(x, y, w, h) for x, y, w, h, _ in all_boxes]
        
        # Group near components
        grouped = []
        remaining = set(range(len(boxes)))
        
        while remaining:
            # Get a seed box
            idx = next(iter(remaining))
            group = {idx}
            remaining.remove(idx)
            
            # Find connected components
            x1, y1, w1, h1 = boxes[idx]
            changed = True
            
            while changed:
                changed = False
                for i in list(remaining):
                    x2, y2, w2, h2 = boxes[i]
                    
                    # Check if boxes are close enough to be the same character
                    # Use smaller of the two heights as threshold distance
                    threshold = min(h1, h2) * self.CHAR_DISTANCE_FACTOR
                    
                    if (abs((x1 + w1/2) - (x2 + w2/2)) < threshold and 
                        abs((y1 + h1/2) - (y2 + h2/2)) < threshold):
                        group.add(i)
                        remaining.remove(i)
                        changed = True
            
            # Store the group
            grouped.append([all_boxes[i] for i in group])
        
        # Merge each group into a single component
        for group in grouped:
            if not group:
                continue
                
            # Calculate enclosing rect for the group
            contours = [cont for _, _, _, _, cont in group]
            all_points = np.vstack([c.reshape(-1, 2) for c in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            self.connected_components.append((x, y, w, h, all_points))
    
    def _extract_equations(self, preprocessed):
        """
        Extract equations from the detected lines
        Uses heuristics to handle equals signs and multi-part equations
        """
        self.detected_equations = []
        
        # Process each detected line
        for line_idx, line in enumerate(self.detected_lines):
            boxes = line["boxes"]
            
            # Skip lines with too few elements (likely not equations)
            if len(boxes) < 3:  # Need at least left side, equals sign, right side
                continue
            
            # Check for equals sign within the line
            equals_candidates = []
            for i, (x, y, w, h, _) in enumerate(boxes):
                # Check aspect ratio and size typical for equals sign
                aspect_ratio = w / max(h, 1)
                
                # Equals signs tend to have aspect ratio > 1 and be relatively small
                if 1.5 < aspect_ratio < 4.0:
                    # Extract the region of the potential equals sign
                    roi = preprocessed[y:y+h, x:x+w]
                    
                    # Check horizontal projection profile for two peaks
                    if self._check_equals_sign_profile(roi):
                        equals_candidates.append((i, x, y, w, h))
            
            # If no equals sign found, this may not be an equation
            if not equals_candidates:
                continue
            
            # Extract equations based on equals signs
            for eq_idx, (equals_idx, eq_x, eq_y, eq_w, eq_h) in enumerate(equals_candidates):
                # Get elements before equals sign (left side)
                left_side = boxes[:equals_idx]
                
                # Get elements after equals sign (right side)
                right_side = boxes[equals_idx+1:]
                
                # If we have both sides, consider it an equation
                if left_side and right_side:
                    # Calculate equation bounding box with enhanced precision
                    # Use padding to ensure the full equation is captured
                    left_x = max(0, min([x for x, _, _, _, _ in left_side]) - 5)
                    right_x = min(preprocessed.shape[1], max([x + w for x, _, w, _, _ in right_side]) + 5)
                    
                    # Get the vertical bounds with improved padding for superscripts and subscripts
                    all_components = left_side + [(eq_x, eq_y, eq_w, eq_h, 0)] + right_side
                    top_y = max(0, min([y for _, y, _, _, _ in all_components]) - 8)
                    bottom_y = min(preprocessed.shape[0], max([y + h for _, y, _, h, _ in all_components]) + 8)
                    
                    # Calculate the centroid of the equation to help with alignment
                    total_area = sum([w * h for _, _, w, h, _ in all_components])
                    centroid_x = sum([x * w * h for x, _, w, h, _ in all_components]) / total_area
                    centroid_y = sum([y * w * h for _, y, w, h, _ in all_components]) / total_area
                    
                    # Store equation data with enhanced bounding box
                    equation = {
                        "line_index": line_idx,
                        "bounding_box": (left_x, top_y, right_x - left_x, bottom_y - top_y),
                        "centroid": (centroid_x, centroid_y),
                        "equals_sign": (eq_x, eq_y, eq_w, eq_h),
                        "left_side": left_side,
                        "right_side": right_side
                    }
                    
                    self.detected_equations.append(equation)
        
        # Apply advanced post-processing to refine equation boxes
        self._refine_equation_bounding_boxes(preprocessed)
        
        return self.detected_equations
    
    def _refine_equation_bounding_boxes(self, preprocessed):
        """
        Apply advanced post-processing to refine and adjust equation bounding boxes
        Ensures boxes accurately enclose the full equation with appropriate padding
        """
        if not self.detected_equations:
            return
        
        refined_equations = []
        
        for equation in self.detected_equations:
            x, y, w, h = equation["bounding_box"]
            
            # Extract the region containing the equation
            roi = preprocessed[y:y+h, x:x+w]
            if roi.size == 0:
                refined_equations.append(equation)
                continue
                
            # Find contours within the ROI to refine the box
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                refined_equations.append(equation)
                continue
                
            # Create mask from contours
            mask = np.zeros_like(roi)
            for cnt in contours:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Remove small noise contours
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find refined bounds from mask
            non_zero_points = cv2.findNonZero(mask)
            if non_zero_points is None:
                refined_equations.append(equation)
                continue
                
            x_min, y_min, box_w, box_h = cv2.boundingRect(non_zero_points)
            
            # Add padding to ensure full coverage of the equation
            padding_x = int(box_w * 0.05) + 3  # 5% of width plus fixed padding
            padding_y = int(box_h * 0.1) + 5   # 10% of height plus fixed padding
            
            # Update the bounding box with refined coordinates
            refined_x = max(0, x + x_min - padding_x)
            refined_y = max(0, y + y_min - padding_y)
            refined_w = min(preprocessed.shape[1] - refined_x, box_w + 2 * padding_x)
            refined_h = min(preprocessed.shape[0] - refined_y, box_h + 2 * padding_y)
            
            # Ensure the refined box still contains the equals sign
            eq_x, eq_y, eq_w, eq_h = equation["equals_sign"]
            if (eq_x < refined_x or eq_x + eq_w > refined_x + refined_w or
                eq_y < refined_y or eq_y + eq_h > refined_y + refined_h):
                # If equals sign is outside, use the original box
                refined_equations.append(equation)
                continue
                
            # Update equation with refined bounding box
            refined_equation = equation.copy()
            refined_equation["bounding_box"] = (refined_x, refined_y, refined_w, refined_h)
            refined_equation["refined"] = True
            
            refined_equations.append(refined_equation)
        
        self.detected_equations = refined_equations
    
    def _check_equals_sign_profile(self, roi):
        """Check if ROI has profile consistent with equals sign"""
        # Sum pixels horizontally to get vertical profile
        h_profile = np.sum(roi, axis=1)
        
        # Normalize profile
        if np.max(h_profile) > 0:
            h_profile = h_profile / np.max(h_profile)
        
        # Count peaks in profile
        peaks, _ = self._find_peaks(h_profile)
        
        # Equals sign should have 2 distinct horizontal lines
        return len(peaks) == 2
    
    def _find_peaks(self, profile, min_distance=2, threshold=0.5):
        """Find peaks in a profile"""
        # Smooth profile
        smoothed = ndimage.gaussian_filter1d(profile, sigma=1)
        
        # Find local maxima
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > smoothed[i-1] and 
                smoothed[i] > smoothed[i+1] and 
                smoothed[i] > threshold * np.max(smoothed)):
                peaks.append(i)
        
        # Filter peaks that are too close
        filtered_peaks = []
        for peak in peaks:
            if not filtered_peaks or abs(peak - filtered_peaks[-1]) >= min_distance:
                filtered_peaks.append(peak)
        
        return filtered_peaks, smoothed
    
    def _analyze_equations(self, equations, image):
        """
        Analyze each detected equation
        Extracts operation type, numbers, and checks for errors
        """
        equation_data = []
        
        for idx, equation in enumerate(equations):
            # Extract equation components
            left_side = equation["left_side"]
            right_side = equation["right_side"]
            bbox = equation["bounding_box"]
            
            # Determine operation type (addition, subtraction, etc.)
            operation, operands = self._detect_operation(left_side, image)
            
            # Extract result from right side
            result = self._extract_result(right_side, image)
            
            # Calculate correct result based on operands and operation
            correct_result = self._calculate_result(operands, operation)
            
            # Check if result is correct
            is_correct = (result == correct_result)
            
            # Store equation analysis
            eq_data = {
                "id": idx + 1,
                "bounding_box": bbox,
                "operation": operation,
                "operands": operands,
                "student_result": result,
                "correct_result": correct_result,
                "is_correct": is_correct,
                "confidence": self._calculate_detection_confidence(left_side, right_side)
            }
            
            equation_data.append(eq_data)
        
        return equation_data
    
    def _detect_operation(self, left_side, image):
        """
        Detect mathematical operation and operands from the left side of equation
        Uses character recognition and positioning to identify operations
        """
        # Default values
        operation = "unknown"
        operands = []
        
        # Sort boxes left to right
        left_side.sort(key=lambda box: box[0])
        
        # Find operation symbol
        for i, (x, y, w, h, _) in enumerate(left_side):
            # Extract symbol ROI
            roi = image[y:y+h, x:x+w]
            
            # Check aspect ratio and size for operation symbol
            aspect_ratio = w / max(h, 1)
            
            # Check for addition symbol
            if 0.8 < aspect_ratio < 1.2:  # Plus tends to be squarish
                # Check for vertical and horizontal line profile
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Check horizontal and vertical profiles for plus sign pattern
                h_profile = np.sum(binary, axis=0)
                v_profile = np.sum(binary, axis=1)
                
                h_peaks, _ = self._find_peaks(h_profile / np.max(h_profile) if np.max(h_profile) > 0 else h_profile)
                v_peaks, _ = self._find_peaks(v_profile / np.max(v_profile) if np.max(v_profile) > 0 else v_profile)
                
                if len(h_peaks) == 1 and len(v_peaks) == 1:
                    operation = "addition"
                    
                    # Extract operands (numbers before and after + sign)
                    if i > 0:
                        first_operand = self._extract_number(left_side[:i], image)
                        operands.append(first_operand)
                    
                    if i < len(left_side) - 1:
                        second_operand = self._extract_number(left_side[i+1:], image)
                        operands.append(second_operand)
                    
                    break
            
            # Check for subtraction symbol
            elif 1.5 < aspect_ratio < 4.0 and h < w:  # Minus is typically horizontal rectangle
                # Check horizontal profile for minus sign pattern
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                h_profile = np.sum(binary, axis=0)
                v_profile = np.sum(binary, axis=1)
                
                v_peaks, _ = self._find_peaks(v_profile / np.max(v_profile) if np.max(v_profile) > 0 else v_profile)
                
                if len(v_peaks) == 1:
                    operation = "subtraction"
                    
                    # Extract operands (numbers before and after - sign)
                    if i > 0:
                        first_operand = self._extract_number(left_side[:i], image)
                        operands.append(first_operand)
                    
                    if i < len(left_side) - 1:
                        second_operand = self._extract_number(left_side[i+1:], image)
                        operands.append(second_operand)
                    
                    break
        
        # If no operation found but we have components, make best guess
        if operation == "unknown" and left_side:
            # Check for pattern: single number followed by equals
            operands = [self._extract_number(left_side, image)]
        
        return operation, operands
    
    def _extract_number(self, components, image):
        """
        Extract a number from a set of components
        Uses basic digit recognition for education context
        """
        # For demo purposes, use a very simple approach
        # In a real implementation, use the OCR model here
        
        # For now, we'll extract based on total width as proxy for number of digits
        if not components:
            return 0
            
        total_width = sum(comp[2] for comp in components)
        
        # Simulate recognition based on width
        # This is just a placeholder - real system would use OCR
        if total_width < 20:
            return 1
        elif total_width < 30:
            return 2
        else:
            return 3
    
    def _extract_result(self, right_side, image):
        """Extract the result from the right side of the equation"""
        # Similar placeholder approach as extract_number
        return self._extract_number(right_side, image)
    
    def _calculate_result(self, operands, operation):
        """Calculate the correct result based on operands and operation"""
        if len(operands) < 2:
            return 0
            
        if operation == "addition":
            return operands[0] + operands[1]
        elif operation == "subtraction":
            return operands[0] - operands[1]
        else:
            return 0
    
    def _calculate_detection_confidence(self, left_side, right_side):
        """Calculate confidence score for equation detection"""
        # Placeholder for confidence calculation
        # Real implementation would use model confidences
        if not left_side or not right_side:
            return 0.5
        return 0.8
    
    def _create_debug_visualizations(self, image_path):
        """Create debug visualizations to help understand localization results"""
        # Load original image for visualization
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = self.original_image.copy()
        
        # Visualize detected lines
        line_vis = image.copy()
        for line in self.detected_lines:
            for x, y, w, h, _ in line["boxes"]:
                cv2.rectangle(line_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        self.debug_images['lines'] = line_vis
        
        # Visualize detected equations with improved boxes
        equation_vis = image.copy()
        for eq_idx, equation in enumerate(self.detected_equations):
            # Draw bounding box with adjustable color based on refinement
            x, y, w, h = equation["bounding_box"]
            # Use a different color for refined boxes
            box_color = (0, 165, 255) if equation.get("refined", False) else (0, 0, 255)
            cv2.rectangle(equation_vis, (x, y), (x+w, y+h), box_color, 2)
            
            # Add equation label with confidence if available
            conf_text = f"Eq {eq_idx+1}"
            if "confidence" in equation:
                conf_text += f" ({equation['confidence']:.2f})"
            cv2.putText(equation_vis, conf_text, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # Highlight equals sign
            eq_x, eq_y, eq_w, eq_h = equation["equals_sign"]
            cv2.rectangle(equation_vis, (eq_x, eq_y), (eq_x+eq_w, eq_y+eq_h), (255, 0, 0), 2)
            
            # Optionally mark the centroid for alignment verification
            if "centroid" in equation:
                cx, cy = equation["centroid"]
                cv2.circle(equation_vis, (int(cx), int(cy)), 3, (0, 255, 255), -1)
        
        self.debug_images['equations'] = equation_vis
        
        # Save debug images if needed
        if isinstance(image_path, str):
            debug_dir = os.path.join(os.path.dirname(image_path), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            base_name = os.path.basename(image_path).split('.')[0]
            
            for name, img in self.debug_images.items():
                output_path = os.path.join(debug_dir, f"{base_name}_{name}.jpg")
                cv2.imwrite(output_path, img)
    
    def generate_practice_sheet(self, analysis_results, output_path):
        """
        Generate a practice sheet based on error analysis
        Creates targeted exercises to address specific weaknesses
        """
        # Extract error patterns
        errors = []
        for eq in analysis_results.get("equations", []):
            if not eq.get("is_correct", True):
                errors.append({
                    "operation": eq.get("operation", "unknown"),
                    "operands": eq.get("operands", []),
                    "student_result": eq.get("student_result", 0),
                    "correct_result": eq.get("correct_result", 0)
                })
        
        # Create practice sheet image
        sheet_width, sheet_height = 1000, 1400
        sheet = np.ones((sheet_height, sheet_width, 3), dtype=np.uint8) * 255
        
        # Add header
        cv2.putText(sheet, "Math Practice Sheet", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(sheet, "Based on your specific needs", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add practice problems based on error patterns
        y_offset = 150
        
        # Group errors by operation
        operation_errors = defaultdict(list)
        for error in errors:
            operation_errors[error["operation"]].append(error)
        
        # Generate practice for each operation type
        for operation, op_errors in operation_errors.items():
            # Add section header
            y_offset += 70
            operation_name = operation.capitalize() if operation != "unknown" else "Mixed Operations"
            cv2.putText(sheet, f"{operation_name} Practice", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            y_offset += 50
            
            # Generate 5 practice problems for this operation
            practice_problems = self._generate_practice_problems(operation, op_errors, 5)
            
            # Draw problems
            for i, problem in enumerate(practice_problems):
                problem_text = f"{problem['text']}"
                cv2.putText(sheet, problem_text, 
                           (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                
                # Draw answer line
                cv2.line(sheet, (350, y_offset - 15), (450, y_offset - 15), (0, 0, 0), 2)
                
                y_offset += 70
            
            # Add a tip for this operation
            tip = self._get_operation_tip(operation)
            y_offset += 20
            cv2.putText(sheet, "Tip:", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            y_offset += 30
            # Split tip into multiple lines if needed
            words = tip.split()
            line = ""
            for word in words:
                test_line = line + " " + word if line else word
                if len(test_line) > 50:  # max chars per line
                    cv2.putText(sheet, line, 
                               (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    y_offset += 30
                    line = word
                else:
                    line = test_line
            
            if line:
                cv2.putText(sheet, line, 
                           (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            y_offset += 50
        
        # Save the practice sheet
        cv2.imwrite(output_path, sheet)
        
        return output_path
    
    def _generate_practice_problems(self, operation, errors, count):
        """Generate practice problems based on error patterns"""
        problems = []
        
        # Analyze error patterns
        error_values = set()
        for error in errors:
            for operand in error.get("operands", []):
                error_values.add(operand)
        
        # Default values if no clear pattern
        if not error_values:
            error_values = {1, 2, 3}
        
        # Generate problems focusing on error values
        for i in range(count):
            if operation == "addition":
                a = np.random.choice(list(error_values))
                b = np.random.randint(1, 5)  # Vary second operand
                problem = {
                    "text": f"{a} + {b} = _____",
                    "answer": a + b
                }
            elif operation == "subtraction":
                a = np.random.choice(list(error_values))
                b = np.random.randint(1, min(a, 5))  # Keep answers positive
                problem = {
                    "text": f"{a} - {b} = _____",
                    "answer": a - b
                }
            else:
                # Mixed operation
                a = np.random.choice(list(error_values))
                b = np.random.randint(1, 5)
                op = "+" if np.random.random() > 0.5 else "-"
                problem = {
                    "text": f"{a} {op} {b} = _____",
                    "answer": a + b if op == "+" else a - b
                }
            
            problems.append(problem)
        
        return problems
    
    def _get_operation_tip(self, operation):
        """Get a helpful tip for an operation"""
        tips = {
            "addition": "When adding, count forward from the first number. You can use your fingers or draw dots.",
            "subtraction": "When subtracting, count backward from the first number. Start with the bigger number and take away the smaller number.",
            "unknown": "Break down each problem into steps. Check your work by using the opposite operation."
        }
        
        return tips.get(operation, "Practice these problems carefully step by step.")
