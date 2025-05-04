"""
Azure OCR Integration for Math Expression Detection

This module provides integration with Azure Cognitive Services for superior OCR
performance, especially with handwritten math notation.

To use this module:
1. Get an Azure Cognitive Services API key and endpoint
2. Set environment variables:
   - AZURE_VISION_KEY: Your API key
   - AZURE_VISION_ENDPOINT: Your endpoint URL (e.g., https://your-resource.cognitiveservices.azure.com/)

Or provide these directly when initializing the AzureOCR class.
"""

import os
import sys
import time
import json
import requests
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Type alias for bounding boxes
BBox = tuple[int, int, int, int, str]


class AzureOCR:
    """Azure Cognitive Services OCR integration for math expression detection."""
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Initialize Azure OCR with API key and endpoint.
        
        Args:
            api_key: Azure Cognitive Services API key (or set AZURE_VISION_KEY env var)
            endpoint: Azure endpoint URL (or set AZURE_VISION_ENDPOINT env var)
        """
        self.api_key = api_key or os.environ.get("AZURE_VISION_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
        self._is_available = self.api_key is not None and self.endpoint is not None
    
    def is_available(self) -> bool:
        """Check if Azure OCR is properly configured."""
        return self._is_available
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for Azure OCR."""
        return """
To set up Azure OCR:
1. Create an Azure Cognitive Services resource at https://portal.azure.com
2. Get your API key and endpoint
3. Set these environment variables:
   - AZURE_VISION_KEY: Your API key
   - AZURE_VISION_ENDPOINT: Your endpoint URL

Or provide them directly when initializing AzureOCR.
"""
    
    def detect_expressions(self, image: np.ndarray) -> List[BBox]:
        """
        Detect math expressions in an image using Azure OCR.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of (x1, y1, x2, y2, text) tuples for expressions
        """
        if not self._is_available:
            print("[AzureOCR] Not properly configured. Cannot detect expressions.")
            print(self.get_setup_instructions())
            return []
        
        # Convert image to bytes
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        # API details
        vision_url = f"{self.endpoint}/vision/v3.2/read/analyze"
        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        
        try:
            # Submit the image for analysis
            response = requests.post(
                vision_url, 
                headers=headers, 
                data=img_bytes
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Get operation location for result polling
            operation_location = response.headers["Operation-Location"]
            
            # Poll for results (with timeout)
            analysis = {}
            poll_timeout = 30  # seconds
            start_time = time.time()
            
            while (time.time() - start_time) < poll_timeout:
                result_response = requests.get(
                    operation_location, 
                    headers={"Ocp-Apim-Subscription-Key": self.api_key}
                )
                analysis = result_response.json()
                
                # Check if analysis is complete
                if "status" in analysis and analysis["status"] == "succeeded":
                    break
                
                # Wait before polling again
                time.sleep(1)
            
            # Process results
            boxes = []
            
            # Special check for division operations that might be missed
            # Pre-process image to detect potential division lines
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use multiple threshold values to ensure we catch the division lines
            thresholds = [
                cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1],
                cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1],
                cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1],
                cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
            ]
            
            division_regions = []
            
            # Process each threshold to find horizontal lines
            for thresh in thresholds:
                # Detect horizontal lines that could be division symbols with varied parameters
                for width in [10, 15, 20, 25]:
                    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
                    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
                    
                    # Find contours that might be division lines
                    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Only consider lines that could be division operations (more permissive)
                        if w > 5 and h <= 5:  # Lowered width threshold, increased height threshold
                            # Expand region to include numerator and denominator
                            expanded_y = max(0, y - 40)  # Increased vertical expansion
                            expanded_h = min(image.shape[0] - expanded_y, 80)  # Increased height
                            division_regions.append((x, expanded_y, x + w, expanded_y + expanded_h))
            
            # Merge overlapping division regions
            if division_regions:
                merged_regions = []
                division_regions.sort()  # Sort by x coordinate
                
                current_region = division_regions[0]
                for region in division_regions[1:]:
                    # Check if regions overlap
                    curr_x1, curr_y1, curr_x2, curr_y2 = current_region
                    x1, y1, x2, y2 = region
                    
                    if x1 <= curr_x2 and y1 <= curr_y2 and x2 >= curr_x1 and y2 >= curr_y1:
                        # Merge regions
                        merged_x1 = min(curr_x1, x1)
                        merged_y1 = min(curr_y1, y1)
                        merged_x2 = max(curr_x2, x2)
                        merged_y2 = max(curr_y2, y2)
                        current_region = (merged_x1, merged_y1, merged_x2, merged_y2)
                    else:
                        merged_regions.append(current_region)
                        current_region = region
                
                merged_regions.append(current_region)
                division_regions = merged_regions
            
            # Add to debug output
            print(f"[AzureOCR] Found {len(division_regions)} potential division operations")
            
            # Standard OCR processing
            if "analyzeResult" in analysis and "readResults" in analysis["analyzeResult"]:
                for read_result in analysis["analyzeResult"]["readResults"]:
                    for line in read_result["lines"]:
                        text = line["text"]
                        
                        # Check if this looks like a math expression
                        if self._is_potential_math(text):
                            # Get bounding box
                            bbox = line["boundingBox"]
                            x1, y1 = min(bbox[0], bbox[2], bbox[4], bbox[6]), min(bbox[1], bbox[3], bbox[5], bbox[7])
                            x2, y2 = max(bbox[0], bbox[2], bbox[4], bbox[6]), max(bbox[1], bbox[3], bbox[5], bbox[7])
                            
                            # Convert to integers
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            boxes.append((x1, y1, x2, y2, text))
            
            # Add division regions with special handling
            for x1, y1, x2, y2 in division_regions:
                # Extract region and run additional OCR if needed
                region = image[y1:y2, x1:x2]
                region_height, region_width = region.shape[:2]
                
                # Only process regions of sufficient size
                if region_width > 20 and region_height > 20:
                    # Check if this region overlaps with existing boxes
                    is_new_region = True
                    for bx1, by1, bx2, by2, _ in boxes:
                        if (x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1):
                            is_new_region = False
                            break
                    
                    if is_new_region:
                        # Add as a potential division equation
                        boxes.append((x1, y1, x2, y2, "division_operation"))
            
            return boxes
            
        except Exception as e:
            print(f"[AzureOCR] Error: {e}")
            return []
            
    def _is_potential_math(self, text: str) -> bool:
        """Check if text could be a mathematical expression."""
        # Check for numbers
        if any(c.isdigit() for c in text):
            # Check for math operators
            if any(op in text for op in "+-*/=()[]{}"):
                return True
                
        # Check for algebra (variables)
        if any(c.isalpha() for c in text) and any(op in text for op in "=+-*/()"):
            return True
            
        # Specific checks for division notation
        if "/" in text or "\\" in text:
            return True
            
        return False
