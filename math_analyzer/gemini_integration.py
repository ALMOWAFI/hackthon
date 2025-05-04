import os
import json
import base64
import cv2
import requests
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
def load_api_config():
    """Load API configuration from api.env file"""
    env_path = Path(__file__).parent / "api.env"
    load_dotenv(dotenv_path=env_path)
    
    config = {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_version": os.getenv("MODEL_VERSION", "gemini-pro-vision"),
        "ocr_engine_path": os.getenv("OCR_ENGINE_PATH")
    }
    
    if not config["api_key"]:
        raise ValueError("API key not found in environment variables")
        
    return config

class GeminiMathAnalyzer:
    """Class to integrate with Google's Gemini API for math analysis"""
    
    def __init__(self):
        """Initialize the Gemini Math Analyzer"""
        self.config = load_api_config()
        self.api_key = self.config["api_key"]
        self.model = self.config["model_version"]
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    def encode_image(self, image_path):
        """Encode image to base64 for API request"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    def analyze_math_image(self, image_path, prompt=None):
        """
        Send image to Gemini for math analysis
        
        Args:
            image_path: Path to the math homework image
            prompt: Optional custom prompt to guide the analysis
            
        Returns:
            JSON response with analysis results
        """
        # Default prompt if none provided
        if prompt is None:
            prompt = (
                "Analyze this math homework image. Identify all math problems, "
                "check for errors in calculations or steps, and provide feedback. "
                "For each error, specify the exact location, the type of error, "
                "correction needed, and a brief explanation. "
                "Format your response as JSON with the following structure: "
                "{ 'errors': [ { 'text': 'error text', 'type': 'error type', "
                "'correction': 'corrected text', 'explanation': 'explanation', "
                "'bounding_box': { 'top_left_x': x1, 'top_left_y': y1, 'bottom_right_x': x2, 'bottom_right_y': y2 } } ] }"
            )
            
        # Encode image
        try:
            encoded_image = self.encode_image(image_path)
        except Exception as e:
            print(f"Warning: Image encoding failed: {str(e)}")
            print("Falling back to local analysis only")
            return {"error": f"Image encoding failed: {str(e)}"}
            
        # Prepare request
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image
                            }
                        }
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        # Make request
        try:
            print("Sending request to Gemini API...")
            response = requests.post(url, json=payload, headers=headers)
            
            # Print detailed error information
            if response.status_code != 200:
                print(f"API Error - Status Code: {response.status_code}")
                print(f"Error Response: {response.text}")
                response.raise_for_status()
                
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            print("Falling back to local analysis only")
            return {"error": f"API request failed: {str(e)}"}
            
    def process_response(self, response):
        """Process and extract structured information from Gemini response"""
        try:
            if "error" in response:
                return response
                
            text_response = response["candidates"][0]["content"]["parts"][0]["text"]
            
            # Try to extract structured data if present
            # Look for JSON blocks in the response
            json_start = text_response.find("```json")
            json_end = text_response.find("```", json_start + 7) if json_start != -1 else -1
            
            if json_start != -1 and json_end != -1:
                json_str = text_response[json_start + 7:json_end].strip()
                try:
                    structured_data = json.loads(json_str)
                    return {
                        "raw_response": text_response,
                        "structured_data": structured_data
                    }
                except json.JSONDecodeError:
                    pass
            
            # If no JSON found, return the raw text
            return {
                "raw_response": text_response,
                "structured_data": None
            }
            
        except Exception as e:
            return {"error": f"Failed to process response: {str(e)}"}
            
    def analyze_and_visualize(self, image_path, output_dir="results"):
        """Analyze math image and visualize results"""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze image
        print(f"Sending image to Gemini API for analysis: {image_path}")
        response = self.analyze_math_image(image_path)
        processed = self.process_response(response)
        
        if "error" in processed:
            print(f"Error: {processed['error']}")
            return processed
            
        # Save the raw response
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        response_path = os.path.join(output_dir, f"{base_name}_gemini_response.json")
        
        with open(response_path, "w") as f:
            json.dump(processed, f, indent=2)
            
        print(f"Saved Gemini response to {response_path}")
        
        # If structured data is available, visualize results
        if processed.get("structured_data"):
            try:
                # Load original image
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": f"Could not load image: {image_path}"}
                    
                # Create visualization based on error types
                marked_image = self._mark_errors_on_image(image, processed["structured_data"])
                
                # Save the marked image
                output_path = os.path.join(output_dir, f"{base_name}_gemini_marked.jpg")
                cv2.imwrite(output_path, marked_image)
                print(f"Saved visualization to {output_path}")
                
                return {
                    "response": processed,
                    "marked_image_path": output_path
                }
            except Exception as e:
                return {"error": f"Visualization failed: {str(e)}"}
        
        return processed
    
    def _mark_errors_on_image(self, image, data):
        """Mark errors on the image based on structured data from Gemini"""
        # Create a copy of the image to draw on
        marked_image = image.copy()
        
        # Process each error if available in the structured data
        errors = data.get("errors", [])
        if not errors and "problems" in data:
            # Alternative structure sometimes returned by Gemini
            for problem in data["problems"]:
                if "errors" in problem:
                    errors.extend(problem["errors"])
                    
        for i, error in enumerate(errors):
            if "bounding_box" in error:
                bbox = error["bounding_box"]
                # Convert normalized coordinates to pixel coordinates if needed
                if all(0 <= coord <= 1 for coord in [bbox.get("top_left_x", 0), bbox.get("top_left_y", 0), 
                                                   bbox.get("bottom_right_x", 1), bbox.get("bottom_right_y", 1)]):
                    h, w = marked_image.shape[:2]
                    top_left = (int(bbox["top_left_x"] * w), int(bbox["top_left_y"] * h))
                    bottom_right = (int(bbox["bottom_right_x"] * w), int(bbox["bottom_right_y"] * h))
                else:
                    # Assume pixel coordinates
                    top_left = (int(bbox["top_left_x"]), int(bbox["top_left_y"]))
                    bottom_right = (int(bbox["bottom_right_x"]), int(bbox["bottom_right_y"]))
                
                # Draw rectangle around error
                cv2.rectangle(marked_image, top_left, bottom_right, (0, 0, 255), 2)
                
                # Draw error number
                cv2.putText(marked_image, f"#{i+1}", 
                           (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return marked_image

def main():
    """Main function to test Gemini integration"""
    # Check if we have a valid API key
    try:
        analyzer = GeminiMathAnalyzer()
        print("Successfully initialized Gemini Math Analyzer")
        
        # Test with a sample image if available
        test_image = "uploads/math7.jpeg"
        if os.path.exists(test_image):
            print(f"Analyzing test image: {test_image}")
            result = analyzer.analyze_and_visualize(test_image)
            
            if "error" in result:
                print(f"Analysis failed: {result['error']}")
            else:
                print("Analysis successful!")
                if "structured_data" in result.get("response", {}):
                    print(f"Found {len(result['response']['structured_data'].get('errors', []))} errors")
        else:
            print(f"Test image not found: {test_image}")
            
    except Exception as e:
        print(f"Initialization failed: {str(e)}")

if __name__ == "__main__":
    main()
