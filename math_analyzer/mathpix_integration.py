"""
MathPix API integration for math notation recognition
"""
import os
import requests
import base64
from dotenv import load_dotenv
from PIL import Image
import io
import json

class MathPixIntegration:
    def __init__(self):
        load_dotenv("math_analyzer/api.env")
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")
        self.base_url = "https://api.mathpix.com/v3"
    
    def _encode_image(self, image_path):
        """Convert image to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def recognize_math(self, image_path):
        """Send image to MathPix API for math recognition"""
        # Prepare headers
        headers = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "Content-type": "application/json"
        }
        
        # Prepare data
        data = {
            "src": f"data:image/jpeg;base64,{self._encode_image(image_path)}",
            "formats": ["text", "latex", "data"],
            "data_options": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
        
        # Make API request
        response = requests.post(
            f"{self.base_url}/text",
            headers=headers,
            json=data
        )
        
        return response.json()
    
    def process_math_image(self, image_path):
        """Process a math image and return structured results"""
        try:
            result = self.recognize_math(image_path)
            
            if "error" in result:
                print(f"Error from MathPix API: {result['error']}")
                return None
            
            return {
                "text": result.get("text", ""),
                "latex": result.get("latex", ""),
                "confidence": result.get("confidence", 0),
                "data": result.get("data", {})
            }
            
        except Exception as e:
            print(f"Error processing image with MathPix: {str(e)}")
            return None

def test_mathpix():
    """Test MathPix integration with a sample image"""
    mathpix = MathPixIntegration()
    
    # Check if credentials are set
    if mathpix.app_id == "trial_account" or mathpix.app_key == "trial_key":
        print("Please set up your MathPix credentials in api.env first!")
        print("Visit https://mathpix.com/ocr to get your free trial API keys")
        return
    
    print("MathPix integration is ready!")
    print(f"Using app_id: {mathpix.app_id[:5]}...")
    
    # Test with a sample image if available
    sample_images = [
        "test_images/math1.jpg",
        "test_images/equation.png",
        "test_homework_image.jpg"
    ]
    
    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"\nTesting with {image_path}...")
            result = mathpix.process_math_image(image_path)
            if result:
                print("Recognition successful!")
                print(f"Detected text: {result['text']}")
                print(f"LaTeX: {result['latex']}")
                print(f"Confidence: {result['confidence']}")
                break
    else:
        print("\nNo sample images found. Please add test images to test the integration.")

if __name__ == "__main__":
    test_mathpix()
