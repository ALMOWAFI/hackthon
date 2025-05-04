"""
Test script for processing math homework images with our enhanced system
"""

import os
import json
from math_analyzer.azure_integration import AzureMathAnalyzer

def run_test_on_image(image_path):
    """Run analysis on a test image and print detailed results"""
    print(f"Running analysis on image: {image_path}")
    
    # Initialize the analyzer
    analyzer = AzureMathAnalyzer()
    
    # Process the image
    results = analyzer.analyze_image(image_path)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
        
    # Initialize marked_image_path
    results['marked_image_path'] = None
    
    # Print the actual structure of the results for debugging
    print("\n===== ANALYSIS RESULTS STRUCTURE =====")
    for key in results.keys():
        print(f"Found key: {key}")
        
    # Check if OCR results are available
    if 'ocr_result' in results:
        ocr_result = results['ocr_result']
        print("\n===== DETECTED TEXT =====")
        
        # Display lines of text found in the image
        if 'lines' in ocr_result:
            for i, line in enumerate(ocr_result['lines']):
                print(f"Line #{i+1}: {line.get('text', 'No text')} (Confidence: {line.get('confidence', 'N/A')})")
                
                # Check if this looks like a math problem
                text = line.get('text', '')
                if any(op in text for op in ['+', '-', '×', '÷', '=', '²']):
                    print(f"  Detected potential math expression: {text}")
        
        # Display document type if available
        if 'document_type' in results:
            print(f"\nDocument type: {results['document_type']}")
            
        # Check for expressions that might be math problems
        math_expressions = []
        if 'expressions' in ocr_result:
            math_expressions = ocr_result['expressions']
        elif 'lines' in ocr_result:
            # Extract potential math expressions from lines
            for line in ocr_result['lines']:
                text = line.get('text', '')
                if any(op in text for op in ['+', '-', '×', '÷', '=', '²']):
                    math_expressions.append(line)
        
        print("\n===== POTENTIAL MATH PROBLEMS =====")
        for i, expr in enumerate(math_expressions):
            print(f"Problem #{i+1}: {expr.get('text', 'No text')}")
    
    # Create a simplified analysis from the OCR results for the JSON output
    simplified_analysis = {
        "detected_text": results.get('ocr_result', {}).get('lines', []),
        "document_type": results.get('document_type', 'unknown'),
        "potential_math_problems": [],
        "analysis_summary": "OCR analysis completed. For complete math analysis, please use the full Math Feedback System."
    }
    
    # Extract potential math problems from OCR results
    if 'ocr_result' in results and 'lines' in results['ocr_result']:
        for line in results['ocr_result']['lines']:
            text = line.get('text', '')
            if any(op in text for op in ['+', '-', '×', '÷', '=', '²']):
                simplified_analysis["potential_math_problems"].append({
                    "text": text,
                    "confidence": line.get('confidence', 0.0),
                    "bounding_box": line.get('bounding_box', [])
                })
    
    # Print summary
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Text lines detected: {len(results.get('ocr_result', {}).get('lines', []))}")
    print(f"Potential math problems: {len(simplified_analysis['potential_math_problems'])}")
    print(f"Document type: {simplified_analysis['document_type']}")
    
    # Create a nicely formatted JSON output
    output_path = os.path.join(os.path.dirname(image_path), "ocr_analysis.json")
    with open(output_path, "w") as f:
        json.dump(simplified_analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    # Path to the test image
    # Replace this with your actual image path
    image_dir = os.path.join(os.getcwd(), "test_images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Check for available test images
    available_images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    
    if available_images:
        test_image = os.path.join(image_dir, available_images[0])
        print(f"Using test image: {test_image}")
        run_test_on_image(test_image)
    else:
        print(f"No test images found in: {image_dir}")
        
        # Try to use one of the sample images from the main directory
        sample_images = [
            os.path.join(os.getcwd(), "math_test_sample.jpg"),
            os.path.join(os.getcwd(), "math_formulas_test.jpg"),
            os.path.join(os.getcwd(), "math_homework.jpg")
        ]
        
        for img in sample_images:
            if os.path.exists(img):
                print(f"Using alternative test image: {img}")
                run_test_on_image(img)
                break
        else:
            print("No test images found. Please add an image to test.")
