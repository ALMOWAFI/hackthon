"""
Integrated test script for the Math Feedback System
This script tests the full pipeline from OCR to math analysis to feedback generation
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from math_analyzer.handwritten_math_ocr import HandwrittenMathOCR
from math_analyzer.azure_integration import AzureMathAnalyzer
from math_analyzer.advanced_math_analyzer import AdvancedMathAnalyzer
from math_analyzer.ocr_integration import MathRecognitionSystem

def test_ocr_components(image_path):
    """Test individual OCR components to ensure they're working"""
    print(f"\n===== TESTING OCR COMPONENTS ON {image_path} =====")
    
    results = {}
    
    # Test Azure OCR integration
    try:
        print("\n1. Testing Azure OCR Integration")
        azure_analyzer = AzureMathAnalyzer()
        azure_results = azure_analyzer.analyze_image(image_path)
        
        if "error" in azure_results:
            print(f"  Error: {azure_results['error']}")
        else:
            print(f"  Success! Azure OCR detected text from the image")
            print(f"  Document type: {azure_results.get('document_type', 'unknown')}")
            text_lines = azure_results.get('ocr_result', {}).get('lines', [])
            print(f"  Detected {len(text_lines)} text lines")
            
            # Display detected text
            for i, line in enumerate(text_lines):
                if i < 5:  # Just show first 5 lines
                    print(f"  Line {i+1}: {line.get('text', 'No text')}")
                    
        results["azure_ocr"] = azure_results
    except Exception as e:
        print(f"  Azure OCR test failed: {str(e)}")
        results["azure_ocr"] = {"error": str(e)}

    # Test specialized OCR
    try:
        print("\n2. Testing Handwritten Math OCR")
        math_ocr = HandwrittenMathOCR()
        ocr_results = math_ocr.recognize_expression(image_path)
        
        if not ocr_results:
            print("  No results returned from HandwrittenMathOCR")
            results["math_ocr"] = {"error": "No results returned"}
        else:
            print(f"  Success! Math OCR processed the image")
            print(f"  Detected {len(ocr_results.get('expressions', []))} math expressions")
            
            # Display detected expressions
            for i, expr in enumerate(ocr_results.get('expressions', [])[:3]):
                print(f"  Expression {i+1}: {expr.get('text', 'No text')} (Confidence: {expr.get('confidence', 'N/A')})")
                
            results["math_ocr"] = ocr_results
    except Exception as e:
        print(f"  Handwritten Math OCR test failed: {str(e)}")
        results["math_ocr"] = {"error": str(e)}
    
    return results

def test_integrated_system(image_path):
    """Test the fully integrated system"""
    print(f"\n===== TESTING INTEGRATED SYSTEM ON {image_path} =====")
    
    try:
        # Initialize the integrated system
        print("\nInitializing MathRecognitionSystem...")
        math_system = MathRecognitionSystem()
        
        # Process the homework image
        print(f"\nProcessing image through the integrated system...")
        results = math_system.process_homework_image(image_path)
        
        if not results:
            print("No results returned from integrated system")
            return {"error": "No results returned"}
            
        print("\n===== INTEGRATED RESULTS =====")
        if "feedback" in results:
            feedback_count = len(results.get("feedback", []))
            print(f"Generated {feedback_count} feedback items")
            
            # Show sample feedback
            for i, feedback in enumerate(results.get("feedback", [])[:3]):
                print(f"\nFeedback {i+1}:")
                for key, value in feedback.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        
        # Save the detailed results
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"integrated_results_{Path(image_path).stem}.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"\nDetailed results saved to {output_path}")
        return results
    
    except Exception as e:
        print(f"Integrated system test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def test_math_analyzer(expression):
    """Test the math analysis component with a given expression"""
    print(f"\n===== TESTING MATH ANALYZER WITH '{expression}' =====")
    
    try:
        # Initialize the math analyzer
        analyzer = AdvancedMathAnalyzer()
        
        # Analyze the expression
        analysis = analyzer.analyze_expression(expression)
        
        print("\nAnalysis Results:")
        print(f"Expression type: {analysis.get('type', 'Unknown')}")
        
        # Print errors
        errors = analysis.get('errors', [])
        print(f"\nDetected {len(errors)} errors:")
        for i, error in enumerate(errors):
            print(f"  Error {i+1}: {error.get('type', 'Unknown')} - {error.get('description', 'No description')}")
        
        # Print feedback
        feedback = analysis.get('feedback', [])
        print(f"\nGenerated {len(feedback)} feedback items:")
        for i, fb in enumerate(feedback):
            print(f"  Feedback {i+1} ({fb.get('type', 'Unknown')}): {fb.get('content', 'No content')[:100]}...")
        
        return analysis
    
    except Exception as e:
        print(f"Math analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_comprehensive_tests():
    """Run a comprehensive set of tests on all system components"""
    
    # Test images
    test_images = [
        os.path.join(os.getcwd(), "math_test_sample.jpg"),
        os.path.join(os.getcwd(), "math_formulas_test.jpg"),
        os.path.join(os.getcwd(), "math_homework.jpg"),
        os.path.join(os.getcwd(), "test_images", "math8.jpeg")
    ]
    
    # Test expressions
    test_expressions = [
        "1 + 1 = 3",  # Simple error
        "x + 5 = 10",  # Linear equation
        "x^2 + 3x + 2 = 0",  # Quadratic equation
        "a^2 + b^2 = c^2",  # Pythagorean theorem
        "3² + 4² = 5²",  # Numerical pythagorean
    ]
    
    results = {}
    
    # 1. Test OCR components on all images
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting OCR with {os.path.basename(image_path)}")
            ocr_results = test_ocr_components(image_path)
            results[f"ocr_{os.path.basename(image_path)}"] = ocr_results
    
    # 2. Test math analyzer on expressions
    for expr in test_expressions:
        analysis = test_math_analyzer(expr)
        results[f"analysis_{expr}"] = analysis
    
    # 3. Test integrated system on one image
    for image_path in test_images:
        if os.path.exists(image_path):
            integrated_results = test_integrated_system(image_path)
            results[f"integrated_{os.path.basename(image_path)}"] = integrated_results
            break  # Just test one image for now
    
    return results

if __name__ == "__main__":
    print("===== COMPREHENSIVE MATH FEEDBACK SYSTEM TEST =====")
    print("This test will verify all components of the system are working properly.")
    
    # Check if we have any test images
    test_img_dir = os.path.join(os.getcwd(), "test_images")
    main_dir = os.getcwd()
    
    # Identify available test images
    test_images = []
    for img_name in ["math_test_sample.jpg", "math_formulas_test.jpg", "math_homework.jpg"]:
        img_path = os.path.join(main_dir, img_name)
        if os.path.exists(img_path):
            test_images.append(img_path)
    
    # Add images from test_images directory
    if os.path.exists(test_img_dir):
        for file in os.listdir(test_img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_img_dir, file))
    
    if not test_images:
        print("No test images found. Please add test images to the project directory.")
    else:
        print(f"Found {len(test_images)} test images.")
        
        # Run full test suite
        results = run_comprehensive_tests()
        
        # Save detailed results
        os.makedirs("results", exist_ok=True)
        with open(os.path.join("results", "comprehensive_test_results.json"), "w", encoding="utf-8") as f:
            # Convert complex objects to strings to make them JSON serializable
            serializable_results = {}
            for key, value in results.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[key] = value
                except:
                    # If not serializable, convert to string representation
                    serializable_results[key] = str(value)
                    
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        print("\n===== TEST COMPLETE =====")
        print(f"Detailed results saved to {os.path.join('results', 'comprehensive_test_results.json')}")
