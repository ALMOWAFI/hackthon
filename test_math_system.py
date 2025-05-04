"""
Test script to demonstrate the complete Math Feedback System
This shows how to use the system with both Azure OCR and the specialized handwritten math OCR
"""

import os
import argparse
import json
import cv2
import matplotlib.pyplot as plt
from math_analyzer.ocr_integration import MathRecognitionSystem
from math_analyzer.azure_integration import AzureMathAnalyzer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the Math Feedback System")
    
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the homework image to analyze")
    
    parser.add_argument("--ocr", type=str, default="azure", choices=["azure", "specialized"],
                        help="Which OCR system to use (azure or specialized)")
    
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained handwritten math OCR model (required if ocr=specialized)")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the results")
    
    return parser.parse_args()

def display_results(image_path, feedback_path, visualization_path):
    """Display the results with matplotlib"""
    # Load the original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Load the visualization
    visualization = cv2.imread(visualization_path)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    # Load the feedback
    with open(feedback_path, "r") as f:
        feedback = json.load(f)
    
    # Plot the images
    plt.figure(figsize=(18, 10))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")
    
    # Visualization
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Analysis Results")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(visualization_path), "comparison.png"))
    
    # Print some key feedback
    print("\n===== FEEDBACK SUMMARY =====")
    print(f"Total problems: {feedback['assignment_info']['total_problems']}")
    print(f"Correct problems: {feedback['assignment_info']['total_correct']}")
    print(f"Score: {feedback['assignment_info']['score_percentage']}%")
    
    print("\nDetected Problems:")
    for problem in feedback['problems']:
        status = "✓" if problem['is_correct'] else "✗"
        print(f"  {status} Problem #{problem['id']}: {problem['text']}")
        
        if not problem['is_correct']:
            print(f"    Error: {problem['error_details']['description']}")
            print(f"    Feedback: {problem['feedback']['learning_strategy']}")
    
    print("\nAreas for Improvement:")
    for area in feedback['summary']['areas_for_improvement']:
        print(f"  • {area}")
        
    print("\nRecommendations:")
    for rec in feedback['summary']['teacher_recommendations']:
        print(f"  • {rec}")
    
    return feedback

def test_azure_ocr(image_path, output_dir):
    """Test the Azure OCR integration directly"""
    print("\n===== Testing Azure OCR Integration =====")
    
    # Initialize Azure analyzer
    azure = AzureMathAnalyzer()
    
    # Process the image
    print(f"Processing image: {image_path}")
    results = azure.analyze_and_visualize(image_path, output_dir=output_dir)
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        return None
    
    print(f"Azure OCR analysis complete.")
    print(f"Found {results.get('error_count', 0)} errors in the image.")
    print(f"Results saved to: {output_dir}")
    
    return results

def test_complete_system(image_path, ocr_type, model_path, output_dir):
    """Test the complete math feedback system"""
    print(f"\n===== Testing Complete Math Feedback System with {ocr_type.upper()} OCR =====")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the system
    system = MathRecognitionSystem(ocr_model_path=model_path if ocr_type == "specialized" else None)
    
    # Process the image
    print(f"Processing image: {image_path}")
    
    results = system.process_homework_image(
        image_path,
        use_specialized_ocr=(ocr_type == "specialized"),
        output_dir=output_dir
    )
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
        return None
    
    print(f"Analysis complete.")
    print(f"Feedback saved to: {results['feedback_file']}")
    print(f"Visualization saved to: {results['visualization_file']}")
    
    # Display the results
    feedback = display_results(image_path, results['feedback_file'], results['visualization_file'])
    
    return feedback

def main():
    """Main function to test the system"""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if args.ocr == "specialized" and not args.model:
        print("Error: Model path is required when using specialized OCR.")
        print("Use --model to specify the path to the trained model.")
        return
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test Azure OCR directly (for comparison)
    azure_results = test_azure_ocr(args.image, args.output_dir)
    
    # Test the complete system
    system_results = test_complete_system(args.image, args.ocr, args.model, args.output_dir)
    
    print("\n===== Testing complete! =====")
    print(f"All results saved to: {args.output_dir}")
    print("You can now review the feedback and visualizations.")

if __name__ == "__main__":
    main()
