import json
from math_analyzer.azure_integration import AzureMathAnalyzer

def main():
    analyzer = AzureMathAnalyzer()
    result = analyzer.analyze_math_image('test_images/math8.jpeg')
    
    # Save results to a file
    with open('results/math8_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Analysis results saved to results/math8_analysis.json")

if __name__ == "__main__":
    main()
