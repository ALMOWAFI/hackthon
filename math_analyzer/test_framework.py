"""
Testing Framework for Math OCR and Analysis System
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from .train_ocr_model import MathOCRTrainer
from .advanced_math_analyzer import AdvancedMathAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MathTestFramework:
    def __init__(self):
        self.ocr_trainer = MathOCRTrainer()
        self.math_analyzer = AdvancedMathAnalyzer()
        
        # Define test categories
        self.problem_types = {
            'arithmetic': {
                'addition': self._generate_addition_tests,
                'subtraction': self._generate_subtraction_tests,
                'multiplication': self._generate_multiplication_tests,
                'division': self._generate_division_tests
            },
            'algebra': {
                'linear': self._generate_linear_tests,
                'quadratic': self._generate_quadratic_tests,
                'systems': self._generate_system_tests
            },
            'geometry': {
                'pythagorean': self._generate_pythagorean_tests,
                'area': self._generate_area_tests,
                'volume': self._generate_volume_tests
            },
            'calculus': {
                'derivatives': self._generate_derivative_tests,
                'integrals': self._generate_integral_tests
            }
        }
        
    def _generate_addition_tests(self, num_tests=10):
        """Generate addition test cases"""
        tests = []
        for _ in range(num_tests):
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            tests.append({
                'problem': f"{a} + {b} = ?",
                'solution': str(a + b),
                'type': 'addition'
            })
        return tests
        
    def _generate_linear_tests(self, num_tests=10):
        """Generate linear equation test cases"""
        tests = []
        for _ in range(num_tests):
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            c = np.random.randint(1, 100)
            x = (c - b) / a
            tests.append({
                'problem': f"{a}x + {b} = {c}",
                'solution': f"x = {x}",
                'type': 'linear'
            })
        return tests
        
    def _generate_quadratic_tests(self, num_tests=10):
        """Generate quadratic equation test cases"""
        tests = []
        for _ in range(num_tests):
            a = np.random.randint(1, 5)
            b = np.random.randint(-10, 10)
            c = np.random.randint(-10, 10)
            tests.append({
                'problem': f"{a}x² + {b}x + {c} = 0",
                'solution': f"quadratic: a={a}, b={b}, c={c}",
                'type': 'quadratic'
            })
        return tests
        
    def generate_test_suite(self, categories=None):
        """Generate comprehensive test suite"""
        if categories is None:
            categories = list(self.problem_types.keys())
            
        test_suite = {}
        for category in categories:
            test_suite[category] = {}
            for problem_type, generator in self.problem_types[category].items():
                test_suite[category][problem_type] = generator()
                
        return test_suite
        
    def run_ocr_tests(self, image_dir):
        """Test OCR accuracy on real images"""
        results = {
            'total_images': 0,
            'successful_ocr': 0,
            'accuracy': 0.0,
            'errors': []
        }
        
        for image_file in os.listdir(image_dir):
            if not image_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(image_dir, image_file)
            try:
                # Load ground truth from corresponding JSON file
                truth_path = image_path.rsplit('.', 1)[0] + '.json'
                with open(truth_path, 'r') as f:
                    ground_truth = json.load(f)
                    
                # Run OCR
                ocr_result = self.ocr_trainer.model.predict(Image.open(image_path))
                
                # Compare with ground truth
                results['total_images'] += 1
                if self._compare_ocr_results(ocr_result, ground_truth):
                    results['successful_ocr'] += 1
                    
            except Exception as e:
                results['errors'].append({
                    'image': image_file,
                    'error': str(e)
                })
                
        results['accuracy'] = results['successful_ocr'] / results['total_images']
        return results
        
    def run_analysis_tests(self, test_suite):
        """Test math analysis accuracy"""
        results = {
            'by_category': {},
            'overall': {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0
            }
        }
        
        for category, problems in test_suite.items():
            category_results = {
                'total': 0,
                'correct': 0,
                'by_type': {}
            }
            
            for problem_type, tests in problems.items():
                type_results = {
                    'total': len(tests),
                    'correct': 0,
                    'examples': []
                }
                
                for test in tests:
                    analysis = self.math_analyzer.analyze_expression(test['problem'])
                    if self._verify_analysis(analysis, test):
                        type_results['correct'] += 1
                        
                    type_results['examples'].append({
                        'problem': test['problem'],
                        'expected': test['solution'],
                        'analysis': analysis
                    })
                    
                type_results['accuracy'] = type_results['correct'] / type_results['total']
                category_results['by_type'][problem_type] = type_results
                category_results['total'] += type_results['total']
                category_results['correct'] += type_results['correct']
                
            category_results['accuracy'] = category_results['correct'] / category_results['total']
            results['by_category'][category] = category_results
            
            results['overall']['total'] += category_results['total']
            results['overall']['correct'] += category_results['correct']
            
        results['overall']['accuracy'] = results['overall']['correct'] / results['overall']['total']
        return results
        
    def _verify_analysis(self, analysis, test_case):
        """Verify if analysis matches the test case"""
        if 'solution' not in analysis:
            return False
            
        if test_case['type'] == 'linear':
            return self._verify_linear_solution(analysis['solution'], test_case['solution'])
        elif test_case['type'] == 'quadratic':
            return self._verify_quadratic_solution(analysis['solution'], test_case['solution'])
            
        return str(analysis['solution']) == str(test_case['solution'])
        
    def run_all_tests(self, image_dir=None):
        """Run complete test suite"""
        results = {
            'ocr_results': None,
            'analysis_results': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate and run analysis tests
        test_suite = self.generate_test_suite()
        results['analysis_results'] = self.run_analysis_tests(test_suite)
        
        # Run OCR tests if image directory provided
        if image_dir:
            results['ocr_results'] = self.run_ocr_tests(image_dir)
            
        return results
        
    def save_results(self, results, output_file):
        """Save test results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
if __name__ == "__main__":
    # Example usage
    framework = MathTestFramework()
    
    # Run tests
    results = framework.run_all_tests(image_dir="test_images")
    
    # Save results
    framework.save_results(results, "test_results.json")
