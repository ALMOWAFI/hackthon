import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from typing import List, Dict, Any, Tuple, Union, Optional
import datetime
import cv2
from .advanced_ocr import AdvancedMathOCR, MathOCRFactory
from .feedback_generator import MathFeedbackGenerator
from .dataset_generator import MathProblemGenerator, DatasetGenerator
import concurrent.futures
from dataclasses import dataclass
import re

@dataclass
class TestCase:
    """Structure for individual test cases"""
    id: str
    type: str  # 'ocr', 'feedback', 'integration', etc.
    input_data: Any
    expected_output: Any
    metadata: Dict = None
    
@dataclass
class TestResult:
    """Structure for test results"""
    test_case_id: str
    passed: bool
    actual_output: Any
    error_message: Optional[str] = None
    performance_metrics: Dict = None
    timestamp: datetime.datetime = datetime.datetime.now()

class MathFeedbackTester:
    """Comprehensive testing framework for math feedback systems"""
    
    def __init__(self, results_dir="test_results"):
        self.results_dir = results_dir
        self.ocr_tester = OCRTester()
        self.feedback_tester = FeedbackTester()
        self.integration_tester = IntegrationTester()
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Track test history
        self.test_history = []
        
    def run_automated_test_suite(self, config=None):
        """Run a full automated test suite based on config"""
        config = config or self._default_test_config()
        
        results = {
            "ocr_tests": [],
            "feedback_tests": [],
            "integration_tests": [],
            "summary": {}
        }
        
        # Run OCR tests
        if config.get("run_ocr_tests", True):
            print("Running OCR tests...")
            ocr_results = self.ocr_tester.run_test_suite(
                n_samples=config.get("ocr_samples", 10),
                doc_types=config.get("ocr_doc_types", ["handwritten", "printed", "mixed"])
            )
            results["ocr_tests"] = ocr_results
            
        # Run feedback generator tests
        if config.get("run_feedback_tests", True):
            print("Running feedback generator tests...")
            feedback_results = self.feedback_tester.run_test_suite(
                n_samples=config.get("feedback_samples", 20),
                error_types=config.get("error_types", ["calculation", "procedural", "conceptual"])
            )
            results["feedback_tests"] = feedback_results
            
        # Run integration tests
        if config.get("run_integration_tests", True):
            print("Running integration tests...")
            integration_results = self.integration_tester.run_test_suite(
                n_samples=config.get("integration_samples", 5)
            )
            results["integration_tests"] = integration_results
            
        # Calculate summary metrics
        summary = self._calculate_summary(results)
        results["summary"] = summary
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
            
        # Generate visualizations
        if config.get("generate_visualizations", True):
            self._generate_visualizations(results, f"{self.results_dir}/visualizations_{timestamp}")
            
        return results, filename
        
    def _default_test_config(self):
        """Default configuration for automated tests"""
        return {
            "run_ocr_tests": True,
            "run_feedback_tests": True,
            "run_integration_tests": True,
            "ocr_samples": 10,
            "feedback_samples": 20,
            "integration_samples": 5,
            "ocr_doc_types": ["handwritten", "printed", "mixed"],
            "error_types": ["calculation", "procedural", "conceptual"],
            "generate_visualizations": True
        }
        
    def _make_serializable(self, results):
        """Convert results to JSON serializable format"""
        if isinstance(results, dict):
            return {k: self._make_serializable(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._make_serializable(item) for item in results]
        elif isinstance(results, (TestResult, TestCase)):
            # Convert dataclass to dict
            result_dict = results.__dict__.copy()
            # Handle nested non-serializable objects
            return self._make_serializable(result_dict)
        elif isinstance(results, datetime.datetime):
            return results.isoformat()
        else:
            return results
    
    def _calculate_summary(self, results):
        """Calculate summary metrics from test results"""
        summary = {}
        
        # OCR summary
        ocr_results = results.get("ocr_tests", [])
        if ocr_results:
            ocr_passed = sum(1 for r in ocr_results if r.passed)
            summary["ocr"] = {
                "total": len(ocr_results),
                "passed": ocr_passed,
                "pass_rate": ocr_passed / len(ocr_results) if ocr_results else 0,
                "accuracy": np.mean([r.performance_metrics.get("accuracy", 0) for r in ocr_results if r.performance_metrics])
            }
            
        # Feedback summary
        feedback_results = results.get("feedback_tests", [])
        if feedback_results:
            feedback_passed = sum(1 for r in feedback_results if r.passed)
            summary["feedback"] = {
                "total": len(feedback_results),
                "passed": feedback_passed,
                "pass_rate": feedback_passed / len(feedback_results) if feedback_results else 0
            }
            
        # Integration summary
        integration_results = results.get("integration_tests", [])
        if integration_results:
            integration_passed = sum(1 for r in integration_results if r.passed)
            summary["integration"] = {
                "total": len(integration_results),
                "passed": integration_passed,
                "pass_rate": integration_passed / len(integration_results) if integration_results else 0,
                "end_to_end_accuracy": np.mean([r.performance_metrics.get("end_to_end_accuracy", 0) 
                                               for r in integration_results if r.performance_metrics])
            }
            
        # Overall summary
        all_tests = ocr_results + feedback_results + integration_results
        all_passed = sum(1 for r in all_tests if r.passed)
        summary["overall"] = {
            "total": len(all_tests),
            "passed": all_passed,
            "pass_rate": all_passed / len(all_tests) if all_tests else 0
        }
        
        return summary
    
    def _generate_visualizations(self, results, output_dir):
        """Generate visualizations of test results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Overall pass rates pie chart
        summary = results["summary"]
        if "overall" in summary:
            passed = summary["overall"]["passed"]
            failed = summary["overall"]["total"] - passed
            
            plt.figure(figsize=(8, 6))
            plt.pie([passed, failed], labels=['Passed', 'Failed'], autopct='%1.1f%%', 
                   colors=['#4CAF50', '#F44336'], startangle=90)
            plt.title('Overall Test Results')
            plt.savefig(f"{output_dir}/overall_results_pie.png")
            plt.close()
            
        # Component-wise bar chart
        components = ['ocr', 'feedback', 'integration']
        pass_rates = []
        
        for component in components:
            if component in summary:
                pass_rates.append(summary[component].get("pass_rate", 0) * 100)
            else:
                pass_rates.append(0)
                
        plt.figure(figsize=(10, 6))
        bars = plt.bar(components, pass_rates, color=['#2196F3', '#FF9800', '#9C27B0'])
        
        # Add percentage labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
                   
        plt.ylim(0, 105)  # Leave room for text
        plt.title('Pass Rate by Component')
        plt.ylabel('Pass Rate (%)')
        plt.savefig(f"{output_dir}/component_pass_rates.png")
        plt.close()
        
        # If OCR tests were run, generate OCR-specific visualizations
        ocr_results = results.get("ocr_tests", [])
        if ocr_results:
            # OCR accuracy by document type
            doc_types = {}
            for result in ocr_results:
                if result.metadata and 'doc_type' in result.metadata:
                    doc_type = result.metadata['doc_type']
                    if doc_type not in doc_types:
                        doc_types[doc_type] = []
                    doc_types[doc_type].append(
                        result.performance_metrics.get("accuracy", 0) if result.performance_metrics else 0
                    )
            
            if doc_types:
                plt.figure(figsize=(10, 6))
                data = [(key, np.mean(values) * 100) for key, values in doc_types.items()]
                types, accuracies = zip(*data)
                
                bars = plt.bar(types, accuracies, color=['#CDDC39', '#00BCD4', '#E91E63'])
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
                           
                plt.ylim(0, 105)
                plt.title('OCR Accuracy by Document Type')
                plt.ylabel('Accuracy (%)')
                plt.savefig(f"{output_dir}/ocr_accuracy_by_type.png")
                plt.close()
        
        # Generate feedback-specific visualizations if available
        # This is just a placeholder for additional visualizations
        
    def create_test_report(self, results, include_visualizations=True):
        """Create a comprehensive HTML test report"""
        # This is a placeholder for report generation
        # In a complete implementation, this would generate an HTML report
        # with detailed test results and visualizations
        pass

class OCRTester:
    """Testing framework specific to OCR components"""
    
    def __init__(self):
        self.ocr_systems = {
            "handwritten": MathOCRFactory.create_processor("handwritten"),
            "printed": MathOCRFactory.create_processor("printed"),
            "mixed": MathOCRFactory.create_processor("mixed")
        }
        
    def run_test_suite(self, n_samples=10, doc_types=None):
        """Run a full test suite for OCR functionality"""
        doc_types = doc_types or ["handwritten", "printed", "mixed"]
        results = []
        
        # Create test cases
        test_cases = self._generate_test_cases(n_samples, doc_types)
        
        # Run each test case
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
            
        return results
        
    def _generate_test_cases(self, n_samples, doc_types):
        """Generate synthetic test cases for OCR testing"""
        test_cases = []
        
        # For demonstration purposes, we'll create synthetic test cases
        # In a real implementation, this would use actual images with ground truth
        
        # Simple expressions for testing
        expressions = [
            "x^2 + 3x - 2 = 0",
            "\\frac{a+b}{c} = d",
            "3 + 4 = 7",
            "\\sqrt{16} = 4",
            "\\int_0^1 x^2 dx = \\frac{1}{3}",
            "\\sum_{i=1}^n i = \\frac{n(n+1)}{2}",
            "ax^2 + bx + c = 0",
            "E = mc^2",
            "\\cos^2 \\theta + \\sin^2 \\theta = 1",
            "f'(x) = 2x"
        ]
        
        case_id = 1
        for doc_type in doc_types:
            for _ in range(n_samples // len(doc_types) + 1):
                if len(test_cases) >= n_samples:
                    break
                    
                # Select a random expression
                expr = random.choice(expressions)
                
                # For a real test, this would be an actual image
                # For this example, we're using a placeholder
                placeholder_image = np.ones((100, 300), dtype=np.uint8) * 255
                cv2.putText(placeholder_image, expr.replace("\\", ""), (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                           
                test_case = TestCase(
                    id=f"ocr_{case_id}",
                    type="ocr",
                    input_data=placeholder_image,
                    expected_output=expr,
                    metadata={"doc_type": doc_type}
                )
                
                test_cases.append(test_case)
                case_id += 1
                
        return test_cases
        
    def run_single_test(self, test_case):
        """Run a single OCR test case"""
        try:
            # Get the appropriate OCR processor
            doc_type = test_case.metadata.get("doc_type", "mixed")
            ocr_system = self.ocr_systems[doc_type]
            
            # Process the image
            actual_output = ocr_system.extract_math_from_image(
                test_case.input_data, return_format="latex")
                
            # Calculate string similarity between expected and actual
            similarity = self._calculate_string_similarity(
                test_case.expected_output, actual_output)
                
            # Test passes if similarity is above threshold
            passed = similarity > 0.7
            
            # Performance metrics
            metrics = {
                "similarity": similarity,
                "accuracy": similarity,
                "expected_length": len(test_case.expected_output),
                "actual_length": len(actual_output) if actual_output else 0
            }
            
            return TestResult(
                test_case_id=test_case.id,
                passed=passed,
                actual_output=actual_output,
                performance_metrics=metrics
            )
            
        except Exception as e:
            return TestResult(
                test_case_id=test_case.id,
                passed=False,
                actual_output=None,
                error_message=str(e)
            )
            
    def _calculate_string_similarity(self, s1, s2):
        """Calculate similarity between two strings"""
        # Very simple implementation - in reality, you'd want something more sophisticated
        # that understands LaTeX equivalence
        if not s1 or not s2:
            return 0.0
            
        # Normalize strings
        s1 = s1.replace(" ", "").lower()
        s2 = s2.replace(" ", "").lower()
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        
        # Return similarity ratio
        return matches / max(len(s1), len(s2))
        
class FeedbackTester:
    """Testing framework specific to feedback generation components"""
    
    def __init__(self):
        self.feedback_generator = MathFeedbackGenerator()
        self.problem_generator = MathProblemGenerator()
        
    def run_test_suite(self, n_samples=20, error_types=None):
        """Run a comprehensive test suite for feedback generation"""
        error_types = error_types or ["calculation", "procedural", "conceptual"]
        results = []
        
        # Create test cases
        test_cases = self._generate_test_cases(n_samples, error_types)
        
        # Run each test case
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
            
        return results
        
    def _generate_test_cases(self, n_samples, error_types):
        """Generate test cases for feedback testing"""
        test_cases = []
        
        # Map error types to internal error types
        error_type_map = {
            "calculation": "CALCULATION",
            "procedural": "PROCEDURAL",
            "conceptual": "CONCEPTUAL"
        }
        
        case_id = 1
        for error_type in error_types:
            for _ in range(n_samples // len(error_types) + 1):
                if len(test_cases) >= n_samples:
                    break
                    
                # Generate a problem and solution
                problem_type = random.choice([
                    "algebra_linear", "algebra_quadratic", 
                    "arithmetic_basic", "arithmetic_fractions"
                ])
                problem, correct_answer = self.problem_generator.generate_problem(
                    problem_type, "medium")
                    
                # Generate an incorrect answer based on error type
                _, incorrect_answer, error_desc = self.problem_generator.generate_incorrect_solution(
                    problem_type, problem, correct_answer)
                    
                # Format as analysis result
                analysis = {
                    "question": problem,
                    "student_answer": str(incorrect_answer),
                    "correct_answer": str(correct_answer),
                    "score": random.randint(0, 70),
                    "errors": [{
                        "type": error_type_map.get(error_type, "CALCULATION"),
                        "description": error_desc
                    }]
                }
                
                # Expected output should have certain key qualities
                expected_qualities = {
                    "has_problem_statement": True,
                    "has_error_identification": True,
                    "has_correct_answer": True,
                    "has_practice_problems": True
                }
                
                test_case = TestCase(
                    id=f"feedback_{case_id}",
                    type="feedback",
                    input_data=analysis,
                    expected_output=expected_qualities,
                    metadata={"error_type": error_type}
                )
                
                test_cases.append(test_case)
                case_id += 1
                
        return test_cases
        
    def run_single_test(self, test_case):
        """Run a single feedback generator test"""
        try:
            # Generate feedback using the feedback generator
            actual_output = self.feedback_generator.generate_feedback(test_case.input_data)
            
            # Check for expected qualities in the feedback
            expected_qualities = test_case.expected_output
            results = {}
            
            # Check for problem statement
            has_problem = test_case.input_data["question"] in actual_output.get("feedback", "")
            results["has_problem_statement"] = has_problem
            
            # Check for error identification
            error_desc = test_case.input_data["errors"][0]["description"]
            has_error_id = "error" in actual_output.get("feedback", "").lower()
            results["has_error_identification"] = has_error_id
            
            # Check for correct answer
            correct_answer = test_case.input_data["correct_answer"]
            has_correct = correct_answer in actual_output.get("feedback", "")
            results["has_correct_answer"] = has_correct
            
            # Check for practice problems
            has_practice = (
                "practice" in actual_output.get("feedback", "").lower() or
                len(actual_output.get("practice_problems", [])) > 0
            )
            results["has_practice_problems"] = has_practice
            
            # Test passes if all expected qualities are present
            all_present = all(results.values())
            
            return TestResult(
                test_case_id=test_case.id,
                passed=all_present,
                actual_output=actual_output,
                performance_metrics=results
            )
            
        except Exception as e:
            return TestResult(
                test_case_id=test_case.id,
                passed=False,
                actual_output=None,
                error_message=str(e)
            )
            
class IntegrationTester:
    """Testing framework for end-to-end integration tests"""
    
    def __init__(self):
        self.ocr_system = MathOCRFactory.create_processor("mixed")
        self.feedback_generator = MathFeedbackGenerator()
        
    def run_test_suite(self, n_samples=5):
        """Run comprehensive end-to-end integration tests"""
        results = []
        
        # Create test cases
        test_cases = self._generate_test_cases(n_samples)
        
        # Run each test case
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
            
        return results
        
    def _generate_test_cases(self, n_samples):
        """Generate end-to-end test cases"""
        # This would normally include actual images with ground truth
        # For demonstration, we'll use placeholder data
        
        test_cases = []
        
        # Sample problems and solutions
        sample_problems = [
            ("2x + 3 = 7", "x = 2", "x = 2"),
            ("x^2 - 4 = 0", "x = 2, x = -2", "x = 2, x = -2"),
            ("3 + 4 = ?", "7", "7")
        ]
        
        for i in range(n_samples):
            # Select a random problem
            problem, correct, student = random.choice(sample_problems)
            
            # For a real test, this would be an actual image
            placeholder_image = np.ones((100, 300), dtype=np.uint8) * 255
            cv2.putText(placeholder_image, problem, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                       
            # Sometimes make the student answer incorrect
            if random.random() < 0.5:
                if "=" in student:
                    parts = student.split("=")
                    student = f"{parts[0]}= {float(parts[1].strip()) + 1}"
                else:
                    student = str(int(student) + 1)
                    
            # Create the complete expected results pipeline
            expected_ocr = problem
            expected_analysis = {
                "question": problem,
                "student_answer": student,
                "correct_answer": correct,
                "score": 100 if student == correct else 0,
                "errors": [] if student == correct else [{"type": "CALCULATION", "description": "Calculation error"}]
            }
            
            expected_feedback_qualities = {
                "has_problem_statement": True,
                "has_error_identification": student != correct,
                "has_correct_answer": True
            }
            
            test_case = TestCase(
                id=f"integration_{i+1}",
                type="integration",
                input_data=placeholder_image,
                expected_output={
                    "ocr": expected_ocr,
                    "analysis": expected_analysis,
                    "feedback": expected_feedback_qualities
                },
                metadata={"has_error": student != correct}
            )
            
            test_cases.append(test_case)
            
        return test_cases
        
    def run_single_test(self, test_case):
        """Run a single end-to-end integration test"""
        try:
            # Step 1: OCR processing
            ocr_result = self.ocr_system.extract_math_from_image(
                test_case.input_data, return_format="text")
                
            # Step 2: Analysis (simplified for this example)
            # In a real system, this would parse the OCR output and perform analysis
            expected_analysis = test_case.expected_output["analysis"]
            analysis = {
                "question": ocr_result,
                "student_answer": expected_analysis["student_answer"],
                "correct_answer": expected_analysis["correct_answer"],
                "score": expected_analysis["score"],
                "errors": expected_analysis["errors"]
            }
            
            # Step 3: Feedback generation
            feedback = self.feedback_generator.generate_feedback(analysis)
            
            # Check results against expected output
            ocr_similarity = self._calculate_string_similarity(
                test_case.expected_output["ocr"], ocr_result)
                
            # Check feedback qualities
            expected_feedback = test_case.expected_output["feedback"]
            feedback_results = {}
            
            feedback_text = feedback.get("feedback", "")
            
            # Check for problem statement
            has_problem = analysis["question"] in feedback_text
            feedback_results["has_problem_statement"] = has_problem
            
            # Check for error identification
            has_error_id = (
                not expected_feedback["has_error_identification"] or 
                "error" in feedback_text.lower()
            )
            feedback_results["has_error_identification"] = has_error_id
            
            # Check for correct answer
            has_correct = analysis["correct_answer"] in feedback_text
            feedback_results["has_correct_answer"] = has_correct
            
            # Calculate overall success
            ocr_success = ocr_similarity > 0.7
            feedback_success = all(feedback_results.values())
            overall_success = ocr_success and feedback_success
            
            # Calculate metrics
            metrics = {
                "ocr_accuracy": ocr_similarity,
                "feedback_quality": sum(feedback_results.values()) / len(feedback_results),
                "end_to_end_accuracy": 0.5 * ocr_similarity + 0.5 * (sum(feedback_results.values()) / len(feedback_results))
            }
            
            return TestResult(
                test_case_id=test_case.id,
                passed=overall_success,
                actual_output={
                    "ocr": ocr_result,
                    "analysis": analysis,
                    "feedback": feedback
                },
                performance_metrics=metrics
            )
            
        except Exception as e:
            return TestResult(
                test_case_id=test_case.id,
                passed=False,
                actual_output=None,
                error_message=str(e)
            )
            
    def _calculate_string_similarity(self, s1, s2):
        """Calculate similarity between two strings"""
        if not s1 or not s2:
            return 0.0
            
        # Normalize
        s1 = s1.replace(" ", "").lower()
        s2 = s2.replace(" ", "").lower()
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        
        # Return similarity ratio
        return matches / max(len(s1), len(s2))

# Run all tests function for easy execution
def run_all_tests(output_dir="results/tests"):
    """Run all test suites and generate reports"""
    tester = MathFeedbackTester(results_dir=output_dir)
    results, filename = tester.run_automated_test_suite()
    
    print(f"Test results saved to {filename}")
    print(f"\nSummary:")
    
    summary = results["summary"]
    if "overall" in summary:
        total = summary["overall"]["total"]
        passed = summary["overall"]["passed"]
        pass_rate = summary["overall"]["pass_rate"] * 100
        print(f"Overall: {passed}/{total} tests passed ({pass_rate:.1f}%)")
        
    components = ["ocr", "feedback", "integration"]
    for component in components:
        if component in summary:
            total = summary[component]["total"]
            passed = summary[component]["passed"]
            pass_rate = summary[component]["pass_rate"] * 100
            print(f"{component.capitalize()}: {passed}/{total} tests passed ({pass_rate:.1f}%)")
            
    return results

# Example usage:
# results = run_all_tests()
