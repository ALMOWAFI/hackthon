import random
import numpy as np
from collections import defaultdict

class TeachingPerspectives:
    """
    Advanced teaching perspectives module that generates pedagogically sound
    feedback using different instructional approaches and cognitive frameworks.
    """
    
    # Teaching styles and their characteristics
    TEACHING_STYLES = {
        'SOCRATIC': {
            'description': 'Uses questions to guide student discovery',
            'prompt_types': ['inquiry', 'reflection', 'extension'],
            'tone': 'curious',
            'suitable_for': ['conceptual_errors', 'logical_errors']
        },
        'CONSTRUCTIVIST': {
            'description': 'Builds on student\'s existing knowledge',
            'prompt_types': ['connection', 'scaffolding', 'pattern_recognition'],
            'tone': 'supportive',
            'suitable_for': ['all_error_types']
        },
        'DIRECT_INSTRUCTION': {
            'description': 'Clear, explicit explanation of concepts',
            'prompt_types': ['instruction', 'clarification', 'examples'],
            'tone': 'authoritative',
            'suitable_for': ['procedural_errors', 'calculation_errors']
        },
        'GROWTH_MINDSET': {
            'description': 'Emphasizes effort and improvement',
            'prompt_types': ['encouragement', 'strategy', 'progress'],
            'tone': 'encouraging',
            'suitable_for': ['all_error_types']
        },
        'METACOGNITIVE': {
            'description': 'Focuses on thinking about thinking',
            'prompt_types': ['reflection', 'strategy_evaluation', 'self_monitoring'],
            'tone': 'analytical',
            'suitable_for': ['procedural_errors', 'conceptual_errors']
        }
    }
    
    # Mathematical concept principles (MCPs) - advanced mathematical ideas
    MATH_CONCEPT_PRINCIPLES = {
        'NUMBER_SENSE': {
            'concepts': [
                'Place value', 'Number composition', 'Estimation', 
                'Number relationships', 'Properties of operations'
            ],
            'misconceptions': [
                'Confusing numerals with quantity',
                'Believing larger digits always mean larger numbers',
                'Ignoring place value in calculations',
                'Assuming operations have the same properties'
            ],
            'remediations': [
                'Use visual representations to illustrate quantity',
                'Practice decomposing numbers in multiple ways',
                'Compare quantities using different models',
                'Explore when properties apply and don\'t apply'
            ]
        },
        'ALGEBRAIC_THINKING': {
            'concepts': [
                'Variable relationships', 'Function behaviors', 'Equation properties',
                'Pattern recognition', 'Algebraic structures'
            ],
            'misconceptions': [
                'Treating equal sign as an operation instead of equivalence',
                'Applying arithmetic rules incorrectly to algebraic expressions',
                'Misunderstanding the nature of variables',
                'Overgeneralizing patterns'
            ],
            'remediations': [
                'Use balance models to demonstrate equation properties',
                'Connect algebraic expressions to real situations',
                'Compare expressions across different representations',
                'Explore pattern breakdowns and exceptions'
            ]
        },
        'GEOMETRIC_REASONING': {
            'concepts': [
                'Spatial relationships', 'Transformations', 'Coordinate systems',
                'Geometric properties', 'Dimensional analysis'
            ],
            'misconceptions': [
                'Confusing area and perimeter',
                'Believing measurements scale linearly across dimensions',
                'Overgeneralizing properties from specific cases',
                'Misapplying formulas without conceptual understanding'
            ],
            'remediations': [
                'Use dynamic visualizations to explore properties',
                'Compare measurements across different shapes',
                'Investigate counterexamples to test conjectures',
                'Connect formal definitions to visual representations'
            ]
        },
        'MEASUREMENT_LOGIC': {
            'concepts': [
                'Unit relationships', 'Precision', 'Proportional reasoning',
                'Conversion principles', 'Measurement systems'
            ],
            'misconceptions': [
                'Confusion between different units',
                'Applying additive thinking to multiplicative situations',
                'Ignoring the need for common units',
                'Misunderstanding significant figures'
            ],
            'remediations': [
                'Use physical models to demonstrate unit relationships',
                'Practice scaling up and down with different factors',
                'Emphasize unit analysis in all problems',
                'Compare precision across measurement tools'
            ]
        },
        'DATA_REASONING': {
            'concepts': [
                'Distribution properties', 'Statistical inference', 'Probability models',
                'Data representation', 'Uncertainty analysis'
            ],
            'misconceptions': [
                'Confusing correlation with causation',
                'Ignoring sample size and representativeness',
                'Misinterpreting probability statements',
                'Focusing on individual data points instead of distributions'
            ],
            'remediations': [
                'Analyze multiple data representations of the same information',
                'Compare outcomes across different sample sizes',
                'Use simulation to explore probability concepts',
                'Critique misleading data representations'
            ]
        },
        'CALCULUS_THINKING': {
            'concepts': [
                'Rate of change', 'Accumulation', 'Limit behavior',
                'Infinite processes', 'Approximation techniques'
            ],
            'misconceptions': [
                'Treating derivatives as fractions',
                'Confusing instantaneous and average rates',
                'Believing infinity is a number',
                'Misunderstanding the nature of asymptotic behavior'
            ],
            'remediations': [
                'Connect graphical, numerical, and symbolic representations',
                'Use dynamic models to visualize changing rates',
                'Compare finite and infinite processes',
                'Explore error bounds in approximations'
            ]
        },
        'MATHEMATICAL_LOGIC': {
            'concepts': [
                'Proof techniques', 'Logical quantifiers', 'Contraposition',
                'Logical equivalence', 'Axiomatic systems'
            ],
            'misconceptions': [
                'Mistaking examples for proof',
                'Confusing necessary and sufficient conditions',
                'Misunderstanding negation of complex statements',
                'Applying proof techniques incorrectly'
            ],
            'remediations': [
                'Practice translating between logical forms',
                'Analyze the structure of mathematical arguments',
                'Explore the logical connections between concepts',
                'Construct counterexamples to test validity'
            ]
        },
        'ABSTRACT_ALGEBRA': {
            'concepts': [
                'Group properties', 'Ring structures', 'Field extensions',
                'Isomorphism', 'Abstract structures'
            ],
            'misconceptions': [
                'Assuming all algebraic structures behave like familiar number systems',
                'Overgeneralizing properties from specific examples',
                'Misunderstanding the significance of axioms',
                'Confusing different types of algebraic structures'
            ],
            'remediations': [
                'Use concrete examples of different algebraic structures',
                'Compare and contrast familiar and unfamiliar systems',
                'Trace how properties depend on axiomatic foundations',
                'Visualize abstract structures using diagrams'
            ]
        }
    }
    
    # Advanced pedagogical frameworks
    PEDAGOGICAL_FRAMEWORKS = {
        'COGNITIVE_LOAD_THEORY': {
            'description': 'Optimizes learning by managing mental effort',
            'strategies': [
                'Chunk complex information into manageable parts',
                'Use worked examples for complex procedures',
                'Remove extraneous details that distract',
                'Scaffold difficult concepts with supportive structures'
            ]
        },
        'SPACED_PRACTICE': {
            'description': 'Distributes learning over time to enhance retention',
            'strategies': [
                'Revisit key concepts over increasing intervals',
                'Interleave different types of problems',
                'Connect new learning to previously mastered concepts',
                'Build cumulative review into instruction'
            ]
        },
        'PRODUCTIVE_STRUGGLE': {
            'description': 'Encourages persistence through challenging problems',
            'strategies': [
                'Pose problems slightly beyond current mastery',
                'Allow time for exploration before intervention',
                'Validate effort even when solutions are incorrect',
                'Guide with questions rather than direct answers'
            ]
        },
        'CONCEPTUAL_CHANGE': {
            'description': 'Addresses and transforms misconceptions',
            'strategies': [
                'Elicit existing understanding before instruction',
                'Create cognitive conflict to challenge misconceptions',
                'Provide evidence that contradicts faulty ideas',
                'Build bridges between intuitive understanding and formal concepts'
            ]
        },
        'DIFFERENTIATED_INSTRUCTION': {
            'description': 'Tailors teaching to individual learning needs',
            'strategies': [
                'Provide multiple representations of concepts',
                'Offer tiered assignments with varying complexity',
                'Allow different paths to demonstrate understanding',
                'Adjust pace based on student readiness'
            ]
        }
    }
    
    def __init__(self):
        """Initialize the teaching perspectives module"""
        # Track student performance and patterns
        self.student_history = defaultdict(list)
        self.learning_stage = "NOVICE"  # NOVICE, DEVELOPING, PROFICIENT, ADVANCED
        
    def classify_error(self, error_type, error_description):
        """Classify errors into cognitive categories for tailored responses"""
        cognitive_categories = {
            'CALCULATION': ['arithmetic_error', 'computational_mistake', 'numerical_error'],
            'PROCEDURAL': ['algorithm_error', 'sequence_error', 'operation_order'],
            'CONCEPTUAL': ['misunderstanding', 'misconception', 'false_belief'],
            'LOGICAL': ['invalid_reasoning', 'false_premise', 'illogical_conclusion'],
            'REPRESENTATION': ['notation_error', 'symbol_misuse', 'format_error']
        }
        
        # Map error types to cognitive categories
        for category, descriptors in cognitive_categories.items():
            if any(descriptor in error_description.lower() for descriptor in descriptors):
                return category
                
        # Default mapping based on error_type
        error_map = {
            'CALCULATION': 'CALCULATION',
            'PROCEDURAL': 'PROCEDURAL',
            'CONCEPTUAL': 'CONCEPTUAL'
        }
        
        return error_map.get(error_type, 'CONCEPTUAL')
        
    def select_teaching_style(self, error_type, student_history=None):
        """Select appropriate teaching style based on error and student history"""
        # Default teaching styles for each error type
        style_mapping = {
            'CALCULATION': 'DIRECT_INSTRUCTION',
            'PROCEDURAL': 'DIRECT_INSTRUCTION',
            'CONCEPTUAL': 'SOCRATIC',
            'LOGICAL': 'METACOGNITIVE',
            'REPRESENTATION': 'CONSTRUCTIVIST'
        }
        
        # Get default style for this error type
        default_style = style_mapping.get(error_type, 'GROWTH_MINDSET')
        
        # If we have student history, adaptively select style
        if student_history and len(student_history) > 3:
            # If student repeatedly makes same error, try different approaches
            error_counts = defaultdict(int)
            for past_error in student_history:
                error_counts[past_error] += 1
                
            # If this error type is persistent, try a different approach
            if error_counts[error_type] > 2:
                alternate_styles = [s for s in self.TEACHING_STYLES if s != default_style]
                return random.choice(alternate_styles)
        
        return default_style
        
    def identify_relevant_mcp(self, question_text, error_description):
        """Identify which mathematical concept principles are relevant"""
        # Keywords that help identify relevant math domains
        domain_keywords = {
            'NUMBER_SENSE': ['digit', 'place value', 'estimat', 'round', 'number', 'add', 'subtract', 'multiply', 'divide'],
            'ALGEBRAIC_THINKING': ['equation', 'variable', 'expression', 'solve', 'unknown', 'function', 'pattern', 'x', 'y', '='],
            'GEOMETRIC_REASONING': ['shape', 'angle', 'area', 'perimeter', 'volume', 'coordinate', 'triangle', 'circle', 'square'],
            'MEASUREMENT_LOGIC': ['unit', 'measure', 'convert', 'meter', 'gram', 'liter', 'inch', 'pound', 'gallon'],
            'DATA_REASONING': ['data', 'statistic', 'average', 'mean', 'median', 'mode', 'probability', 'likelihood', 'chance'],
            'CALCULUS_THINKING': ['derivative', 'integral', 'rate', 'change', 'limit', 'infinity', 'sequence', 'series'],
            'MATHEMATICAL_LOGIC': ['proof', 'theorem', 'axiom', 'if', 'then', 'implies', 'therefore', 'because', 'conclude'],
            'ABSTRACT_ALGEBRA': ['group', 'ring', 'field', 'isomorphism', 'homomorphism', 'structure', 'module']
        }
        
        # Count keyword matches
        combined_text = (question_text + " " + error_description).lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            domain_scores[domain] = score
            
        # Get domain with highest score
        if any(domain_scores.values()):
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            return best_domain
            
        # Special cases for common math operations
        if '+' in question_text or 'add' in combined_text:
            return 'NUMBER_SENSE'
        if '-' in question_text or 'subtract' in combined_text:
            return 'NUMBER_SENSE'
        if '*' in question_text or '×' in question_text or 'multiply' in combined_text:
            return 'NUMBER_SENSE'
        if '/' in question_text or '÷' in question_text or 'divide' in combined_text:
            return 'NUMBER_SENSE'
        if '=' in question_text:
            return 'ALGEBRAIC_THINKING'
            
        # Default to NUMBER_SENSE as a fallback
        return 'NUMBER_SENSE'
        
    def generate_conceptual_explanation(self, concept_domain, misconception=None):
        """Generate deep conceptual explanation based on identified domain"""
        domain_data = self.MATH_CONCEPT_PRINCIPLES.get(concept_domain)
        if not domain_data:
            return "This concept relates to foundational mathematical principles."
            
        # Get relevant concepts from domain
        concepts = domain_data['concepts']
        
        # Build explanation
        explanation = f"This problem involves {concept_domain.lower().replace('_', ' ')}. "
        explanation += f"Key concepts here include {', '.join(concepts[:2])}. "
        
        # If we identified a misconception, address it
        if misconception:
            # Find most relevant misconception in our database
            domain_misconceptions = domain_data['misconceptions']
            best_match = None
            highest_score = 0
            
            for domain_misconception in domain_misconceptions:
                # Simple word overlap score
                score = sum(word in misconception.lower() for word in domain_misconception.lower().split())
                if score > highest_score:
                    highest_score = score
                    best_match = domain_misconception
                    
            if best_match:
                explanation += f"A common misconception is '{best_match}'. "
                
                # Add relevant remediation
                idx = domain_data['misconceptions'].index(best_match)
                if idx < len(domain_data['remediations']):
                    explanation += domain_data['remediations'][idx]
            else:
                # No good match, use a generic remediation
                explanation += random.choice(domain_data['remediations'])
        else:
            # No specific misconception, provide general insight
            explanation += "Understanding these concepts deeply helps build mathematical fluency."
            
        return explanation
        
    def generate_learning_strategy(self, concept_domain, error_type):
        """Generate specific learning strategies based on domain and error type"""
        # Get relevant pedagogical framework based on error type
        framework_mapping = {
            'CALCULATION': 'SPACED_PRACTICE',
            'PROCEDURAL': 'COGNITIVE_LOAD_THEORY',
            'CONCEPTUAL': 'CONCEPTUAL_CHANGE',
            'LOGICAL': 'PRODUCTIVE_STRUGGLE',
            'REPRESENTATION': 'DIFFERENTIATED_INSTRUCTION'
        }
        
        framework = framework_mapping.get(error_type, 'COGNITIVE_LOAD_THEORY')
        framework_data = self.PEDAGOGICAL_FRAMEWORKS.get(framework)
        
        # Build strategy suggestion
        strategy = f"To improve in this area, try this strategy: "
        strategy += random.choice(framework_data['strategies'])
        
        # Add domain-specific practice suggestion
        domain_data = self.MATH_CONCEPT_PRINCIPLES.get(concept_domain)
        if domain_data and 'remediations' in domain_data:
            strategy += f" Also, {random.choice(domain_data['remediations']).lower()}"
            
        return strategy
        
    def generate_feedback(self, question_text, student_answer, correct_answer, 
                          error_type=None, error_description=None, is_correct=False):
        """Generate pedagogically sound feedback with advanced teaching perspectives"""
        # Handle correct answers
        if is_correct:
            praise_statements = [
                "Excellent work! Your solution demonstrates clear understanding.",
                "Well done! You've correctly applied the mathematical principles.",
                "Perfect! Your answer shows solid mathematical reasoning.",
                "Great job! You've mastered this concept.",
                "Correct! Your approach to this problem was spot-on."
            ]
            return random.choice(praise_statements)
            
        # For incorrect answers, apply our teaching framework
        # Step 1: Classify the error
        if not error_type and error_description:
            error_type = self.classify_error("UNKNOWN", error_description)
        elif not error_type:
            error_type = "CONCEPTUAL"  # Default assumption
            
        # Step 2: Select teaching style
        teaching_style = self.select_teaching_style(error_type, self.student_history)
        style_data = self.TEACHING_STYLES.get(teaching_style)
        
        # Step 3: Identify relevant mathematical concept principles
        concept_domain = self.identify_relevant_mcp(question_text, error_description or "")
        
        # Step 4: Create pedagogically sound response
        feedback = []
        
        # Opening statement based on teaching style tone
        tone = style_data.get('tone', 'supportive')
        if tone == 'curious':
            feedback.append(f"Have you considered why {student_answer} doesn't work here?")
        elif tone == 'supportive':
            feedback.append(f"I see you answered {student_answer}, which gives us a chance to explore this concept.")
        elif tone == 'authoritative':
            feedback.append(f"The answer {student_answer} is incorrect. The correct answer is {correct_answer}.")
        elif tone == 'encouraging':
            feedback.append(f"You're on the right track with {student_answer}, but let's refine our approach.")
        elif tone == 'analytical':
            feedback.append(f"Let's analyze why {student_answer} differs from the correct answer, {correct_answer}.")
        
        # Core explanation
        if error_description:
            feedback.append(f"{error_description}")
        
        # Conceptual explanation based on MCPs
        conceptual_insight = self.generate_conceptual_explanation(concept_domain, error_description)
        feedback.append(conceptual_insight)
        
        # Learning strategy
        strategy = self.generate_learning_strategy(concept_domain, error_type)
        feedback.append(strategy)
        
        # Closing based on teaching style
        prompt_types = style_data.get('prompt_types', ['reflection'])
        prompt_type = random.choice(prompt_types)
        
        if prompt_type == 'inquiry':
            feedback.append(f"What might happen if you tried solving this using {correct_answer} instead?")
        elif prompt_type == 'reflection':
            feedback.append(f"Take a moment to compare your approach with the one that leads to {correct_answer}.")
        elif prompt_type == 'extension':
            feedback.append(f"Now that we know the answer is {correct_answer}, can you think of a similar problem?")
        elif prompt_type == 'connection':
            feedback.append(f"How does this problem connect to other concepts we've learned?")
        elif prompt_type == 'instruction':
            feedback.append(f"Next time, remember that {question_text} should result in {correct_answer}.")
        
        # Update student history
        self.student_history[error_type].append(error_type)
        
        return "\n\n".join(feedback)
        
    def get_wildest_mcps(self):
        """Return the most advanced/challenging mathematical concept principles"""
        advanced_mcps = [
            ('ABSTRACT_ALGEBRA', self.MATH_CONCEPT_PRINCIPLES['ABSTRACT_ALGEBRA']),
            ('MATHEMATICAL_LOGIC', self.MATH_CONCEPT_PRINCIPLES['MATHEMATICAL_LOGIC']),
            ('CALCULUS_THINKING', self.MATH_CONCEPT_PRINCIPLES['CALCULUS_THINKING'])
        ]
        
        results = []
        for domain, data in advanced_mcps:
            results.append({
                'domain': domain,
                'concepts': data['concepts'],
                'misconceptions': data['misconceptions'][0],  # Most common misconception
                'remediation': data['remediations'][0]  # Primary remediation strategy
            })
            
        return results
