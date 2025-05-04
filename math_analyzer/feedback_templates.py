"""
Math feedback templates organized by error type and teaching style.

This module provides structured templates for generating personalized feedback
for different types of mathematical errors, supporting multiple teaching styles.
"""

# Base templates organized by error type and teaching style
ERROR_TEMPLATES = {
    "CALCULATION": {
        "socratic": [
            "What steps did you follow to calculate {expression}?",
            "Can we verify each step of {expression} together?",
            "If we break down {expression}, what do you notice about the calculation?",
            "What would be another way to check if {expression} is correct?"
        ],
        "direct": [
            "There's a calculation error in {expression}. Let's solve it step by step.",
            "The correct calculation for {expression} would give us {correct_result}.",
            "When calculating {expression}, remember to follow the order of operations: parentheses, exponents, multiplication/division, addition/subtraction."
        ],
        "growth_mindset": [
            "This calculation is tricky! Let's work through {expression} together and strengthen your skills.",
            "Making calculation errors is part of learning. Let's see what we can learn from {expression}.",
            "Each calculation mistake is an opportunity to grow. Let's analyze {expression} carefully."
        ],
        "constructivist": [
            "How does {expression} relate to other problems you've solved?",
            "What existing knowledge can you apply to verify {expression}?",
            "Try building a mental model for {expression} using what you already know."
        ],
        "inquiry_based": [
            "What questions come to mind when you look at {expression}?",
            "How could we investigate whether {expression} is correct?",
            "What patterns do you notice in {expression} that might help us?"
        ]
    },
    
    "NOTATION": {
        "socratic": [
            "How else could we write {expression} to make it clearer?",
            "What does the notation in {expression} mean to you?",
            "How would you explain the notation in {expression} to a classmate?"
        ],
        "direct": [
            "The notation in {expression} needs adjustment. The standard way to write this is {correct_notation}.",
            "In mathematics, we need to be precise with our notation. For {expression}, we should write {correct_notation}.",
            "Let's correct the notation in {expression} to follow mathematical conventions."
        ],
        "growth_mindset": [
            "Mastering mathematical notation takes practice. Let's refine {expression} together.",
            "Good attempt with the notation! Let's polish {expression} to make it even clearer.",
            "Your notation is developing well. Let's make a few refinements to {expression}."
        ],
        "constructivist": [
            "How does the notation in {expression} relate to other notation you've used?",
            "What meaning are you trying to convey with {expression}?",
            "Let's explore how we can express your idea in {expression} using standard notation."
        ],
        "inquiry_based": [
            "What different notations could we use for {expression}?",
            "How might different notations for {expression} affect how we understand the problem?",
            "What would happen if we changed the notation in {expression}?"
        ]
    },
    
    "PROCEDURAL": {
        "socratic": [
            "What procedure did you follow to solve {expression}?",
            "Is there a specific order we should follow when solving {expression}?",
            "What steps might we be missing in {expression}?"
        ],
        "direct": [
            "The procedure for solving {expression} should be: {correct_procedure}",
            "Make sure to follow all steps when solving {expression}. You missed: {missing_step}",
            "Let's review the correct procedure for this type of problem."
        ],
        "growth_mindset": [
            "Developing systematic procedures takes practice. Let's refine your approach to {expression}.",
            "You're developing good procedural skills. Let's enhance your strategy for {expression}.",
            "I like how you're approaching this methodically. Let's add one more step to perfect your solution to {expression}."
        ],
        "constructivist": [
            "How does your approach to {expression} compare to other procedures you've learned?",
            "Can you connect this procedure to other math concepts you know?",
            "Let's build on your existing knowledge to develop a robust procedure for {expression}."
        ],
        "inquiry_based": [
            "What if we approached {expression} with a different procedure?",
            "How could we test if our procedure for {expression} is efficient?",
            "What patterns do you notice in the procedure for solving {expression}?"
        ]
    },
    
    "CONCEPTUAL": {
        "socratic": [
            "What does {expression} mean conceptually?",
            "How would you explain the concept behind {expression} to someone else?",
            "What's the underlying idea in {expression}?"
        ],
        "direct": [
            "The concept you need to understand for {expression} is {concept_explanation}",
            "Let's review the key concept needed for {expression}: {concept_explanation}",
            "{expression} involves the concept of {concept_name}, which means {concept_explanation}"
        ],
        "growth_mindset": [
            "Conceptual understanding develops over time. Let's deepen your understanding of {expression}.",
            "You're building good conceptual foundations. Let's explore {expression} more deeply.",
            "I see your conceptual thinking emerging. Let's strengthen your understanding of {expression}."
        ],
        "constructivist": [
            "How does {expression} relate to concepts you already understand?",
            "Let's connect {expression} to real-world situations you're familiar with.",
            "Can you think of a metaphor or analogy for {expression}?"
        ],
        "inquiry_based": [
            "What might be some applications of the concept in {expression}?",
            "How might we investigate the concept behind {expression} further?",
            "What predictions can we make based on our understanding of {expression}?"
        ]
    },
    "DIVISION_BY_ZERO": {
        "socratic": [
            "What happens mathematically when we try to compute {expression}? Why is this operation special?",
            "How does dividing by zero affect the result of {expression}?",
            "What mathematical concept is {expression} based on? How does it apply here?"
        ],
        "direct": [
            "Division by zero as in {expression} is undefined in mathematics. This is a fundamental concept in arithmetic.",
            "When we divide by zero in {expression}, we encounter a mathematical contradiction.",
            "Let's explore why {expression} is undefined in mathematics."
        ],
        "growth_mindset": [
            "Understanding why we can't divide by zero is an important mathematical insight. Let's explore why {expression} presents a challenge.",
            "Developing a deep understanding of mathematical concepts like {expression} takes time and practice.",
            "You're on the right track! Let's refine your understanding of why {expression} is undefined."
        ],
        "constructivist": [
            "Think about division as sharing into equal groups. How would you interpret dividing into zero groups?",
            "How does the concept of division by zero relate to other mathematical concepts you've learned?",
            "Let's build on your existing knowledge to understand why {expression} is undefined."
        ],
        "inquiry_based": [
            "What happens as we divide by smaller and smaller numbers? What does this tell us about {expression}?",
            "How might we investigate the concept of division by zero further?",
            "What mathematical principles are at work in {expression}?"
        ]
    }
}

# Templates for correct answers by teaching style
CORRECT_TEMPLATES = {
    "socratic": [
        "Great work! Can you explain your approach to solving {expression}?",
        "Well done! How did you know which steps to follow for {expression}?",
        "Excellent! What strategy did you use to solve {expression}?"
    ],
    "direct": [
        "Correct! Your solution to {expression} follows the proper procedure.",
        "Well done! {expression} is solved correctly.",
        "Excellent work! Your answer to {expression} is correct."
    ],
    "growth_mindset": [
        "Excellent work! Your persistence in solving {expression} has paid off.",
        "Great job! Your practice with problems like {expression} is showing results.",
        "Well done! You've mastered the skills needed for {expression}."
    ],
    "constructivist": [
        "Well done! How does your solution to {expression} build on what you've learned before?",
        "Great work! How might you apply your approach to {expression} to more complex problems?",
        "Excellent! How does solving {expression} connect to other mathematical concepts?"
    ],
    "inquiry_based": [
        "Correct! What new questions does solving {expression} raise for you?",
        "Well done! What variations of {expression} might be interesting to explore?",
        "Great work! How might we extend {expression} to explore related concepts?"
    ]
}

# Templates for next steps by error type
NEXT_STEPS_TEMPLATES = {
    "CALCULATION": {
        "practice_problems": [
            "Basic arithmetic exercises",
            "Mental math strategies",
            "Calculation with similar numbers",
            "Estimation practice"
        ],
        "concepts_to_review": [
            "Number facts",
            "Order of operations",
            "Place value",
            "Regrouping"
        ],
        "suggested_resources": [
            "Number line practice",
            "Calculation games",
            "Step-by-step calculation guides",
            "Visual calculation aids"
        ]
    },
    
    "NOTATION": {
        "practice_problems": [
            "Writing mathematical expressions",
            "Translating words to symbols",
            "Notation consistency exercises",
            "Symbol recognition practice"
        ],
        "concepts_to_review": [
            "Mathematical notation",
            "Symbolic representation",
            "Mathematical conventions",
            "Mathematical language"
        ],
        "suggested_resources": [
            "Notation guide",
            "Mathematical symbol reference",
            "Practice with notation cards",
            "Interactive notation exercises"
        ]
    },
    
    "PROCEDURAL": {
        "practice_problems": [
            "Step-by-step problem solving",
            "Procedure sequencing exercises",
            "Multi-step problems",
            "Procedural fluency drills"
        ],
        "concepts_to_review": [
            "Problem-solving strategies",
            "Order of operations",
            "Algorithmic thinking",
            "Procedural knowledge"
        ],
        "suggested_resources": [
            "Procedure flowcharts",
            "Step-by-step guides",
            "Procedural checklists",
            "Strategy posters"
        ]
    },
    
    "CONCEPTUAL": {
        "practice_problems": [
            "Conceptual understanding exercises",
            "Application problems",
            "Explain-your-reasoning tasks",
            "Concept mapping activities"
        ],
        "concepts_to_review": [
            "Fundamental concepts",
            "Conceptual foundations",
            "Mathematical principles",
            "Concept relationships"
        ],
        "suggested_resources": [
            "Conceptual math videos",
            "Visual representations",
            "Conceptual explanation guides",
            "Interactive concept explorations"
        ]
    }
}

# Correct answers templates for next steps
CORRECT_NEXT_STEPS = {
    "practice_problems": [
        "Try similar problems with larger numbers",
        "Apply this skill to word problems",
        "Combine this concept with related skills",
        "Create your own problems of this type"
    ],
    "concepts_to_review": [
        "Continue practicing this type of problem",
        "Connect this concept to related ideas",
        "Apply this skill in different contexts",
        "Explore extensions of this concept"
    ],
    "suggested_resources": [
        "Advanced exercises in this topic",
        "Challenge problems",
        "Real-world applications",
        "Related mathematical concepts"
    ]
}

def get_error_feedback(error_type, teaching_style, expression, **kwargs):
    """Get feedback for a specific error type and teaching style."""
    templates = {
        "CALCULATION": {
            "socratic": "If we break down {expression}, what do you notice about the calculation?",
            "direct": "When calculating {expression}, remember to follow the order of operations: parentheses, exponents, multiplication/division, addition/subtraction.",
            "growth_mindset": "Each calculation mistake is an opportunity to grow. Let's analyze {expression} carefully.",
            "constructivist": "Try building a mental model for {expression} using what you already know.",
            "inquiry_based": "What patterns do you notice in {expression} that might help us?"
        },
        "NOTATION": {
            "socratic": "What does the notation in {expression} represent mathematically?",
            "direct": "When writing {expression}, we need to use proper mathematical notation: {correct_notation}",
            "growth_mindset": "Understanding mathematical notation takes practice. Let's improve how we write {expression}.",
            "constructivist": "How does the notation in {expression} relate to the concepts you've learned?",
            "inquiry_based": "How might we express {expression} differently to make its meaning clearer?"
        },
        "PROCEDURAL": {
            "socratic": "What steps are involved in solving {expression}? Which step might need reconsideration?",
            "direct": "When solving {expression}, you need to follow these steps: {correct_procedure}",
            "growth_mindset": "Learning procedures takes practice. Let's review the approach for {expression}.",
            "constructivist": "How do the procedures you know apply to {expression}?",
            "inquiry_based": "What strategy would be most efficient for solving {expression}?"
        },
        "CONCEPTUAL": {
            "socratic": "What mathematical concept is {expression} based on? How does it apply here?",
            "direct": "The concept behind {expression} is: {concept_explanation}",
            "growth_mindset": "Deepening our conceptual understanding helps us solve problems like {expression} more effectively.",
            "constructivist": "How does {expression} connect to math concepts you already understand?",
            "inquiry_based": "What mathematical principles are at work in {expression}?"
        },
        "DIVISION_BY_ZERO": {
            "socratic": "What happens mathematically when we try to compute {expression}? Why is this operation special?",
            "direct": "Division by zero as in {expression} is undefined in mathematics. This is a fundamental concept in arithmetic.",
            "growth_mindset": "Understanding why we can't divide by zero is an important mathematical insight. Let's explore why {expression} presents a challenge.",
            "constructivist": "Think about division as sharing into equal groups. How would you interpret dividing into zero groups?",
            "inquiry_based": "What happens as we divide by smaller and smaller numbers? What does this tell us about {expression}?"
        }
    }
    
    # Fall back to CALCULATION if error type not found
    if error_type not in templates:
        error_type = "CALCULATION"
    
    # Fall back to direct instruction if teaching style not found
    if teaching_style not in templates[error_type]:
        teaching_style = "direct"
    
    # Check for division by zero specifically
    if error_type == "CONCEPTUAL" and "/0" in expression:
        error_type = "DIVISION_BY_ZERO"
    
    # For CONCEPTUAL errors, provide a default concept explanation if none provided
    if error_type == "CONCEPTUAL" and "concept_explanation" not in kwargs:
        kwargs["concept_explanation"] = "understanding the underlying mathematical principles"
    
    template = templates[error_type][teaching_style]
    return template.format(expression=expression, **kwargs)

def get_correct_feedback(teaching_style, expression):
    """Get a feedback template for correct answers."""
    if teaching_style not in CORRECT_TEMPLATES:
        teaching_style = "direct"  # Default to direct style
        
    templates = CORRECT_TEMPLATES[teaching_style]
    return templates[hash(expression) % len(templates)].format(expression=expression)

def get_next_steps(error_types):
    """Get next steps based on error types."""
    practice_problems = []
    concepts_to_review = []
    suggested_resources = []
    
    # If no errors, return templates for correct answers
    if not error_types:
        practice_idx = hash('practice') % len(CORRECT_NEXT_STEPS["practice_problems"])
        concept_idx = hash('concepts') % len(CORRECT_NEXT_STEPS["concepts_to_review"])
        resource_idx = hash('resources') % len(CORRECT_NEXT_STEPS["suggested_resources"])
        
        return {
            "practice_problems": [CORRECT_NEXT_STEPS["practice_problems"][practice_idx]],
            "concepts_to_review": [CORRECT_NEXT_STEPS["concepts_to_review"][concept_idx]],
            "suggested_resources": [CORRECT_NEXT_STEPS["suggested_resources"][resource_idx]]
        }
    
    # Add next steps for each error type
    for error_type in error_types:
        if error_type in NEXT_STEPS_TEMPLATES:
            # Select one item from each category based on error type
            practice_idx = hash(error_type + 'practice') % len(NEXT_STEPS_TEMPLATES[error_type]["practice_problems"])
            concept_idx = hash(error_type + 'concepts') % len(NEXT_STEPS_TEMPLATES[error_type]["concepts_to_review"])
            resource_idx = hash(error_type + 'resources') % len(NEXT_STEPS_TEMPLATES[error_type]["suggested_resources"])
            
            practice_problems.append(NEXT_STEPS_TEMPLATES[error_type]["practice_problems"][practice_idx])
            concepts_to_review.append(NEXT_STEPS_TEMPLATES[error_type]["concepts_to_review"][concept_idx])
            suggested_resources.append(NEXT_STEPS_TEMPLATES[error_type]["suggested_resources"][resource_idx])
    
    return {
        "practice_problems": practice_problems,
        "concepts_to_review": concepts_to_review,
        "suggested_resources": suggested_resources
    }
