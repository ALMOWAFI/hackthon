class ErrorLocalizationEngine:
    # ... (other methods)

    def _check_distribution_error(self, left, right):
        """
        Check for distribution errors (e.g., incorrect application of distributive property)
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Look for expressions like a(b+c) where distribution might be needed
        distribution_pattern = r'(\d+)\(([^)]+)\)'
        left_matches = re.finditer(distribution_pattern, left)
        right_matches = re.finditer(distribution_pattern, right)
        
        # Check left side of equation
        for match in left_matches:
            coefficient = int(match.group(1))
            inner_expr = match.group(2)
            
            # Check if there are terms to distribute (contains + or -)
            if '+' in inner_expr or '-' in inner_expr:
                original_term = match.group(0)  # The entire a(b+c) expression
                
                # Parse inner expression
                inner_terms = []
                # Add + if expression doesn't start with a sign
                if not inner_expr.startswith('+') and not inner_expr.startswith('-'):
                    inner_expr = '+' + inner_expr
                
                # Find all terms with their signs
                term_matches = re.finditer(r'([+\-][^+\-]*)', inner_expr)
                for term_match in term_matches:
                    inner_terms.append(term_match.group(1))
                
                # Create the distributed form
                distributed_terms = []
                for term in inner_terms:
                    sign = term[0]  # + or -
                    term_value = term[1:]
                    
                    # Multiply coefficient by term
                    distributed_terms.append(f"{sign}{coefficient}{term_value}")
                
                distributed_expr = ''.join(distributed_terms)
                if distributed_expr.startswith('+'):
                    distributed_expr = distributed_expr[1:]  # Remove leading +
                
                return (original_term, distributed_expr, 
                       f"When distributing {coefficient} across {inner_expr}, multiply each term by {coefficient}.")
        
        # Check right side with same logic
        for match in right_matches:
            coefficient = int(match.group(1))
            inner_expr = match.group(2)
            
            if '+' in inner_expr or '-' in inner_expr:
                original_term = match.group(0)
                
                # Process similar to above
                inner_terms = []
                if not inner_expr.startswith('+') and not inner_expr.startswith('-'):
                    inner_expr = '+' + inner_expr
                
                term_matches = re.finditer(r'([+\-][^+\-]*)', inner_expr)
                for term_match in term_matches:
                    inner_terms.append(term_match.group(1))
                
                distributed_terms = []
                for term in inner_terms:
                    sign = term[0]
                    term_value = term[1:]
                    distributed_terms.append(f"{sign}{coefficient}{term_value}")
                
                distributed_expr = ''.join(distributed_terms)
                if distributed_expr.startswith('+'):
                    distributed_expr = distributed_expr[1:]
                
                return (original_term, distributed_expr, 
                       f"When distributing {coefficient} across {inner_expr}, multiply each term by {coefficient}.")
        
        return None

    def _check_exponent_error(self, left, right):
        """
        Check for exponent errors (e.g., incorrect application of exponent rules)
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Look for expressions with exponents
        exponent_pattern = r'([a-z0-9]+)\^(\d+)'
        
        # Check for (a*b)^n mistakenly written as a^n*b^n
        product_pattern = r'([a-z0-9]+)\^(\d+)\s*\*\s*([a-z0-9]+)\^(\d+)'
        
        # Check for exponent rule errors on left side
        product_matches = re.finditer(product_pattern, left)
        for match in product_matches:
            base1 = match.group(1)
            exp1 = int(match.group(2))
            base2 = match.group(3)
            exp2 = int(match.group(4))
            
            # Check if this might be from expanding (a*b)^n
            if exp1 == exp2:
                original_term = match.group(0)
                correction = f"({base1}*{base2})^{exp1}"
                explanation = f"If the exponents are the same, this might be (base1*base2)^exponent"
                return (original_term, correction, explanation)
        
        # Check for common exponent errors on terms
        exponent_matches = re.finditer(exponent_pattern, left + right)
        for match in exponent_matches:
            base = match.group(1)
            exponent = int(match.group(2))
            
            # This is a placeholder for more specific exponent error checking
            # In a complete implementation, we'd check for:
            # - (a+b)^2 expanded incorrectly
            # - a^m * a^n = a^(m+n) applied incorrectly
            # - a^m / a^n = a^(m-n) applied incorrectly
            # - (a^m)^n = a^(m*n) applied incorrectly
        
        # Check exponent pattern on right side
        exponent_matches = re.finditer(exponent_pattern, right)
        
        return None

    def _check_substitution_error(self, left, right, variable_values):
        """
        Check for variable substitution errors
        
        Args:
            left, right: Left and right sides of the equation
            variable_values: Dictionary of known variable values
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # If we have variable values to check against
        if variable_values:
            # Look for variables that should be substituted
            for var, value in variable_values.items():
                # Check if the variable appears but hasn't been substituted
                var_pattern = r'\b' + re.escape(var) + r'\b'
                
                if re.search(var_pattern, left) or re.search(var_pattern, right):
                    # Variable is present but may not be substituted correctly
                    # In a complete implementation, we'd verify the substitution
                    
                    # Placeholder for actual checking logic
                    return (var, str(value), 
                           f"The variable {var} should be substituted with its value {value}.")
        
        return None

    def _check_factoring_error(self, left, right):
        """
        Check for factoring errors
        
        Args:
            left, right: Left and right sides of the equation
            
        Returns:
            Tuple of (erroneous term, correction, explanation) or None if no error
        """
        # Look for quadratic expressions that could be factored
        quadratic_pattern = r'([a-z])(\^2|\²)\s*([+\-])\s*(\d+)([a-z])\s*([+\-])\s*(\d+)'
        
        # Check left side for factorable expressions
        quadratic_matches = re.finditer(quadratic_pattern, left)
        for match in quadratic_matches:
            var = match.group(1)  # Variable (e.g., x)
            coef_x = int(match.group(4))  # Coefficient of x
            sign_x = match.group(3)  # Sign before x term
            if sign_x == '-':
                coef_x = -coef_x
                
            constant = int(match.group(7))  # Constant term
            sign_const = match.group(6)  # Sign before constant
            if sign_const == '-':
                constant = -constant
                
            # Check if this is a perfect square trinomial: x² + 2ab + b²
            if coef_x % 2 == 0 and (coef_x // 2)**2 == constant:
                b = coef_x // 2
                original_term = match.group(0)
                correction = f"({var} + {b})²"
                return (original_term, correction, 
                       f"This is a perfect square trinomial that factors to ({var} + {b})²")
            
            # Check for other factorable quadratics
            # In a complete implementation, we would use the quadratic formula
            # to check if the expression can be factored
        
        # Check right side with same logic
        quadratic_matches = re.finditer(quadratic_pattern, right)
        # Similar logic as above
        
        return None

class ErrorType(Enum):
    """Types of mathematical errors"""
    ARITHMETIC = "arithmetic_error"  # Basic calculation mistakes
    ALGEBRAIC = "algebraic_error"    # Issues with algebraic manipulation
    SIGN = "sign_error"              # Mistakes with positive/negative signs
    PROCEDURAL = "procedural_error"  # Wrong steps or order of operations
    CONCEPTUAL = "conceptual_error"  # Fundamental misunderstandings
    MISSING_STEP = "missing_step"    # Skipped necessary steps
    NOTATION = "notation_error"      # Incorrect mathematical notation
    FRACTION = "fraction_error"      # Errors in fraction handling
    EXPONENT = "exponent_error"      # Issues with powers and exponents
    DISTRIBUTION = "distribution_error"  # Errors in distributing terms
    VARIABLE_SUBSTITUTION = "variable_substitution_error"  # Errors in substituting values
    FACTORING = "factoring_error"  # Errors in factoring expressions
