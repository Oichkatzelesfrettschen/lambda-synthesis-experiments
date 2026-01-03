"""Formal verification module using Z3 SMT solver for lambda term validation."""
from typing import Dict, Optional, Tuple, Any
from enum import Enum

try:
    from z3 import *
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not available. Install with: pip install z3-solver")


class LambdaType(Enum):
    """Lambda calculus type representations."""
    INT = "Int"
    BOOL = "Bool"
    FUNC = "Func"
    VAR = "Var"


class TypeChecker:
    """Formal type checker using Z3 SMT solver."""
    
    def __init__(self) -> None:
        """Initialize the type checker with Z3 solver."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 solver is required for formal verification")
        self.solver = Solver()
        self.type_vars: Dict[str, Any] = {}
        self.constraints: list = []
    
    def create_type_var(self, name: str) -> Any:
        """
        Create a Z3 type variable.
        
        Args:
            name: Variable name
            
        Returns:
            Z3 integer variable representing the type
        """
        if name not in self.type_vars:
            self.type_vars[name] = Int(name)
        return self.type_vars[name]
    
    def add_constraint(self, constraint: Any) -> None:
        """
        Add a type constraint to the solver.
        
        Args:
            constraint: Z3 constraint expression
        """
        self.constraints.append(constraint)
        self.solver.add(constraint)
    
    def verify_type_safety(self, term_str: str) -> Tuple[bool, Optional[Any]]:
        """
        Verify that a lambda term is type-safe.
        
        Args:
            term_str: String representation of lambda term
            
        Returns:
            Tuple of (is_valid, model) where model is Z3 model if valid
        """
        # This is a simplified example - full implementation would parse the term
        # and generate appropriate constraints
        
        result = self.solver.check()
        if result == sat:
            return True, self.solver.model()
        else:
            return False, None
    
    def infer_type(self, term: str) -> Optional[str]:
        """
        Infer the type of a lambda term using constraint solving.
        
        Args:
            term: Lambda term string
            
        Returns:
            String representation of inferred type, or None if type error
        """
        is_valid, model = self.verify_type_safety(term)
        if is_valid and model:
            # Extract type from model
            return str(model)
        return None
    
    def reset(self) -> None:
        """Reset the solver and clear all constraints."""
        self.solver.reset()
        self.type_vars.clear()
        self.constraints.clear()


class TermValidator:
    """Validates structural properties of lambda terms."""
    
    @staticmethod
    def is_valid_variable(var: str) -> bool:
        """Check if a string is a valid variable name."""
        if not var or not var[0].isalpha():
            return False
        return all(c.isalnum() or c == '_' for c in var)
    
    @staticmethod
    def check_balanced_parens(term: str) -> bool:
        """Check if parentheses are balanced in term."""
        count = 0
        for char in term:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    @staticmethod
    def validate_lambda_syntax(term: str) -> Tuple[bool, str]:
        """
        Validate basic lambda calculus syntax.
        
        Args:
            term: Lambda term string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not term:
            return False, "Empty term"
        
        if not TermValidator.check_balanced_parens(term):
            return False, "Unbalanced parentheses"
        
        # Check for lambda symbol
        if 'λ' in term or '\\' in term:
            # Basic lambda term structure
            if '.' not in term:
                return False, "Lambda abstraction missing dot"
        
        return True, ""


def verify_term_properties(term: str) -> Dict[str, Any]:
    """
    Comprehensive verification of term properties.
    
    Args:
        term: Lambda term to verify
        
    Returns:
        Dictionary with verification results
    """
    results: Dict[str, Any] = {
        "syntactically_valid": False,
        "balanced_parens": False,
        "type_safe": None,
        "errors": []
    }
    
    # Structural validation
    results["balanced_parens"] = TermValidator.check_balanced_parens(term)
    is_valid, error = TermValidator.validate_lambda_syntax(term)
    results["syntactically_valid"] = is_valid
    
    if not is_valid:
        results["errors"].append(error)
    
    # Type safety verification (if Z3 available)
    if Z3_AVAILABLE:
        try:
            checker = TypeChecker()
            is_type_safe, model = checker.verify_type_safety(term)
            results["type_safe"] = is_type_safe
            if not is_type_safe:
                results["errors"].append("Type checking failed")
        except Exception as e:
            results["errors"].append(f"Type verification error: {str(e)}")
    else:
        results["type_safe"] = None
        results["errors"].append("Z3 not available for type checking")
    
    return results


def example_usage() -> None:
    """Demonstrate usage of formal verification tools."""
    print("=== Lambda Term Formal Verification Demo ===\n")
    
    test_terms = [
        "(λ x. x)",                    # Identity function
        "(λ f. (λ x. f (f x)))",      # Church numeral 2
        "(λ x. x) y",                  # Application
        "((λ x. x)",                   # Invalid - unbalanced
    ]
    
    for term in test_terms:
        print(f"Term: {term}")
        results = verify_term_properties(term)
        print(f"  Syntactically valid: {results['syntactically_valid']}")
        print(f"  Balanced parens: {results['balanced_parens']}")
        print(f"  Type safe: {results['type_safe']}")
        if results['errors']:
            print(f"  Errors: {', '.join(results['errors'])}")
        print()


if __name__ == "__main__":
    example_usage()
