"""Error categorization system for structured failure analysis."""

import ast
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from infrastructure.logging import get_logger

logger = get_logger(__name__)


class ErrorCategory(Enum):
    """Categories of execution errors."""
    
    SYNTAX = "syntax"           # Python syntax errors, invalid code structure
    RUNTIME = "runtime"         # Runtime exceptions, type errors, attribute errors
    LOGIC = "logic"            # Logical errors in analysis approach
    DATA = "data"              # Data-related issues (missing columns, wrong types)
    PERFORMANCE = "performance" # Timeout, memory issues, inefficient operations
    SECURITY = "security"       # Security validation failures
    DEPENDENCY = "dependency"   # Missing imports, library issues
    UNKNOWN = "unknown"         # Unclassified errors


@dataclass
class ErrorAnalysis:
    """Detailed analysis of an error."""
    
    category: ErrorCategory
    confidence: float  # 0.0 to 1.0
    primary_cause: str
    contributing_factors: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    is_recoverable: bool = True
    context_clues: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class CategorizedError:
    """An error with its categorization and analysis."""
    
    original_error: Exception
    error_message: str
    traceback_str: str
    analysis: ErrorAnalysis
    timestamp: float
    execution_context: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies and analyzes execution errors."""
    
    def __init__(self):
        """Initialize the error classifier."""
        self.logger = logger.getChild("ErrorClassifier")
        
        # Patterns for different error types
        self._syntax_patterns = [
            r"SyntaxError",
            r"IndentationError", 
            r"TabError",
            r"invalid syntax",
            r"unexpected EOF",
            r"unmatched",
        ]
        
        self._runtime_patterns = [
            r"NameError",
            r"TypeError", 
            r"AttributeError",
            r"IndexError",
            r"KeyError",
            r"ValueError",
            r"ZeroDivisionError",
            r"ImportError",
            r"ModuleNotFoundError",
        ]
        
        self._data_patterns = [
            r"KeyError.*column",
            r"column.*not found",
            r"DataFrame.*no attribute",
            r"cannot convert",
            r"invalid dtype",
            r"missing.*required.*column",
            r"empty.*DataFrame",
        ]
        
        self._performance_patterns = [
            r"timeout",
            r"memory.*exceeded",
            r"process.*killed",
            r"resource.*limit",
            r"too.*large",
        ]

    def classify_error(
        self, 
        error: Exception, 
        code: str = "", 
        context: Optional[Dict[str, Any]] = None
    ) -> CategorizedError:
        """
        Classify and analyze an error.
        
        Args:
            error: The exception that occurred
            code: The code that caused the error
            context: Execution context (dataframes, variables, etc.)
            
        Returns:
            CategorizedError with detailed analysis
        """
        context = context or {}
        error_message = str(error)
        traceback_str = traceback.format_exc()
        
        self.logger.debug(f"Classifying error: {error_message}")
        
        # Determine error category
        category, confidence = self._categorize_error(error, error_message, traceback_str, code)
        
        # Perform detailed analysis
        analysis = self._analyze_error(
            error, error_message, traceback_str, code, category, confidence, context
        )
        
        return CategorizedError(
            original_error=error,
            error_message=error_message,
            traceback_str=traceback_str,
            analysis=analysis,
            timestamp=__import__('time').time(),
            execution_context=context,
        )

    def _categorize_error(
        self, 
        error: Exception, 
        error_message: str, 
        traceback_str: str,
        code: str
    ) -> Tuple[ErrorCategory, float]:
        """Categorize the error and return confidence score."""
        
        error_type = type(error).__name__
        full_error_text = f"{error_type}: {error_message}\n{traceback_str}"
        
        # Check for syntax errors first (highest confidence)
        if any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self._syntax_patterns):
            return ErrorCategory.SYNTAX, 0.95
            
        # Check for data-specific errors
        if any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self._data_patterns):
            return ErrorCategory.DATA, 0.90
            
        # Check for performance issues
        if any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self._performance_patterns):
            return ErrorCategory.PERFORMANCE, 0.85
            
        # Check for runtime errors
        if any(re.search(pattern, full_error_text, re.IGNORECASE) for pattern in self._runtime_patterns):
            return ErrorCategory.RUNTIME, 0.80
            
        # Check for dependency issues
        if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            return ErrorCategory.DEPENDENCY, 0.90
            
        # Check for security issues (based on validation context)
        if "security" in error_message.lower() or "forbidden" in error_message.lower():
            return ErrorCategory.SECURITY, 0.85
            
        # Default to unknown with low confidence
        return ErrorCategory.UNKNOWN, 0.30

    def _analyze_error(
        self,
        error: Exception,
        error_message: str, 
        traceback_str: str,
        code: str,
        category: ErrorCategory,
        confidence: float,
        context: Dict[str, Any]
    ) -> ErrorAnalysis:
        """Perform detailed error analysis."""
        
        # Extract primary cause
        primary_cause = self._extract_primary_cause(error, error_message, category)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            error, error_message, traceback_str, code, context, category
        )
        
        # Generate suggested fixes
        suggested_fixes = self._generate_suggested_fixes(
            error, error_message, code, category, context
        )
        
        # Determine severity
        severity = self._assess_severity(error, category, context)
        
        # Check if recoverable
        is_recoverable = self._assess_recoverability(error, category)
        
        # Extract context clues
        context_clues = self._extract_context_clues(error, traceback_str, context)
        
        return ErrorAnalysis(
            category=category,
            confidence=confidence,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            suggested_fixes=suggested_fixes,
            severity=severity,
            is_recoverable=is_recoverable,
            context_clues=context_clues,
        )

    def _extract_primary_cause(self, error: Exception, error_message: str, category: ErrorCategory) -> str:
        """Extract the primary cause of the error."""
        error_type = type(error).__name__
        
        if category == ErrorCategory.SYNTAX:
            return f"Python syntax error: {error_message}"
        elif category == ErrorCategory.DATA:
            return f"Data structure issue: {error_message}"
        elif category == ErrorCategory.RUNTIME:
            return f"Runtime exception ({error_type}): {error_message}"
        elif category == ErrorCategory.PERFORMANCE:
            return f"Performance/resource limitation: {error_message}"
        elif category == ErrorCategory.DEPENDENCY:
            return f"Missing dependency or import issue: {error_message}"
        elif category == ErrorCategory.SECURITY:
            return f"Security validation failure: {error_message}"
        else:
            return f"Unclassified error ({error_type}): {error_message}"

    def _identify_contributing_factors(
        self,
        error: Exception,
        error_message: str,
        traceback_str: str, 
        code: str,
        context: Dict[str, Any],
        category: ErrorCategory
    ) -> List[str]:
        """Identify contributing factors to the error."""
        factors = []
        
        # Check for common contributing factors based on category
        if category == ErrorCategory.DATA:
            # Check for DataFrame-related issues
            if "DataFrame" in str(context.get("raw_dataset", "")):
                df = context.get("raw_dataset")
                if df is not None and hasattr(df, 'columns'):
                    if len(df.columns) == 0:
                        factors.append("DataFrame has no columns")
                    if len(df) == 0:
                        factors.append("DataFrame is empty")
                        
        elif category == ErrorCategory.RUNTIME:
            # Check for variable reference issues
            if "NameError" in str(error):
                factors.append("Variable referenced before assignment or definition")
            if "AttributeError" in str(error):
                factors.append("Method or attribute does not exist on the object")
                
        elif category == ErrorCategory.PERFORMANCE:
            # Check for performance indicators
            if "timeout" in error_message.lower():
                factors.append("Operation exceeded time limit")
            if "memory" in error_message.lower():
                factors.append("Insufficient memory for operation")
                
        # Check code complexity
        if code and len(code.split('\n')) > 50:
            factors.append("Code complexity may contribute to execution issues")
            
        return factors

    def _generate_suggested_fixes(
        self,
        error: Exception,
        error_message: str,
        code: str, 
        category: ErrorCategory,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate suggested fixes for the error."""
        fixes = []
        
        if category == ErrorCategory.SYNTAX:
            fixes.extend([
                "Check Python syntax and indentation",
                "Verify all brackets, quotes, and parentheses are properly closed",
                "Ensure proper code structure and formatting"
            ])
            
        elif category == ErrorCategory.DATA:
            fixes.extend([
                "Verify column names exist in the DataFrame",
                "Check data types are compatible with operations",
                "Handle missing or null values appropriately",
                "Validate DataFrame structure before operations"
            ])
            
        elif category == ErrorCategory.RUNTIME:
            error_type = type(error).__name__
            if error_type == "NameError":
                fixes.append("Define the variable before using it")
            elif error_type == "AttributeError": 
                fixes.append("Check if the object has the required method or attribute")
            elif error_type == "KeyError":
                fixes.append("Verify the key exists in the dictionary or DataFrame")
            elif error_type == "IndexError":
                fixes.append("Check array/list bounds before accessing indices")
                
        elif category == ErrorCategory.PERFORMANCE:
            fixes.extend([
                "Optimize code for better performance",
                "Process data in smaller chunks",
                "Consider using more efficient algorithms",
                "Add progress indicators for long-running operations"
            ])
            
        elif category == ErrorCategory.DEPENDENCY:
            fixes.extend([
                "Install missing Python packages",
                "Check import statements for typos",
                "Verify package versions are compatible"
            ])
            
        # Add general fixes
        fixes.append("Review error message and traceback for specific guidance")
        
        return fixes

    def _assess_severity(self, error: Exception, category: ErrorCategory, context: Dict[str, Any]) -> str:
        """Assess the severity of the error."""
        
        if category == ErrorCategory.SECURITY:
            return "critical"
        elif category == ErrorCategory.SYNTAX:
            return "high"  # Prevents execution entirely
        elif category == ErrorCategory.PERFORMANCE:
            return "medium"  # May affect user experience
        elif category == ErrorCategory.DATA:
            # Data errors can be critical if they corrupt analysis
            if "cannot convert" in str(error).lower():
                return "high"
            return "medium"
        elif category == ErrorCategory.RUNTIME:
            return "medium"
        else:
            return "low"

    def _assess_recoverability(self, error: Exception, category: ErrorCategory) -> bool:
        """Assess if the error is recoverable through retry or modification."""
        
        # Syntax errors require code changes
        if category == ErrorCategory.SYNTAX:
            return True  # Can be fixed by correcting code
            
        # Security errors typically require manual intervention
        if category == ErrorCategory.SECURITY:
            return False
            
        # Performance errors might be recoverable with optimization
        if category == ErrorCategory.PERFORMANCE:
            return True
            
        # Most other errors are recoverable
        return True

    def _extract_context_clues(
        self, 
        error: Exception, 
        traceback_str: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract contextual information that might help with error resolution."""
        clues = {}
        
        # Extract line number from traceback
        line_match = re.search(r'line (\d+)', traceback_str)
        if line_match:
            clues["error_line"] = int(line_match.group(1))
            
        # Extract function/method name
        function_match = re.search(r'in (\w+)', traceback_str)
        if function_match:
            clues["error_function"] = function_match.group(1)
            
        # Add context information
        if context:
            if "raw_dataset" in context:
                df = context["raw_dataset"]
                if hasattr(df, 'shape'):
                    clues["dataset_shape"] = df.shape
                if hasattr(df, 'columns'):
                    clues["dataset_columns"] = list(df.columns)
                    
        # Extract variable names mentioned in error
        var_matches = re.findall(r"'(\w+)' is not defined", str(error))
        if var_matches:
            clues["undefined_variables"] = var_matches
            
        return clues