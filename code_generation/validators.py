"""Code validation and security scanning for generated Python code."""
import ast
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Set

from config import config
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    syntax_errors: List[str]
    security_warnings: List[str]
    performance_warnings: List[str]
    validation_time: float
    security_score: float


class CodeValidator:
    """Validates generated Python code for syntax, security, and performance."""
    
    def __init__(self):
        """Initialize validator with security configuration."""
        self.forbidden_patterns = config.security_settings.forbidden_patterns
        self.allowed_imports = set(config.security_settings.allowed_imports)
        
        # Additional security patterns
        self.dangerous_functions = {
            '__import__', 'eval', 'exec', 'compile', 'globals', 'locals', 'vars',
            'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'callable'
        }
        
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'importlib', 'builtins', 'socket',
            'urllib', 'requests', 'http', 'ftplib', 'smtplib', 'pickle',
            'shelve', 'marshal', 'ctypes', 'multiprocessing', 'threading'
        }
        
        self.file_operations = {
            'open', 'file', 'input', 'raw_input', 'execfile'
        }
    
    def validate(self, code: str) -> ValidationResult:
        """
        Comprehensive validation of Python code.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with validation details
        """
        start_time = time.time()
        
        syntax_errors = []
        security_warnings = []
        performance_warnings = []
        security_score = 1.0
        
        # 1. Syntax validation
        syntax_valid, syntax_error = self._validate_syntax(code)
        if not syntax_valid:
            syntax_errors.append(syntax_error)
        
        # 2. Security validation
        security_issues = self._validate_security(code)
        security_warnings.extend(security_issues)
        
        # 3. Import validation
        import_issues = self._validate_imports(code)
        security_warnings.extend(import_issues)
        
        # 4. Performance validation
        performance_issues = self._validate_performance(code)
        performance_warnings.extend(performance_issues)
        
        # Calculate security score (0.0 = very dangerous, 1.0 = safe)
        security_score = max(0.0, 1.0 - (len(security_warnings) * 0.1))
        
        # Overall validation result
        is_valid = (
            syntax_valid and 
            len(security_warnings) == 0 and 
            security_score >= 0.7
        )
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=is_valid,
            syntax_errors=syntax_errors,
            security_warnings=security_warnings,
            performance_warnings=performance_warnings,
            validation_time=validation_time,
            security_score=security_score
        )
    
    def _validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Syntax validation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _validate_security(self, code: str) -> List[str]:
        """Validate code for security issues."""
        issues = []
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern in code:
                issues.append(f"Forbidden pattern detected: {pattern}")
        
        # Check for dangerous functions
        for func in self.dangerous_functions:
            if re.search(rf'\b{re.escape(func)}\s*\(', code):
                issues.append(f"Dangerous function detected: {func}")
        
        # Check for file operations
        for file_op in self.file_operations:
            if re.search(rf'\b{re.escape(file_op)}\s*\(', code):
                issues.append(f"File operation detected: {file_op}")
        
        # Check for shell command patterns
        shell_patterns = [
            r'os\.system\s*\(',
            r'os\.popen\s*\(',
            r'subprocess\.',
            r'shell\s*=\s*True',
            r'exec\s*\(',
            r'eval\s*\('
        ]
        
        for pattern in shell_patterns:
            if re.search(pattern, code):
                issues.append(f"Shell command pattern detected: {pattern}")
        
        # Check for network operations
        network_patterns = [
            r'socket\.',
            r'urllib\.',
            r'requests\.',
            r'http\.',
            r'ftp\.'
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, code):
                issues.append(f"Network operation detected: {pattern}")
        
        return issues
    
    def _validate_imports(self, code: str) -> List[str]:
        """Validate import statements."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in self.allowed_imports:
                            issues.append(f"Unauthorized import: {module_name}")
                        if module_name in self.dangerous_modules:
                            issues.append(f"Dangerous module import: {module_name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in self.allowed_imports:
                            issues.append(f"Unauthorized import from: {module_name}")
                        if module_name in self.dangerous_modules:
                            issues.append(f"Dangerous module import from: {module_name}")
        
        except Exception as e:
            issues.append(f"Error parsing imports: {str(e)}")
        
        return issues
    
    def _validate_performance(self, code: str) -> List[str]:
        """Validate code for potential performance issues."""
        warnings = []
        
        # Check for nested loops
        nested_loop_pattern = r'for\s+\w+\s+in\s+.*:\s*.*for\s+\w+\s+in'
        if re.search(nested_loop_pattern, code, re.DOTALL):
            warnings.append("Nested loops detected - may impact performance")
        
        # Check for large data operations
        if 'df.iterrows()' in code:
            warnings.append("iterrows() detected - consider vectorized operations")
        
        if 'df.apply(' in code and 'axis=1' in code:
            warnings.append("Row-wise apply detected - consider vectorized operations")
        
        # Check for inefficient string operations
        if '+=' in code and 'str' in code:
            warnings.append("String concatenation with += detected - consider join()")
        
        # Check for excessive plotting
        plot_count = len(re.findall(r'plt\.figure\(|plt\.subplot\(|sns\.', code))
        if plot_count > 10:
            warnings.append(f"Many plots ({plot_count}) detected - may consume memory")
        
        return warnings
    
    def get_allowed_imports(self) -> Set[str]:
        """Get set of allowed import modules."""
        return self.allowed_imports.copy()
    
    def add_allowed_import(self, module_name: str) -> None:
        """Add a module to the allowed imports list."""
        self.allowed_imports.add(module_name)
        logger.info(f"Added {module_name} to allowed imports")
    
    def remove_allowed_import(self, module_name: str) -> None:
        """Remove a module from the allowed imports list."""
        if module_name in self.allowed_imports:
            self.allowed_imports.remove(module_name)
            logger.info(f"Removed {module_name} from allowed imports")
