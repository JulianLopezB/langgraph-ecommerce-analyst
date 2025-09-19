"""Security isolation components for safe code execution."""

import importlib
import warnings
from typing import Any, Dict, List, Set

from infrastructure.logging import get_logger

logger = get_logger(__name__)


class SecurityIsolator:
    """Handles security isolation and safe environment creation."""
    
    def __init__(self, allowed_builtins: Set[str] = None, allowed_modules: Dict[str, str] = None):
        """
        Initialize security isolator.
        
        Args:
            allowed_builtins: Set of allowed builtin function names
            allowed_modules: Dict mapping module names to import aliases
        """
        self.allowed_builtins = allowed_builtins or self._get_default_builtins()
        self.allowed_modules = allowed_modules or self._get_default_modules()
        
    def create_safe_globals(self) -> Dict[str, Any]:
        """
        Create a safe global environment for code execution.
        
        Returns:
            Dictionary containing safe global variables and modules
        """
        # Create safe builtins
        safe_globals = {
            "__builtins__": self._create_safe_builtins()
        }
        
        # Add allowed modules
        safe_modules = self._import_safe_modules()
        safe_globals.update(safe_modules)
        
        return safe_globals
    
    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Create dictionary of safe builtin functions."""
        safe_builtins = {}
        
        for name in self.allowed_builtins:
            if hasattr(__builtins__, name):
                safe_builtins[name] = getattr(__builtins__, name)
            elif isinstance(__builtins__, dict) and name in __builtins__:
                safe_builtins[name] = __builtins__[name]
                
        return safe_builtins
    
    def _import_safe_modules(self) -> Dict[str, Any]:
        """Import and return safe modules."""
        safe_modules = {}
        
        for module_name, alias in self.allowed_modules.items():
            try:
                if module_name == "datetime":
                    from datetime import datetime, timedelta
                    safe_modules["datetime"] = datetime
                    safe_modules["timedelta"] = timedelta
                elif module_name == "scipy.stats":
                    from scipy.stats import normaltest, pearsonr, shapiro, spearmanr
                    safe_modules.update({
                        "pearsonr": pearsonr,
                        "spearmanr": spearmanr,
                        "normaltest": normaltest,
                        "shapiro": shapiro
                    })
                elif module_name == "sklearn":
                    from sklearn.cluster import KMeans
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from sklearn.linear_model import LinearRegression, LogisticRegression
                    from sklearn.metrics import silhouette_score
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler
                    
                    safe_modules.update({
                        "KMeans": KMeans,
                        "StandardScaler": StandardScaler,
                        "silhouette_score": silhouette_score,
                        "train_test_split": train_test_split,
                        "LinearRegression": LinearRegression,
                        "LogisticRegression": LogisticRegression,
                        "RandomForestClassifier": RandomForestClassifier,
                        "RandomForestRegressor": RandomForestRegressor,
                    })
                else:
                    # Standard module import - use importlib to correctly import submodules
                    module = importlib.import_module(module_name)
                    safe_modules[alias] = module
                    
            except ImportError as e:
                logger.warning(f"Could not import module {module_name}: {e}")
                continue
        
        return safe_modules
    
    def _get_default_builtins(self) -> Set[str]:
        """Get default set of allowed builtin functions."""
        return {
            "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
            "min", "max", "sum", "abs", "round", "sorted", "enumerate", "zip", "range",
            "print", "type", "isinstance", "any", "all", "hasattr", "getattr",
            "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError"
        }
    
    def _get_default_modules(self) -> Dict[str, str]:
        """Get default set of allowed modules with their aliases."""
        return {
            "pandas": "pd",
            "numpy": "np", 
            "matplotlib.pyplot": "plt",
            "seaborn": "sns",
            "plotly.express": "px",
            "plotly.graph_objects": "go",
            "sklearn": "sklearn",  # Special handling in _import_safe_modules
            "scipy.stats": "stats",  # Special handling in _import_safe_modules
            "statsmodels.api": "sm",
            "prophet": "Prophet",
            "datetime": "datetime",  # Special handling in _import_safe_modules
            "math": "math",
            "statistics": "statistics", 
            "json": "json",
            "re": "re",
            "warnings": "warnings"
        }


class ContextIsolator:
    """Handles execution context isolation and preparation."""
    
    def __init__(self):
        """Initialize context isolator."""
        self.security_isolator = SecurityIsolator()
    
    def prepare_execution_context(self, user_context: Dict[str, Any] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare isolated execution context.
        
        Args:
            user_context: User-provided context variables
            
        Returns:
            Tuple of (safe_globals, safe_locals)
        """
        # Create safe globals
        safe_globals = self.security_isolator.create_safe_globals()
        
        # Prepare safe locals with user context
        safe_locals = {}
        if user_context:
            # Filter user context for safety (basic validation)
            safe_locals = self._filter_user_context(user_context)
            
        return safe_globals, safe_locals
    
    def _filter_user_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter user context for basic safety.
        
        Args:
            context: Raw user context
            
        Returns:
            Filtered safe context
        """
        safe_context = {}
        
        for key, value in context.items():
            # Basic filtering - no private attributes or dangerous objects
            if not key.startswith('_') and not callable(value):
                safe_context[key] = value
            else:
                logger.warning(f"Filtered out potentially unsafe context item: {key}")
                
        return safe_context