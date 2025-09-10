"""Secure code execution environment with sandboxing."""
import sys
import io
import time
import traceback
import resource
import signal
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

from config import config
from workflow.state import ExecutionStatus, ExecutionResults
from logging_config import get_logger

logger = get_logger(__name__)


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


class MemoryLimitException(Exception):
    """Exception raised when memory limit is exceeded."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")


class SecureExecutor:
    """Secure Python code executor with resource limits and sandboxing."""
    
    def __init__(self):
        """Initialize the secure executor."""
        self.max_execution_time = config.execution_limits.max_execution_time
        self.max_memory_mb = config.execution_limits.max_memory_mb
        self.max_output_size_mb = config.execution_limits.max_output_size_mb
    
    def execute_code(self, code: str, context: Dict[str, Any] = None) -> ExecutionResults:
        """
        Execute Python code in a secure environment.
        
        Args:
            code: Python code to execute
            context: Context variables to inject (e.g., DataFrame)
            
        Returns:
            ExecutionResults with execution details
        """
        start_time = time.time()
        
        # Prepare execution context
        if context is None:
            context = {}
        
        # Create safe execution environment
        safe_globals = self._create_safe_globals()
        safe_locals = {**context}
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Set resource limits
            self._set_resource_limits()
            
            # Set execution timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.max_execution_time)
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, safe_globals, safe_locals)
            
            # Cancel timeout
            signal.alarm(0)
            
            # Get execution results
            execution_time = time.time() - start_time
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            # Check output size
            total_output_size = len(stdout_content) + len(stderr_content)
            max_output_bytes = self.max_output_size_mb * 1024 * 1024
            
            if total_output_size > max_output_bytes:
                stdout_content = stdout_content[:max_output_bytes//2] + "\n[OUTPUT TRUNCATED]"
                stderr_content = stderr_content[:max_output_bytes//2] + "\n[OUTPUT TRUNCATED]"
            
            # Extract analysis results if available
            output_data = safe_locals.get('analysis_results', {})
            
            # Get memory usage (approximate)
            memory_used = self._get_memory_usage()
            
            return ExecutionResults(
                status=ExecutionStatus.SUCCESS,
                output_data=output_data,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                stdout=stdout_content,
                stderr=stderr_content
            )
            
        except TimeoutException:
            signal.alarm(0)
            execution_time = time.time() - start_time
            error_msg = f"Code execution timed out after {self.max_execution_time} seconds"
            logger.warning(error_msg)
            
            return ExecutionResults(
                status=ExecutionStatus.TIMEOUT,
                execution_time=execution_time,
                error_message=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        except MemoryError:
            signal.alarm(0)
            execution_time = time.time() - start_time
            error_msg = f"Code execution exceeded memory limit of {self.max_memory_mb}MB"
            logger.warning(error_msg)
            
            return ExecutionResults(
                status=ExecutionStatus.FAILED,
                execution_time=execution_time,
                error_message=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        except Exception as e:
            signal.alarm(0)
            execution_time = time.time() - start_time
            error_msg = f"Code execution failed: {str(e)}"
            error_traceback = traceback.format_exc()
            logger.error(f"Execution error: {error_msg}\\n{error_traceback}")
            
            return ExecutionResults(
                status=ExecutionStatus.FAILED,
                execution_time=execution_time,
                error_message=error_msg,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + f"\\n{error_traceback}"
            )
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe global environment for code execution."""
        # Start with minimal builtins
        safe_builtins = {
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
            'min', 'max', 'sum', 'abs', 'round', 'sorted', 'enumerate', 'zip',
            'range', 'print', 'type', 'isinstance', 'any', 'all', '__import__',
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError'
        }
        
        safe_globals = {
            '__builtins__': {name: __builtins__[name] for name in safe_builtins if name in __builtins__}
        }
        
        # Add allowed modules
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from scipy import stats
            from scipy.stats import pearsonr, spearmanr, normaltest, shapiro
            import statsmodels.api as sm
            from prophet import Prophet
            import datetime
            from datetime import datetime, timedelta
            import math
            import statistics
            import json
            import re
            import warnings
            
            safe_globals.update({
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'px': px,
                'go': go,
                'KMeans': KMeans,
                'StandardScaler': StandardScaler,
                'silhouette_score': silhouette_score,
                'train_test_split': train_test_split,
                'LinearRegression': LinearRegression,
                'LogisticRegression': LogisticRegression,
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'stats': stats,
                'pearsonr': pearsonr,
                'spearmanr': spearmanr,
                'normaltest': normaltest,
                'shapiro': shapiro,
                'sm': sm,
                'Prophet': Prophet,
                'datetime': datetime,
                'timedelta': timedelta,
                'math': math,
                'statistics': statistics,
                'json': json,
                're': re,
                'warnings': warnings
            })
            
        except ImportError as e:
            logger.warning(f"Could not import some modules for safe execution: {e}")
        
        return safe_globals
    
    def _set_resource_limits(self):
        """Set resource limits for code execution."""
        try:
            # Set memory limit (soft limit)
            memory_limit = self.max_memory_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            cpu_limit = self.max_execution_time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
        except (OSError, ValueError) as e:
            logger.warning(f"Could not set resource limits: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0


class ProcessSafeExecutor:
    """Process-based executor for additional isolation."""
    
    def __init__(self):
        """Initialize process-safe executor."""
        self.max_execution_time = config.execution_limits.max_execution_time
    
    def execute_code_in_process(self, code: str, context: Dict[str, Any] = None) -> ExecutionResults:
        """
        Execute code in a separate process for additional isolation.
        
        Args:
            code: Python code to execute
            context: Context variables
            
        Returns:
            ExecutionResults with execution details
        """
        # Create a queue to get results from the process
        result_queue = mp.Queue()
        
        # Create and start the execution process
        process = mp.Process(
            target=self._execute_in_process,
            args=(code, context, result_queue)
        )
        
        start_time = time.time()
        process.start()
        process.join(timeout=self.max_execution_time)
        
        if process.is_alive():
            # Process timed out
            process.terminate()
            process.join()
            
            return ExecutionResults(
                status=ExecutionStatus.TIMEOUT,
                execution_time=time.time() - start_time,
                error_message=f"Process execution timed out after {self.max_execution_time} seconds"
            )
        
        # Get results from the queue
        try:
            result = result_queue.get_nowait()
            result.execution_time = time.time() - start_time
            return result
        except:
            return ExecutionResults(
                status=ExecutionStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message="Failed to retrieve execution results from process"
            )
    
    def _execute_in_process(self, code: str, context: Dict[str, Any], result_queue: mp.Queue):
        """Execute code within a separate process."""
        try:
            executor = SecureExecutor()
            result = executor.execute_code(code, context)
            result_queue.put(result)
        except Exception as e:
            error_result = ExecutionResults(
                status=ExecutionStatus.FAILED,
                error_message=f"Process execution error: {str(e)}"
            )
            result_queue.put(error_result)


# Global executor instances
secure_executor = SecureExecutor()
process_executor = ProcessSafeExecutor()
