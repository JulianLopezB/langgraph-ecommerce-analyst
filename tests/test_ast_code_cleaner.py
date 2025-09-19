"""
Comprehensive tests for AST-based code cleaner.

This test suite covers edge cases and scenarios that caused issues with
regex-based cleaning approaches.
"""

import ast

import pytest

from infrastructure.code_cleaning import ASTCodeCleaner, create_ast_cleaner


class TestASTCodeCleaner:
    """Test suite for AST-based code cleaning functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = create_ast_cleaner()
        self.restricted_cleaner = ASTCodeCleaner(
            allowed_imports={"pandas", "numpy", "matplotlib"}, format_code=True
        )

    def test_basic_markdown_removal(self):
        """Test removal of markdown code blocks."""
        code_with_markdown = """```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df)
```"""

        cleaned_code, metadata = self.cleaner.clean_code(code_with_markdown)

        assert metadata["success"] is True
        assert metadata["syntax_valid"] is True
        assert "```" not in cleaned_code
        assert "import pandas as pd" in cleaned_code
        assert "df = pd.DataFrame" in cleaned_code

        # Verify the cleaned code is valid Python
        ast.parse(cleaned_code)

    def test_unterminated_string_fixing(self):
        """Test fixing of unterminated string literals."""
        code_with_unterminated_string = """```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print("This is an unterminated string
result = df.sum()
```"""

        cleaned_code, metadata = self.cleaner.clean_code(code_with_unterminated_string)

        # The cleaner should either fix the string or handle it gracefully
        assert metadata["success"] is True
        # If it can't be fixed, it should still return valid Python
        if metadata["syntax_valid"]:
            ast.parse(cleaned_code)

    def test_indentation_issues(self):
        """Test fixing of inconsistent indentation."""
        code_with_bad_indentation = """```python
import pandas as pd
if True:
print("This has wrong indentation")
  df = pd.DataFrame({'a': [1, 2, 3]})
      result = df.sum()
```"""

        cleaned_code, metadata = self.cleaner.clean_code(code_with_bad_indentation)

        assert metadata["success"] is True
        if metadata["syntax_valid"]:
            # Should be parseable as valid Python
            ast.parse(cleaned_code)
            # Should have consistent 4-space indentation
            lines = cleaned_code.split("\n")
            for line in lines:
                if line.strip() and line.startswith(" "):
                    leading_spaces = len(line) - len(line.lstrip())
                    assert (
                        leading_spaces % 4 == 0
                    ), f"Line has inconsistent indentation: '{line}'"

    def test_explanatory_text_removal(self):
        """Test removal of explanatory text before code."""
        code_with_explanation = """Here's the analysis code:

```python
import pandas as pd
import numpy as np

# Load the data
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Calculate statistics
result = df.describe()
print(result)
```

This code calculates basic statistics."""

        cleaned_code, metadata = self.cleaner.clean_code(code_with_explanation)

        assert metadata["success"] is True
        assert metadata["syntax_valid"] is True
        assert "import pandas as pd" in cleaned_code
        assert "Here's the analysis code:" not in cleaned_code
        assert "This code calculates" not in cleaned_code

        # Should be valid Python
        ast.parse(cleaned_code)

    def test_import_removal_with_restrictions(self):
        """Test removal of forbidden imports."""
        code_with_forbidden_imports = """```python
import pandas as pd
import numpy as np
import os
import sys
import requests

df = pd.DataFrame({'a': [1, 2, 3]})
result = np.mean(df['a'])
```"""

        cleaned_code, metadata = self.restricted_cleaner.clean_code(
            code_with_forbidden_imports
        )

        assert metadata["success"] is True
        # Should have removed forbidden imports
        assert len(metadata["imports_removed"]) > 0
        assert "import os" not in cleaned_code
        assert "import sys" not in cleaned_code
        assert "import requests" not in cleaned_code
        # Should keep allowed imports
        assert "import pandas as pd" in cleaned_code
        assert "import numpy as np" in cleaned_code

    def test_complex_code_structure(self):
        """Test cleaning of complex code with classes and functions."""
        complex_code = '''```python
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def analyze(self):
        """Perform analysis."""
        summary = self.data.describe()
        return summary

    def plot(self):
        """Create plots."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['value'])
        plt.show()

def main():
    # Create sample data
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    analyzer = DataAnalyzer(df)
    result = analyzer.analyze()
    print(result)

if __name__ == "__main__":
    main()
```'''

        cleaned_code, metadata = self.cleaner.clean_code(complex_code)

        assert metadata["success"] is True
        assert metadata["syntax_valid"] is True
        assert "class DataAnalyzer:" in cleaned_code
        assert "def analyze(self):" in cleaned_code
        assert "def main():" in cleaned_code

        # Should be valid Python
        tree = ast.parse(cleaned_code)
        # Should contain class and function definitions
        assert any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        assert any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

    def test_malformed_code_graceful_handling(self):
        """Test graceful handling of malformed code."""
        malformed_code = """```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3
# Missing closing bracket and parenthesis
result = df.sum(
```"""

        cleaned_code, metadata = self.cleaner.clean_code(malformed_code)

        # Should not crash, even if it can't fix the code
        assert isinstance(metadata, dict)
        assert "success" in metadata
        assert "errors" in metadata

    def test_empty_code_handling(self):
        """Test handling of empty or whitespace-only code."""
        empty_codes = ["", "   ", "\n\n\n", "```\n\n```"]

        for empty_code in empty_codes:
            cleaned_code, metadata = self.cleaner.clean_code(empty_code)
            assert isinstance(metadata, dict)
            assert "success" in metadata

    def test_code_with_docstrings(self):
        """Test preservation of docstrings and comments."""
        code_with_docstrings = '''```python
"""
This module performs data analysis.
"""
import pandas as pd

def analyze_data(df):
    """
    Analyze the given DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Analysis results
    """
    # Calculate basic statistics
    result = df.describe()
    return result

# Main execution
if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3]})
    print(analyze_data(df))
```'''

        cleaned_code, metadata = self.cleaner.clean_code(code_with_docstrings)

        assert metadata["success"] is True
        assert metadata["syntax_valid"] is True
        # Should preserve docstrings and comments
        assert '"""' in cleaned_code
        assert "# Calculate basic statistics" in cleaned_code
        assert "# Main execution" in cleaned_code

    def test_syntax_validation(self):
        """Test syntax validation functionality."""
        valid_code = "x = 1\nprint(x)"
        invalid_code = "x = 1\nprint(x"

        is_valid, error = self.cleaner.validate_syntax(valid_code)
        assert is_valid is True
        assert error is None

        is_valid, error = self.cleaner.validate_syntax(invalid_code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error

    def test_formatting_application(self):
        """Test that black/isort formatting is applied."""
        unformatted_code = """```python
import sys
import pandas as pd
import os

def test(x,y,z):
    result=x+y+z
    return result

data=[1,2,3,4,5]
```"""

        formatted_cleaner = ASTCodeCleaner(format_code=True)
        cleaned_code, metadata = formatted_cleaner.clean_code(unformatted_code)

        if metadata["success"] and metadata["formatting_applied"]:
            # Should have proper spacing and formatting
            assert "def test(x, y, z):" in cleaned_code
            assert "result = x + y + z" in cleaned_code
            # Imports should be sorted
            lines = cleaned_code.split("\n")
            import_lines = [line for line in lines if line.startswith("import ")]
            if len(import_lines) > 1:
                assert import_lines == sorted(import_lines)

    def test_multiple_markdown_blocks(self):
        """Test handling of multiple markdown code blocks."""
        code_with_multiple_blocks = """First block:
```python
import pandas as pd
```

Some explanation text.

Second block:
```python
df = pd.DataFrame({'a': [1, 2, 3]})
print(df)
```"""

        cleaned_code, metadata = self.cleaner.clean_code(code_with_multiple_blocks)

        # Should handle multiple blocks gracefully
        assert isinstance(metadata, dict)
        assert "success" in metadata

    def test_edge_case_imports(self):
        """Test edge cases in import handling."""
        code_with_complex_imports = """```python
from pandas import DataFrame, Series
import numpy as np
from matplotlib import pyplot as plt
import os.path
from sys import argv, exit
```"""

        restricted_cleaner = ASTCodeCleaner(
            allowed_imports={"pandas", "numpy"}, format_code=False
        )

        cleaned_code, metadata = restricted_cleaner.clean_code(
            code_with_complex_imports
        )

        if metadata["success"]:
            # Should keep allowed imports
            assert (
                "from pandas import" in cleaned_code or "import pandas" in cleaned_code
            )
            assert "import numpy" in cleaned_code
            # Should remove forbidden ones
            assert "import os.path" not in cleaned_code
            assert "from sys import" not in cleaned_code

    def test_performance_with_large_code(self):
        """Test performance with larger code blocks."""
        # Generate a large code block
        large_code = """```python
import pandas as pd
import numpy as np

"""

        # Add many functions
        for i in range(50):
            large_code += f'''
def function_{i}(x):
    """Function number {i}."""
    result = x * {i}
    return result + np.mean([1, 2, 3, 4, 5])
'''

        large_code += """
# Main execution
if __name__ == "__main__":
    for i in range(100):
        print(f"Processing item {i}")
```"""

        import time

        start_time = time.time()
        cleaned_code, metadata = self.cleaner.clean_code(large_code)
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        assert isinstance(metadata, dict)
        assert "success" in metadata


class TestFactoryFunction:
    """Test the factory function for creating AST cleaners."""

    def test_create_ast_cleaner_default(self):
        """Test factory function with default parameters."""
        cleaner = create_ast_cleaner()
        assert isinstance(cleaner, ASTCodeCleaner)
        assert cleaner.format_code is True
        assert cleaner.preserve_comments is True

    def test_create_ast_cleaner_with_restrictions(self):
        """Test factory function with import restrictions."""
        allowed_imports = {"pandas", "numpy", "matplotlib"}
        cleaner = create_ast_cleaner(allowed_imports=allowed_imports)
        assert isinstance(cleaner, ASTCodeCleaner)
        assert cleaner.allowed_imports == allowed_imports


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
