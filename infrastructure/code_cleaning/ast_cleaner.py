"""
AST-based code cleaning and processing module.

This module provides comprehensive code cleaning functionality using Python's AST
module instead of regex-based approaches, addressing syntax errors and indentation
issues that occur with regex-based cleaning.
"""

import ast
import re
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional dependencies for formatting
try:
    import black

    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False

try:
    import isort

    ISORT_AVAILABLE = True
except ImportError:
    ISORT_AVAILABLE = False

try:
    import astor

    ASTOR_AVAILABLE = True
except ImportError:
    ASTOR_AVAILABLE = False

from infrastructure.logging import get_logger

logger = get_logger(__name__)


class ASTCodeCleaner:
    """
    AST-based code cleaner that properly handles Python code structure.

    This cleaner addresses the issues with regex-based cleaning by:
    - Using AST parsing for proper code structure understanding
    - Preserving indentation through proper AST handling
    - Safe import statement removal
    - Comprehensive syntax validation
    - Automatic formatting with black/isort
    """

    def __init__(
        self,
        allowed_imports: Optional[Set[str]] = None,
        format_code: bool = True,
        preserve_comments: bool = True,
    ):
        """
        Initialize the AST code cleaner.

        Args:
            allowed_imports: Set of allowed import modules. If None, all imports allowed.
            format_code: Whether to format code with black/isort
            preserve_comments: Whether to preserve comments during cleaning
        """
        self.allowed_imports = allowed_imports or set()
        self.format_code = format_code
        self.preserve_comments = preserve_comments

    def clean_code(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """
        Clean Python code using AST-based processing.

        Args:
            code: Raw Python code to clean

        Returns:
            Tuple of (cleaned_code, metadata) where metadata contains:
            - success: bool
            - syntax_valid: bool
            - imports_removed: List[str]
            - formatting_applied: bool
            - original_lines: int
            - cleaned_lines: int
            - errors: List[str]
        """
        metadata = {
            "success": False,
            "syntax_valid": False,
            "imports_removed": [],
            "formatting_applied": False,
            "original_lines": len(code.splitlines()) if code else 0,
            "cleaned_lines": 0,
            "markdown_blocks_removed": 0,
            "errors": [],
        }

        try:
            logger.debug("Starting AST-based code cleaning")

            # Step 1: Remove markdown formatting
            cleaned_code, markdown_blocks_removed = self._remove_markdown_formatting(
                code
            )
            metadata["markdown_blocks_removed"] = markdown_blocks_removed

            # Step 2: Remove explanatory text before code
            cleaned_code = self._remove_explanatory_text(cleaned_code)

            # Step 3: Fix indentation early to avoid syntax errors
            cleaned_code = self._fix_indentation(cleaned_code)

            # Step 4: Parse and validate syntax
            try:
                tree = ast.parse(cleaned_code)
                metadata["syntax_valid"] = True
                logger.debug("Initial syntax validation passed")
            except SyntaxError as e:
                logger.warning(f"Initial syntax error: {e}")
                # Try to fix common syntax issues
                cleaned_code = self._fix_common_syntax_issues(cleaned_code)
                try:
                    tree = ast.parse(cleaned_code)
                    metadata["syntax_valid"] = True
                    logger.debug("Syntax validation passed after fixes")
                except SyntaxError as e2:
                    metadata["errors"].append(f"Syntax error: {e2}")
                    logger.error(f"Could not fix syntax error: {e2}")
                    # Return cleaned code even if syntax is invalid
                    metadata["cleaned_lines"] = len(cleaned_code.splitlines())
                    return cleaned_code, metadata

            # Step 4: Remove forbidden imports if specified
            if self.allowed_imports:
                cleaned_code, removed_imports = self._remove_forbidden_imports(
                    cleaned_code, tree
                )
                metadata["imports_removed"] = removed_imports

                # Re-parse after import removal
                try:
                    tree = ast.parse(cleaned_code)
                except SyntaxError as e:
                    metadata["errors"].append(f"Syntax error after import removal: {e}")
                    logger.error(f"Import removal caused syntax error: {e}")
                    # Return cleaned code even with import removal issues
                    metadata["cleaned_lines"] = len(cleaned_code.splitlines())
                    return cleaned_code, metadata

            # Step 5: Apply black/isort formatting if enabled
            if self.format_code:
                try:
                    cleaned_code = self._apply_formatting(cleaned_code)
                    metadata["formatting_applied"] = True
                    logger.debug("Code formatting applied successfully")
                except Exception as e:
                    logger.warning(f"Formatting failed: {e}")
                    metadata["errors"].append(f"Formatting error: {e}")

            # Step 6: Final syntax validation
            try:
                ast.parse(cleaned_code)
                metadata["syntax_valid"] = True
                metadata["success"] = True
                metadata["cleaned_lines"] = len(cleaned_code.splitlines())

                logger.info(
                    f"Code cleaning successful: {metadata['original_lines']} -> {metadata['cleaned_lines']} lines"
                )

            except SyntaxError as e:
                metadata["errors"].append(f"Final syntax validation failed: {e}")
                logger.error(f"Final syntax validation failed: {e}")
                metadata["cleaned_lines"] = len(cleaned_code.splitlines())
                return cleaned_code, metadata

            return cleaned_code, metadata

        except Exception as e:
            metadata["errors"].append(f"Unexpected error: {e}")
            logger.error(f"Unexpected error in code cleaning: {e}", exc_info=True)
            # Return cleaned_code if available, otherwise original code
            try:
                cleaned_code
                metadata["cleaned_lines"] = len(cleaned_code.splitlines())
                return cleaned_code, metadata
            except NameError:
                return code, metadata

    def _remove_markdown_formatting(self, code: str) -> Tuple[str, int]:
        """Remove markdown code block formatting and count blocks removed."""
        original_code = code
        markdown_blocks_removed = 0

        # Count and remove python code blocks
        python_blocks = len(re.findall(r"^```python\s*\n?", code, flags=re.MULTILINE))
        code = re.sub(r"^```python\s*\n?", "", code, flags=re.MULTILINE)
        markdown_blocks_removed += python_blocks

        # Count and remove generic code blocks
        generic_blocks = len(re.findall(r"^```\s*\n?", code, flags=re.MULTILINE))
        code = re.sub(r"^```\s*\n?", "", code, flags=re.MULTILINE)
        markdown_blocks_removed += generic_blocks

        # Count and remove closing blocks
        closing_blocks = len(re.findall(r"\n```\s*$", code, flags=re.MULTILINE))
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
        markdown_blocks_removed += closing_blocks

        code = code.strip()

        # Only count as markdown blocks removed if we actually changed something
        if code == original_code.strip():
            markdown_blocks_removed = 0

        return code, markdown_blocks_removed

    def _remove_explanatory_text(self, code: str) -> str:
        """
        Remove explanatory text before the actual code using AST-guided approach.

        This method attempts to identify where actual Python code begins by
        looking for valid Python statements rather than using regex patterns.
        """
        lines = code.split("\n")
        code_start_idx = 0
        code_end_idx = len(lines)

        # Find where Python code actually starts
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # Skip obvious non-Python lines
            if any(
                phrase in stripped.lower()
                for phrase in [
                    "here's",
                    "this is",
                    "the following",
                    "code:",
                    "analysis:",
                    "let me",
                    "i'll",
                    "we'll",
                    "you can",
                    "this will",
                    "example:",
                    "note:",
                    "explanation:",
                    "summary:",
                ]
            ):
                continue

            # Check for obvious Python code patterns
            if (
                stripped.startswith(
                    (
                        "import ",
                        "from ",
                        "def ",
                        "class ",
                        "if ",
                        "for ",
                        "while ",
                        "try:",
                        "with ",
                        "@",  # decorators
                    )
                )
                or stripped.startswith(("#", '"""', "'''"))
                or (
                    stripped.endswith(":")
                    and any(
                        keyword in stripped
                        for keyword in [
                            "def ",
                            "class ",
                            "if ",
                            "for ",
                            "while ",
                            "try",
                            "with ",
                            "elif ",
                            "else",
                        ]
                    )
                )
                or (
                    re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=", stripped)
                )  # variable assignment
            ):

                # Extract potential code section (from here to end, or until obvious end)
                potential_code_lines = lines[i:]

                # Remove trailing explanatory text
                for j in range(len(potential_code_lines) - 1, -1, -1):
                    line = potential_code_lines[j].strip()
                    if line and any(
                        phrase in line.lower()
                        for phrase in [
                            "this code",
                            "the code",
                            "explanation",
                            "summary",
                            "note that",
                            "as you can see",
                            "this will",
                            "this shows",
                        ]
                    ):
                        potential_code_lines = potential_code_lines[:j]
                    elif line:  # Stop at first real content line from the end
                        break

                potential_code = "\n".join(potential_code_lines)

                # Try parsing this potential code section
                try:
                    ast.parse(potential_code)
                    code_start_idx = i
                    code_end_idx = i + len(potential_code_lines)
                    break
                except SyntaxError:
                    # Try with indentation fixes
                    try:
                        fixed_code = self._fix_indentation(potential_code)
                        ast.parse(fixed_code)
                        code_start_idx = i
                        code_end_idx = i + len(potential_code_lines)
                        break
                    except SyntaxError:
                        continue

        # Fallback: simple line-by-line approach (much faster)
        if code_start_idx == 0:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue

                # Try parsing from this line onwards to see if it's valid Python
                remaining_code = "\n".join(lines[i:])
                try:
                    ast.parse(remaining_code)
                    code_start_idx = i
                    code_end_idx = len(lines)
                    break
                except SyntaxError:
                    # Try with indentation fixes only
                    try:
                        fixed_remaining = self._fix_indentation(remaining_code)
                        ast.parse(fixed_remaining)
                        code_start_idx = i
                        code_end_idx = len(lines)
                        break
                    except SyntaxError:
                        continue

        if code_start_idx > 0 or code_end_idx < len(lines):
            removed_lines = code_start_idx + (len(lines) - code_end_idx)
            logger.debug(f"Removed {removed_lines} lines of explanatory text")
            return "\n".join(lines[code_start_idx:code_end_idx])

        return code

    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues that can be automatically corrected."""
        lines = code.split("\n")
        fixed_lines = []
        in_multiline_string = False

        for i, line in enumerate(lines):

            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                fixed_lines.append(line)
                continue

            # Fix unterminated string literals
            if line.count('"') % 2 != 0 and not line.strip().startswith("#"):
                if '"' in line and not in_multiline_string:
                    # Check if it's a docstring start
                    if '"""' in line:
                        # Handle docstring
                        fixed_lines.append(line)
                        continue

                    # Find the last quote and ensure it's properly terminated
                    last_quote = line.rfind('"')
                    # Check if there's content after the quote that looks like it should be on next line
                    after_quote = line[last_quote + 1 :].strip()
                    if (
                        after_quote
                        and not after_quote.startswith(")")
                        and not after_quote.startswith(",")
                    ):
                        # Likely unterminated string, close it
                        line = line[: last_quote + 1] + '"'
                    else:
                        line = line[: last_quote + 1]

            # Fix similar issue with single quotes
            if line.count("'") % 2 != 0 and not line.strip().startswith("#"):
                if "'" in line and not in_multiline_string:
                    # Check for triple quotes
                    if "'''" in line:
                        fixed_lines.append(line)
                        continue

                    last_quote = line.rfind("'")
                    after_quote = line[last_quote + 1 :].strip()
                    if (
                        after_quote
                        and not after_quote.startswith(")")
                        and not after_quote.startswith(",")
                    ):
                        line = line[: last_quote + 1] + "'"
                    else:
                        line = line[: last_quote + 1]

            # Fix indentation issues for control structures
            stripped = line.strip()
            if stripped.endswith(":") and i + 1 < len(lines):
                # This is a control structure, check next line indentation
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                if (
                    next_line.strip()
                    and not next_line.startswith(" ")
                    and not next_line.startswith("\t")
                ):
                    # Next line should be indented, fix it
                    if i + 1 < len(lines):
                        lines[i + 1] = "    " + lines[i + 1].lstrip()

            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _remove_forbidden_imports(
        self, code: str, tree: ast.AST
    ) -> Tuple[str, List[str]]:
        """
        Remove forbidden imports using AST analysis.

        This method properly handles import removal while preserving code structure
        and indentation.
        """
        removed_imports = []

        class ImportRemover(ast.NodeTransformer):
            def __init__(self, allowed_imports: Set[str]):
                self.allowed_imports = allowed_imports
                self.removed = []

            def visit_Import(self, node):
                # Check each imported module
                new_names = []
                for alias in node.names:
                    module_name = alias.name.split(".")[0]  # Get root module
                    if module_name in self.allowed_imports or not self.allowed_imports:
                        new_names.append(alias)
                    else:
                        self.removed.append(f"import {alias.name}")

                if new_names:
                    node.names = new_names
                    return node
                else:
                    return None  # Remove the entire import statement

            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split(".")[0]  # Get root module
                    if module_name not in self.allowed_imports and self.allowed_imports:
                        self.removed.append(f"from {node.module} import ...")
                        return None  # Remove the import
                return node

        # Apply import removal
        if self.allowed_imports:
            remover = ImportRemover(self.allowed_imports)
            new_tree = remover.visit(tree)
            removed_imports = remover.removed

            # Convert back to code
            if ASTOR_AVAILABLE:
                try:
                    code = astor.to_source(new_tree)
                except Exception as e:
                    logger.warning(f"astor conversion failed: {e}")
            else:
                logger.warning("astor not available, skipping import removal")
                # Return original code if astor is not available

        return code, removed_imports

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues using syntax-aware approach."""
        try:
            # First, try to dedent the entire code block
            dedented = textwrap.dedent(code)

            # If the dedented code already parses correctly, return it
            try:
                ast.parse(dedented)
                return dedented
            except SyntaxError:
                pass

            # Otherwise, apply indentation fixing
            lines = dedented.split("\n")
            fixed_lines = []
            expected_indent = 0

            for i, line in enumerate(lines):
                stripped = line.strip()

                if not stripped:  # Empty line
                    fixed_lines.append("")
                    continue

                # Check if this line should be dedented (elif, else, except, finally, etc.)
                if stripped.startswith(
                    ("elif ", "else:", "except", "except:", "finally:")
                ):
                    expected_indent = max(0, expected_indent - 1)

                # Apply the expected indentation
                fixed_line = "    " * expected_indent + stripped
                fixed_lines.append(fixed_line)

                # Check if next line should be indented
                if (
                    stripped.endswith(":")
                    and not stripped.startswith(('"""', "'''", "#"))
                    and not (
                        stripped.startswith('"""')
                        and stripped.endswith('"""')
                        and len(stripped) > 6
                    )
                ):
                    expected_indent += 1

            return "\n".join(fixed_lines)

        except Exception as e:
            logger.warning(f"Syntax-aware indentation fixing failed: {e}")
            # Fallback to simple indentation fixing
            try:
                dedented = textwrap.dedent(code)
                lines = dedented.split("\n")
                fixed_lines = []

                for line in lines:
                    if line.strip():  # Non-empty line
                        # Count leading whitespace
                        leading_space = len(line) - len(line.lstrip())
                        indent_level = leading_space // 4  # Assume 4-space indentation
                        remainder = leading_space % 4

                        # Normalize to 4-space indentation
                        if remainder != 0:
                            indent_level += 1  # Round up partial indentation

                        fixed_line = "    " * indent_level + line.lstrip()
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append("")  # Preserve empty lines

                return "\n".join(fixed_lines)
            except Exception as e2:
                logger.warning(f"Fallback indentation fixing also failed: {e2}")
                return code

    def _apply_formatting(self, code: str) -> str:
        """Apply black and isort formatting if available."""
        try:
            # Apply isort first for import sorting
            if ISORT_AVAILABLE:
                code = isort.code(code, profile="black")
            else:
                logger.debug("isort not available, skipping import sorting")

            # Apply black formatting
            if BLACK_AVAILABLE:
                code = black.format_str(code, mode=black.FileMode())
            else:
                logger.debug("black not available, skipping code formatting")

            return code

        except Exception as e:
            logger.warning(f"Code formatting failed: {e}")
            return code

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST parsing.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected validation error: {str(e)}"
            return False, error_msg


def create_ast_cleaner(allowed_imports: Optional[Set[str]] = None) -> ASTCodeCleaner:
    """
    Factory function to create an AST code cleaner with default settings.

    Args:
        allowed_imports: Set of allowed import modules

    Returns:
        Configured ASTCodeCleaner instance
    """
    return ASTCodeCleaner(
        allowed_imports=allowed_imports, format_code=True, preserve_comments=True
    )
