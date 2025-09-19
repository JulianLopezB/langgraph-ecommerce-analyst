"""
Code cleaning and processing infrastructure.

This module provides AST-based code cleaning functionality to replace
regex-based approaches that cause syntax errors and indentation issues.
"""

from .ast_cleaner import ASTCodeCleaner, create_ast_cleaner

__all__ = ['ASTCodeCleaner', 'create_ast_cleaner']