"""
String utility functions for AQUA diagnostics.
"""

def harmonize_lists(*lists, sep: str = ' ') -> list:
    """
    Combines multiple lists element-wise, skipping empty/None values.
    """
    combined = [sep.join(filter(None, map(str, row))).strip() for row in zip(*lists)]
    return [item for item in combined if item]

