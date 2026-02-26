"""
String utility functions for AQUA diagnostics.
"""

def harmonize_lists(*lists, sep: str = ' ') -> list:
    """
    Combine multiple lists element-wise into strings, skipping empty/None values.
    Rows that end up empty after filtering are dropped.

    Args:
        *lists: One or more lists (or iterables) of the same length.
        sep (str): String used to join elements in each row.

    Returns:
        list of str: With no empty strings.
    """
    combined = [sep.join(filter(None, map(str, row))).strip() for row in zip(*lists)]
    return [item for item in combined if item]
