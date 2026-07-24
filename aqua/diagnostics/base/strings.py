"""
String utility functions for AQUA diagnostics.
"""

import re


def collapse_era5_duplicate(text: str) -> str:
    """
    ERA5 is catalogued with both model ('ERA5') and experiment ('era5'), which would
    otherwise render as the duplicate 'ERA5 era5' in titles and captions.

    Args:
        text (str): Title or caption text. Falsy values (None, '') are returned unchanged.

    Returns:
        str: Text with 'ERA5 era5' collapsed to 'ERA5'.
    """
    if not text:
        return text
    return re.sub(r"ERA5 era5(?![\w-])", "ERA5", text)


def harmonize_lists(*lists, sep: str = " ") -> list:
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
