"""Graphics utilities for AQUA diagnostics"""

from .defaults import FIGURE_WIDTH


def figure_size(height_ratio: float) -> tuple:
    """
    Figure size (inches) with the common width FIGURE_WIDTH and height = FIGURE_WIDTH * height_ratio.

    The common width keeps text equally large in all figures shown at the same width (e.g. the dashboard).
    """
    return (FIGURE_WIDTH, FIGURE_WIDTH * height_ratio)
