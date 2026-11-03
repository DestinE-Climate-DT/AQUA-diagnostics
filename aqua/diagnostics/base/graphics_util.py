"""Graphics utilities for AQUA diagnostics"""

from .defaults import FIGURE_WIDTH


def figure_size(height_ratio: float) -> tuple:
    """
    Figure size with the common width.

    The width is the same for every figure, so that text (sized in points by the AQUA style) looks equally
    large once figures are shown at the same width, as in the dashboard. Only the height is chosen, from the
    shape of the layout: e.g. 0.77 for a single global map, or proportional to nrows / ncols for a grid of panels.

    Args:
        height_ratio (float): Height of the figure as a fraction of its width.

    Returns:
        tuple: (FIGURE_WIDTH, FIGURE_WIDTH * height_ratio), in inches.
    """
    return (FIGURE_WIDTH, FIGURE_WIDTH * height_ratio)
