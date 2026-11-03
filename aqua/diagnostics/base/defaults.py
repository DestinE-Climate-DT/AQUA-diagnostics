"""Common defaults for AQUA diagnostics"""

from aqua.core.default import DEFAULT_FIGSIZE_HORIZONTAL

# Default name for the oceanic vertical coordinate used across diagnostics.
DEFAULT_OCEAN_VERT_COORD = "depth"

# Centralised default list of figure formats used when saving plots.
SAVE_FORMAT = ["png", "pdf", "svg"]

# Width (inches) shared by every diagnostic figure, taken from AQUA-core so that one constant drives both
FIGURE_WIDTH = DEFAULT_FIGSIZE_HORIZONTAL[0]
