"""tropical_cyclones module"""

# The following lines are needed so that the tropical cyclones class constructor
# and associated functions are available directly from the module "tropical_cyclones"

from .tropical_cyclones import TCs
from .detect_nodes import DetectNodes
from .stitch_nodes import StitchNodes
from .plots.plotting_TCs import multi_plot, plot_trajectories
from .plots.plotting_hist import plot_hist_cat, plot_press_wind


__version__ = "0.2.0"

# This specifies which methods are exported publicly, used by "from tropical cyclones import *"
__all__ = ["TCs", "DetectNodes", "StitchNodes", "plot_hist_cat", "plot_press_wind", "multi_plot", "plot_trajectories","category_from_slp_pa","get_basin_ibtracs","getTrajectories_direct","plot_trajectories_direct","plot_trajectories_colored", "plot_trajectories_by_category","plot_density_scatter", "plot_track_density_grid","plot_density_scatter_by_category"]

# Changelog

# 0.2.0: adapted to work with DestinE data governance, working with IFS-NEMO, ICON.
# 0.1.0: Initial version, tested with nextGEMS3 simulation