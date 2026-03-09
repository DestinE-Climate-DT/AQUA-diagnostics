"""ssh module"""

# To enable: from ssh import class ssh_variability

from .plot_ssh_variability import ssh_variability_plot
from .ssh_variability import ssh_variability_compute

# This specifies which methods are exported publicly, used by "from ssh_class *"
__all__ = ["ssh_variability_compute", "sshVariablityPlot"]
