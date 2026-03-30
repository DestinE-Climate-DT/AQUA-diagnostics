"""ssh module"""

# To enable: from ssh import class sshVariability

from .plot_ssh_variability import SshVariabilityPlot
from .ssh_variability import SshVariabilityCompute

# This specifies which methods are exported publicly, used by "from ssh_class *"
__all__ = ["SshVariabilityCompute", "sshVariablityPlot"]
