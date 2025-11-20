"""plugin of aqua-diagnostics for AQUA console"""
from importlib import resources

def get_install_paths():
    """
    Return a dictionary describing what folders should be installed in ~/.aqua.
    The framework does not need to know the exact list of diagnostics.
    """
    root = resources.files("aqua.diagnostics") / ".."
    return {
        "diagnostics": root / "diagnostics",
        "templates": root / "templates",
        "config": root / "config",
    }
