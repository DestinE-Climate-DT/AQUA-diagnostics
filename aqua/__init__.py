"""AQUA diagnostics package - provides diagnostic tools"""

# Extend namespace to coexist with aqua-core
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Import and re-export core functionality for backward compatibility
# This ensures 'from aqua import Reader' works from anywhere
try:
    from aqua.core import *
    from aqua.core import __version__, __all__ as _core_all
except ImportError:
    # aqua-core not installed
    _core_all = []

# Import diagnostics functionality
try:
    from aqua.diagnostics import __all__ as _diag_all
except ImportError:
    _diag_all = []

# Combine all exports
__all__ = list(set(_core_all + _diag_all))