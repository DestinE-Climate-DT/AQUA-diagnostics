"""AQUA diagnostics package"""

# Extend namespace to coexist with aqua-core
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# This ensures 'from aqua import Reader' works from anywhere
try:
    from .core import *
    from .core import __version__, __all__
except ImportError:
    pass
