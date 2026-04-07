""" The tropical rainfall module"""

from .src.tropical_rainfall_tools import ToolsClass
from .src.tropical_rainfall_plots import PlottingClass
from .src.tropical_rainfall_main import MainClass
from .src.tropical_rainfall_meta import MetaClass
from .tropical_rainfall_class import TropicalRainfall

__version__ = '0.0.1'

__all__ = ['TropicalRainfall', 'PlottingClass', 'ToolsClass', 'MainClass', 'MetaClass']

# Change log
# 0.0.1: Initial version
