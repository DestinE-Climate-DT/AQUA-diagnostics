"""AQUA Diagnostics Package"""

from .version import __version__
from .teleconnections import NAO, ENSO, MJO
from .teleconnections import PlotNAO, PlotENSO, PlotMJO
from .timeseries import Gregory, SeasonalCycles, Timeseries, PlotTimeseries, PlotSeasonalCycles, PlotGregory
from .lat_lon_profiles import LatLonProfiles, PlotLatLonProfiles
from .histogram import Histogram, PlotHistogram
from .global_biases import GlobalBiases, PlotGlobalBiases, StatGlobalBiases
from .boxplots import Boxplots, PlotBoxplots
from .ensemble import EnsembleTimeseries, EnsembleLatLon, EnsembleZonal
from .ensemble import PlotEnsembleTimeseries, PlotEnsembleLatLon, PlotEnsembleZonal
from .ensemble import reader_retrieve_and_merge, merge_from_data_files, extract_realizations
from .ecmean import PerformanceIndices, GlobalMean
from .seaice import SeaIce, PlotSeaIce, Plot2DSeaIce
from .sshVariability import sshVariabilityCompute, sshVariabilityPlot
from .ocean_drift import Hovmoller, PlotHovmoller
from .ocean_stratification import Stratification, PlotStratification, PlotMLD
from .ocean_trends import Trends, PlotTrends


DIAGNOSTIC_CONFIG_DIRECTORIES = ["analysis", "tools", "collections"]
DIAGNOSTIC_TEMPLATE_DIRECTORIES = ["collections"]

__all__ = ["NAO", "ENSO", "MJO",
           "PlotNAO", "PlotENSO", "PlotMJO",
           "Gregory", "SeasonalCycles", "Timeseries",
           "PlotTimeseries", "PlotSeasonalCycles", "PlotGregory",
           "LatLonProfiles", "PlotLatLonProfiles",
           "Histogram", "PlotHistogram",
           "GlobalBiases", "PlotGlobalBiases",
           "reader_retrieve_and_merge", "merge_from_data_files",
           "EnsembleTimeseries", "EnsembleLatLon", "EnsembleZonal",
           "PlotEnsembleTimeseries", "PlotEnsembleLatLon", "PlotEnsembleZonal",
           "PerformanceIndices", "GlobalMean",
           "SeaIce", "PlotSeaIce", "Plot2DSeaIce",
           "sshVariabilityCompute", "sshVariabilityPlot",
           "Boxplots", "PlotBoxplots", "extract_realizations",
           "Hovmoller", "PlotHovmoller", "Stratification", "PlotStratification", "PlotMLD",
           "Trends", "PlotTrends",
           "DIAGNOSTIC_CONFIG_DIRECTORIES", "DIAGNOSTIC_TEMPLATE_DIRECTORIES"]
