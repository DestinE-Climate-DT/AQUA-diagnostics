"""AQUA Diagnostics Package"""

from .version import __version__
from .teleconnections import NAO, ENSO, MJO
from .teleconnections import PlotNAO, PlotENSO, PlotMJO
from .timeseries import Gregory, SeasonalCycles, Timeseries, PlotTimeseries, PlotSeasonalCycles, PlotGregory
from .lat_lon_profiles import LatLonProfiles, PlotLatLonProfiles
from .histogram import Histogram, PlotHistogram
from .biases import Climatology, PlotClimatology, PlotBias, StatBias
from .boxplots import Boxplots, PlotBoxplots
from .ensemble import EnsembleTimeseries, EnsembleMaps, EnsembleZonal
from .ensemble import PlotEnsembleTimeseries, PlotEnsembleMaps, PlotEnsembleZonal
from .ensemble import reader_retrieve_and_merge, merge_from_data_files, extract_realizations
from .ecmean import PerformanceIndices, GlobalMean
from .seaice import SeaIce, PlotSeaIce, Plot2DSeaIce
from .sshVariability import SshVariabilityCompute, SshVariabilityPlot
from .ocean_drift import Hovmoller, PlotHovmoller
from .ocean_stratification import Stratification, PlotStratification, PlotMLD
from .ocean_trends import PlotTrends as PlotOceanTrends
from .ocean_trends import Trends as OceanTrends
from .trends import Trends, PlotTrends


# Folders installed in the AQUA configuration folder (plugin contract version 2 of aqua-core), by owner:
# project folders in ~/.aqua/<dir>/diagnostics, exclusive folders in ~/.aqua/<dir> (owned by diagnostics alone),
# shared folders in ~/.aqua/<dir>/<entry>, where each entry sits next to those of core and of the other projects
DIAGNOSTIC_PROJECT_DIRECTORIES = ["analysis", "collections"]
DIAGNOSTIC_EXCLUSIVE_DIRECTORIES = ["definitions"]
DIAGNOSTIC_SHARED_DIRECTORIES = ["tools", "templates"]

__all__ = ["__version__", "NAO", "ENSO", "MJO",
           "PlotNAO", "PlotENSO", "PlotMJO",
           "Gregory", "SeasonalCycles", "Timeseries",
           "PlotTimeseries", "PlotSeasonalCycles", "PlotGregory",
           "LatLonProfiles", "PlotLatLonProfiles",
           "Histogram", "PlotHistogram",
           "Climatology", "PlotClimatology", "PlotBias", "StatBias",
           "reader_retrieve_and_merge", "merge_from_data_files",
           "EnsembleTimeseries", "EnsembleMaps", "EnsembleZonal",
           "PlotEnsembleTimeseries", "PlotEnsembleMaps", "PlotEnsembleZonal",
           "PerformanceIndices", "GlobalMean",
           "SeaIce", "PlotSeaIce", "Plot2DSeaIce",
           "SshVariabilityCompute", "SshVariabilityPlot",
           "Boxplots", "PlotBoxplots", "extract_realizations",
           "Hovmoller", "PlotHovmoller", "Stratification", "PlotStratification", "PlotMLD",
           "Trends", "PlotTrends", "PlotOceanTrends", "OceanTrends",
           "DIAGNOSTIC_PROJECT_DIRECTORIES", "DIAGNOSTIC_EXCLUSIVE_DIRECTORIES", "DIAGNOSTIC_SHARED_DIRECTORIES"]


def get_install_dirs():
    """Folders installed by ``aqua install`` (entry point ``diagnostics`` of ``aqua.plugins``, contract version 2)."""
    return {
        "contract": 2,
        "project": DIAGNOSTIC_PROJECT_DIRECTORIES,
        "exclusive": DIAGNOSTIC_EXCLUSIVE_DIRECTORIES,
        "shared": DIAGNOSTIC_SHARED_DIRECTORIES,
    }
