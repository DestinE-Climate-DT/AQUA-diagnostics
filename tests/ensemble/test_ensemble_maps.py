"""
Test ensemble Ensemble module
"""

import os

import pytest
import xarray as xr

from aqua.diagnostics import (
    EnsembleMaps,
    PlotEnsembleMaps,
)
from tests.shared_constants import (
    APPROX_REL,
    LOGLEVEL,
)

# Tolerance and Logging
approx_rel = APPROX_REL
loglevel = LOGLEVEL

# pytestmark groups tests
pytestmark = [pytest.mark.diagnostics]


@pytest.fixture(scope="module")
def ensemble_config():
    """Configuration parameters for the ensemble test."""
    return {
        "var": "2t",
        "catalog_list": ["ci", "ci"],
        "model_list": ["FESOM", "FESOM"],
        "exp_list": ["results", "results"],
        "source_list": ["atmglobalmean2D", "atmglobalmean2D"],
    }


@pytest.fixture
def tmp_path_str(tmp_path):
    """Provide a reliable tmp_path."""
    return str(tmp_path)


def ensemble_maps_instance(ensemble_config, dataset_instance):
    """Create an EnsembleMaps instance."""
    ens = EnsembleMaps(
        var=ensemble_config["var"],
        dataset=dataset_instance,
        catalog_list=ensemble_config["catalog_list"],
        model_list=ensemble_config["model_list"],
        exp_list=ensemble_config["exp_list"],
        source_list=ensemble_config["source_list"],
        ensemble_dimension_name="ensemble",
    )
    ens.run()
    return ens


@pytest.fixture(scope="module")
def plot_ensemble_instance(ensemble_config):
    """Create a PlotEnsembleMaps instance."""
    plot_args = {
        "catalog_list": ensemble_config["catalog_list"],
        "model_list": ensemble_config["model_list"],
        "exp_list": ensemble_config["exp_list"],
        "source_list": ensemble_config["source_list"],
    }
    return PlotEnsembleMaps(**plot_args, outputdir="./")


class TestEnsembleMaps:
    """Test suite for EnsembleMaps diagnostic."""

    def test_initialization(self, dataset_instance):
        """Test if data retrieval was successful."""
        assert dataset_instance is not None
        assert isinstance(dataset_instance, xr.Dataset)

    def test_run(self, ensemble_maps_instance, ensemble_config, tmp_path_str):
        """Test the computation and NetCDF output generation."""
        ens = ensemble_maps_instance
        ens.outputdir = tmp_path_str  # Redirect outputs to tmp_path
        conf = ensemble_config

        # Execution
        ens.run()

        # Check attributes
        assert hasattr(ens, "dataset_mean")
        assert hasattr(ens, "dataset_std")

        # Construct filenames
        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        # Check NetCDF outputs
        nc_mean = os.path.join(tmp_path_str, "netcdf", f"ensemble.ensemblemaps.{cat}.{mod}.{exp}.r1.{var}.mean.nc")
        assert os.path.exists(nc_mean)

        nc_std = os.path.join(tmp_path_str, "netcdf", f"ensemble.ensemblemaps.{cat}.{mod}.{exp}.r1.{var}.std.nc")
        assert os.path.exists(nc_std)

    def test_statistics(self, ensemble_maps_instance):
        """Test the statistical correctness of the ensemble."""
        ens = ensemble_maps_instance
        if not hasattr(ens, "dataset_mean"):
            ens.run()

        assert ens.dataset_mean is not None
        assert ens.dataset_std.all() == 0

    def test_plotting(self, ensemble_maps_instance, plot_ensemble_instance, ensemble_config, tmp_path_str):
        """Test the plotting functionality."""
        ens = ensemble_maps_instance
        plot_ens = plot_ensemble_instance
        plot_ens.outputdir = tmp_path_str
        conf = ensemble_config

        # Plot Mean
        plot_args_mean = {
            "var": conf["var"],
            "dpi": 50,
            "save_format": ("png", "pdf"),
            "title": "Test data Mean",
            "cbar_label": "Test Label",
            "dataset": ens.dataset_mean,
            "data_name": "mean",
        }
        fig_m, ax_m = plot_ens.plot(**plot_args_mean)
        assert fig_m is not None

        # Construct filenames
        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        # Check Output Files for Mean
        png_mean = os.path.join(tmp_path_str, "png", f"ensemble.ensemblemaps.{cat}.{mod}.{exp}.r1.{var}.mean.png")
        assert os.path.exists(png_mean)
