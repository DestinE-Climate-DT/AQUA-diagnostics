"""Test ensemble Ensemble module"""

import os

import pytest
import xarray as xr

from aqua.diagnostics import EnsembleTimeseries, PlotEnsembleTimeseries
from aqua.diagnostics.ensemble.util import reader_retrieve_and_merge
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL

# Tolerance and Logging
approx_rel = APPROX_REL
loglevel = LOGLEVEL

# pytestmark groups tests
pytestmark = [pytest.mark.diagnostics]


# Module-level fixtures
@pytest.fixture(scope="module")
def ts_config():
    """Configuration parameters for the timeseries test."""
    return {
        "var": "2t",
        "catalog_list": ["ci", "ci"],
        "model_list": ["FESOM", "FESOM"],
        "exp_list": ["results", "results"],
        "source_list": ["timeseries1D", "timeseries1D"],
    }


@pytest.fixture(scope="module")
def ts_dataset(ts_config):
    """Retrieve and merge data once for the module."""
    dataset = reader_retrieve_and_merge(
        variable=ts_config["var"],
        catalog_list=ts_config["catalog_list"],
        model_list=ts_config["model_list"],
        exp_list=ts_config["exp_list"],
        source_list=ts_config["source_list"],
        loglevel=loglevel,
        ens_dim="ensemble",
    )
    return dataset


@pytest.fixture(scope="module")
def ensemble_ts_instance(ts_config, ts_dataset, tmp_path_factory):
    """Create an EnsembleTimeseries instance with statistics already computed."""
    outputdir = str(tmp_path_factory.mktemp("output"))
    ts = EnsembleTimeseries(
        var=ts_config["var"],
        monthly_data=ts_dataset,
        annual_data=ts_dataset,
        catalog_list=ts_config["catalog_list"],
        model_list=ts_config["model_list"],
        exp_list=ts_config["exp_list"],
        source_list=ts_config["source_list"],
        ensemble_dimension_name="ensemble",
        outputdir=outputdir,
    )
    ts.run()
    return ts


@pytest.fixture(scope="module")
def plot_ts_instance(ts_config, ensemble_ts_instance):
    """Create a PlotEnsembleTimeseries instance."""
    plot_args = {
        "catalog_list": ts_config["catalog_list"],
        "model_list": ts_config["model_list"],
        "exp_list": ts_config["exp_list"],
        "source_list": ts_config["source_list"],
    }
    return PlotEnsembleTimeseries(**plot_args, outputdir=ensemble_ts_instance.outputdir)


class TestEnsembleTimeseries:
    """Test suite for EnsembleTimeseries diagnostic."""

    def test_initialization(self, ts_dataset):
        """Test if data retrieval was successful."""
        assert ts_dataset is not None
        assert isinstance(ts_dataset, xr.Dataset)

    def test_run(self, ensemble_ts_instance, ts_config):
        """Test the computation and NetCDF output generation."""
        ts = ensemble_ts_instance
        conf = ts_config
        outdir = ts.outputdir

        assert ts.monthly_data_mean is not None
        assert ts.annual_data_mean is not None

        # Construct filenames
        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        # Check NetCDF outputs (Monthly and Annual)
        nc_monthly = os.path.join(outdir, "netcdf", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.monthly.nc")
        assert os.path.exists(nc_monthly)

        nc_annual = os.path.join(outdir, "netcdf", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.annual.nc")
        assert os.path.exists(nc_annual)

    def test_statistics(self, ensemble_ts_instance):
        """Test the statistical correctness of the ensemble."""
        ts = ensemble_ts_instance

        assert ts.monthly_data_mean is not None
        assert ts.annual_data_mean is not None

        # Test if variance is zero (since inputs are identical)
        assert ts.monthly_data_std.values.all() == 0
        assert ts.annual_data_std.values.all() == 0

    def test_plotting(self, ensemble_ts_instance, plot_ts_instance, ts_config):
        """Test the plotting functionality."""
        ts = ensemble_ts_instance
        plot_ts = plot_ts_instance
        conf = ts_config
        outdir = ts.outputdir

        # STD values are zero. Using mean value as std to test visualization pipeline
        plot_arguments = {
            "var": conf["var"],
            "save_format": ("png", "pdf"),
            "plot_ensemble_members": True,
            "title": "test timeseries data",
            "monthly_data": ts.monthly_data,
            "monthly_data_mean": ts.monthly_data_mean,
            "monthly_data_std": ts.monthly_data_mean,  # Artificial STD
            "annual_data": ts.annual_data,
            "annual_data_mean": ts.annual_data_mean,
            "annual_data_std": ts.annual_data_mean,  # Artificial STD
            "ref_monthly_data": ts.monthly_data_mean,
            "ref_annual_data": ts.annual_data_mean,
            "dpi": DPI,
        }

        # The plotting method returns a tuple (fig, ax)
        fig, ax = plot_ts.plot(**plot_arguments)

        assert fig is not None
        assert ax is not None

        # Construct filenames
        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        # Check Output Files
        png_file = os.path.join(outdir, "png", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.png")
        assert os.path.exists(png_file)

        pdf_file = os.path.join(outdir, "pdf", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.pdf")
        assert os.path.exists(pdf_file)
