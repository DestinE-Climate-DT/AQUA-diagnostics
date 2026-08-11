"""Test ensemble Ensemble module"""

import os
import pytest
import xarray as xr

from aqua.diagnostics import (
    EnsembleTimeseries,
    PlotEnsembleTimeseries,
)
from aqua.diagnostics.ensemble.util import (
    reader_retrieve_and_merge,
)
from tests.shared_constants import (
    APPROX_REL,
    DPI,
    LOGLEVEL,
)

approx_rel = APPROX_REL
loglevel = LOGLEVEL
pytestmark = [pytest.mark.diagnostics]


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


@pytest.fixture
def tmp_path_str(tmp_path):
    """Provide reliable tmp_path."""
    return str(tmp_path)


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
def ensemble_ts_instance(ts_config, ts_dataset):
    """Create an EnsembleTimeseries instance."""
    ts = EnsembleTimeseries(
        var=ts_config["var"],
        monthly_data=ts_dataset,
        annual_data=ts_dataset,
        catalog_list=ts_config["catalog_list"],
        model_list=ts_config["model_list"],
        exp_list=ts_config["exp_list"],
        source_list=ts_config["source_list"],
        ensemble_dimension_name="ensemble",
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
        assert ts_dataset is not None
        assert isinstance(ts_dataset, xr.Dataset)

    def test_run(self, ensemble_ts_instance, ts_config, tmp_path_str):
        ts = ensemble_ts_instance
        ts.outputdir = tmp_path_str
        conf = ts_config
        outdir = ts.outputdir

        ts.run()

        assert hasattr(ts, "monthly_data_mean")
        assert hasattr(ts, "annual_data_mean")

        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        nc_monthly = os.path.join(
            tmp_path_str, "netcdf", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.monthly.nc"
        )
        assert os.path.exists(nc_monthly)

    def test_statistics(self, ensemble_ts_instance):
        ts = ensemble_ts_instance

        if getattr(ts, "monthly_data_mean", None) is None:
            ts.run()

        assert ts.monthly_data_mean is not None
        assert ts.annual_data_mean is not None
        assert ts.monthly_data_std.values.all() == 0

    def test_plotting(self, ensemble_ts_instance, plot_ts_instance, ts_config, tmp_path_str):
        ts = ensemble_ts_instance
        plot_ts = plot_ts_instance
        plot_ts.outputdir = tmp_path_str
        conf = ts_config
        outdir = ts.outputdir

        plot_arguments = {
            "var": conf["var"],
            "save_format": ("png", "pdf"),
            "plot_ensemble_members": True,
            "title": "test timeseries data",
            "monthly_data": ts.monthly_data,
            "monthly_data_mean": ts.monthly_data_mean,
            "monthly_data_std": ts.monthly_data_mean,
            "annual_data": ts.annual_data,
            "annual_data_mean": ts.annual_data_mean,
            "annual_data_std": ts.annual_data_mean,
            "dpi": DPI,
        }

        fig, ax = plot_ts.plot(**plot_arguments)

        assert fig is not None
        assert ax is not None

        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        png_file = os.path.join(tmp_path_str, "png", f"ensemble.ensembletimeseries.{cat}.{mod}.{exp}.r1.{var}.mean.png")
        assert os.path.exists(png_file)
