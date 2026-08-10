"""Test ensemble Ensemble module"""

import os
import pytest
import xarray as xr

from aqua.diagnostics import EnsembleZonal, PlotEnsembleZonal
from aqua.diagnostics.ensemble.util import reader_retrieve_and_merge
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL

approx_rel = APPROX_REL
loglevel = LOGLEVEL
pytestmark = [pytest.mark.diagnostics]


@pytest.fixture(scope="module")
def zonal_config():
    """Configuration parameters for the zonal test."""
    return {
        "var": "avg_so",
        "catalog_list": ["ci", "ci"],
        "model_list": ["NEMO", "NEMO"],
        "exp_list": ["results", "results"],
        "source_list": ["zonal_mean-latlev", "zonal_mean-latlev"],
    }


@pytest.fixture
def tmp_path_str(tmp_path):
    return str(tmp_path)


@pytest.fixture(scope="module")
def zonal_dataset(zonal_config):
    dataset = reader_retrieve_and_merge(
        variable=zonal_config["var"],
        catalog_list=zonal_config["catalog_list"],
        model_list=zonal_config["model_list"],
        exp_list=zonal_config["exp_list"],
        source_list=zonal_config["source_list"],
        realization=None,
        loglevel=loglevel,
        ens_dim="ensemble",
    )
    return dataset


@pytest.fixture(scope="module")
def ensemble_zonal_instance(zonal_config, zonal_dataset):
    ens = EnsembleZonal(
        var=zonal_config["var"],
        dataset=zonal_dataset,
        catalog_list=zonal_config["catalog_list"],
        model_list=zonal_config["model_list"],
        exp_list=zonal_config["exp_list"],
        source_list=zonal_config["source_list"],
        ensemble_dimension_name="ensemble",
        outputdir=outputdir,
    )
    ens.run()
    return ens


@pytest.fixture(scope="module")
def plot_zonal_instance(zonal_config):
    plot_args = {
        "catalog_list": zonal_config["catalog_list"],
        "model_list": zonal_config["model_list"],
        "exp_list": zonal_config["exp_list"],
        "source_list": zonal_config["source_list"],
    }
    return PlotEnsembleZonal(**plot_args, outputdir=ensemble_zonal_instance.outputdir)


class TestEnsembleZonal:
    """Test suite for EnsembleZonal diagnostic."""

    def test_initialization(self, zonal_dataset):
        assert zonal_dataset is not None
        assert isinstance(zonal_dataset, xr.Dataset)

    def test_run(self, ensemble_zonal_instance, zonal_config, tmp_path_str):
        ens = ensemble_zonal_instance
        ens.outputdir = tmp_path_str
        conf = zonal_config
        outdir = ens.outputdir

        ens.run()

        assert ens.dataset_mean is not None
        assert ens.dataset_std is not None

        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        nc_mean = os.path.join(tmp_path_str, "netcdf", f"ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.mean.nc")
        assert os.path.exists(nc_mean)

    def test_statistics(self, ensemble_zonal_instance):
        ens = ensemble_zonal_instance

        if ens.dataset_mean is None or ens.dataset_std is None:
            ens.run()

        assert ens.dataset_mean is not None
        assert ens.dataset_std.all() == 0

    def test_plotting(self, ensemble_zonal_instance, plot_zonal_instance, zonal_config, tmp_path_str):
        ens = ensemble_zonal_instance
        plot_ens = plot_zonal_instance
        plot_ens.outputdir = tmp_path_str
        conf = zonal_config
        outdir = ens.outputdir

        plot_arguments = {
            "var": conf["var"],
            "save_format": ("png", "pdf"),
            "title": "Test data",
            "cbar_label": "Test Label",
            "dataset": ens.dataset_mean,
            "data_name": "mean",
            "dpi": DPI,
        }

        fig, ax = plot_ens.plot(**plot_arguments)

        assert fig is not None

        cat, mod, exp = conf["catalog_list"][0], conf["model_list"][0], conf["exp_list"][0]
        var = conf["var"]

        png_mean = os.path.join(tmp_path_str, "png", f"ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.mean.png")
        assert os.path.exists(png_mean)
