"""Test ensemble Ensemble module"""
import pytest
import os
import xarray as xr
from aqua.diagnostics import EnsembleZonal, PlotEnsembleZonal
from aqua.diagnostics.ensemble.util import reader_retrieve_and_merge
from conftest import APPROX_REL, DPI, LOGLEVEL

# Tolerance and Logging
approx_rel = APPROX_REL
loglevel = LOGLEVEL

# pytestmark groups tests
pytestmark = [
    pytest.mark.diagnostics
]

# Module-level fixtures
@pytest.fixture(scope="module")
def zonal_config():
    """Configuration parameters for the zonal test."""
    return {
        'var': 'avg_so',
        'catalog_list': ['ci', 'ci'],
        'model_list': ['NEMO', 'NEMO'],
        'exp_list': ['results', 'results'],
        'source_list': ['zonal_mean-latlev', 'zonal_mean-latlev']
    }

@pytest.fixture
def tmp_path_str():
    """Provide consistent tmp_path as string."""
    return "./"

@pytest.fixture(scope="module")
def zonal_dataset(zonal_config):
    """Retrieve and merge data once for the module."""
    dataset = reader_retrieve_and_merge(
        variable=zonal_config['var'],
        catalog_list=zonal_config['catalog_list'],
        model_list=zonal_config['model_list'],
        exp_list=zonal_config['exp_list'],
        source_list=zonal_config['source_list'],
        realization=None,
        loglevel=loglevel,
        ens_dim="ensemble"
    )
    return dataset

@pytest.fixture(scope="module")
def ensemble_zonal_instance(zonal_config, zonal_dataset):
    """Create an EnsembleZonal instance."""
    ens = EnsembleZonal(
        var=zonal_config['var'],
        dataset=zonal_dataset,
        catalog_list=zonal_config['catalog_list'],
        model_list=zonal_config['model_list'],
        exp_list=zonal_config['exp_list'],
        source_list=zonal_config['source_list'],
        ensemble_dimension_name="ensemble",
        outputdir="./",
    )
    return ens

@pytest.fixture(scope="module")
def plot_zonal_instance(zonal_config):
    """Create a PlotEnsembleZonal instance."""
    plot_args = {
        "catalog_list": zonal_config['catalog_list'],
        "model_list": zonal_config['model_list'],
        "exp_list": zonal_config['exp_list'],
        "source_list": zonal_config['source_list'],
    }
    return PlotEnsembleZonal(**plot_args, outputdir="./")


class TestEnsembleZonal:
    """Test suite for EnsembleZonal diagnostic."""

    def test_initialization(self, zonal_dataset):
        """Test if data retrieval was successful."""
        assert zonal_dataset is not None
        assert isinstance(zonal_dataset, xr.Dataset)

    def test_run(self, ensemble_zonal_instance, zonal_config, tmp_path_str):
        """Test the computation and NetCDF output generation."""
        ens = ensemble_zonal_instance
        conf = zonal_config

        # Execution
        ens.run()

        # Check attributes exist
        assert hasattr(ens, "dataset_mean")
        assert hasattr(ens, "dataset_std")

        # Construct filenames
        cat, mod, exp = conf['catalog_list'][0], conf['model_list'][0], conf['exp_list'][0]
        var = conf['var']

        # Check NetCDF outputs
        nc_mean = os.path.join(tmp_path_str, 'netcdf', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.mean.nc')
        assert os.path.exists(nc_mean)

        nc_std = os.path.join(tmp_path_str, 'netcdf', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.std.nc')
        assert os.path.exists(nc_std)

    def test_statistics(self, ensemble_zonal_instance):
        """Test the statistical correctness of the ensemble."""
        ens = ensemble_zonal_instance

        # Ensure run() has been called
        if not hasattr(ens, 'dataset_mean'):
            ens.run()

        # Test if mean is present and std is zero (identical inputs)
        assert ens.dataset_mean is not None
        assert ens.dataset_std.all() == 0

    def test_plotting(self, ensemble_zonal_instance, plot_zonal_instance, zonal_config, tmp_path_str):
        """Test the plotting functionality."""
        ens = ensemble_zonal_instance
        plot_ens = plot_zonal_instance
        conf = zonal_config

        if not hasattr(ens, 'dataset_mean'):
            ens.run()

        # STD values are zero. Using mean value as std to test visualization pipeline (consistent with comments)
        plot_arguments = {
            "var": conf['var'],
            "save_pdf": True,
            "save_png": True,
            "title_mean": "Test data",
            "title_std": "Test data",
            "cbar_label": "Test Label",
            "dataset_mean": ens.dataset_mean,
            "dataset_std": ens.dataset_mean, # Using mean as proxy for std to ensure valid plot generation
            "dpi": DPI,
        }

        plot_dict = plot_ens.plot(**plot_arguments)

        assert plot_dict['mean_plot'][0] is not None

        # Construct filenames
        cat, mod, exp = conf['catalog_list'][0], conf['model_list'][0], conf['exp_list'][0]
        var = conf['var']

        # Check Output Files
        png_mean = os.path.join(tmp_path_str, 'png', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.mean.png')
        assert os.path.exists(png_mean)

        png_std = os.path.join(tmp_path_str, 'png', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.std.png')
        assert os.path.exists(png_std)

        pdf_mean = os.path.join(tmp_path_str, 'pdf', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.mean.pdf')
        assert os.path.exists(pdf_mean)

        pdf_std = os.path.join(tmp_path_str, 'pdf', f'ensemble.ensemblezonal.{cat}.{mod}.{exp}.r1.{var}.std.pdf')
        assert os.path.exists(pdf_std)
