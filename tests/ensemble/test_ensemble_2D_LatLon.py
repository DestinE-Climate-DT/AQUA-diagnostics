"""Test ensemble Ensemble module"""
import pytest
import os
import xarray as xr
from aqua.diagnostics import EnsembleLatLon, PlotEnsembleLatLon
from aqua.diagnostics.ensemble.util import reader_retrieve_and_merge
from conftest import APPROX_REL, DPI, LOGLEVEL

# Tolerance and Logging
approx_rel = APPROX_REL
loglevel = LOGLEVEL

# pytestmark groups tests
pytestmark = [
    pytest.mark.ensemble
]

# Module-level fixtures to handle data and configuration
@pytest.fixture(scope="module")
def ensemble_config():
    """Configuration parameters for the ensemble test."""
    return {
        'var': '2t',
        'catalog_list': ['ci', 'ci'],
        'model_list': ['FESOM', 'FESOM'],
        'exp_list': ['results', 'results'],
        'source_list': ['atmglobalmean2D', 'atmglobalmean2D']
    }

@pytest.fixture
def tmp_path_str():
    """Provide consistent tmp_path as string."""
    return "./"

@pytest.fixture(scope="module")
def dataset_instance(ensemble_config):
    """Retrieve and merge data once for the module."""
    dataset = reader_retrieve_and_merge(
        variable=ensemble_config['var'], 
        catalog_list=ensemble_config['catalog_list'], 
        model_list=ensemble_config['model_list'], 
        exp_list=ensemble_config['exp_list'], 
        source_list=ensemble_config['source_list'], 
        loglevel=loglevel,
        ens_dim="ensemble",
    )
    return dataset

@pytest.fixture(scope="module")
def ensemble_latlon_instance(ensemble_config, dataset_instance):
    """Create an EnsembleLatLon instance."""
    # Note: outputdir is set to current dir here, but tests can override or check relative paths
    ens = EnsembleLatLon(
        var=ensemble_config['var'],
        dataset=dataset_instance,
        catalog_list=ensemble_config['catalog_list'],
        model_list=ensemble_config['model_list'],
        exp_list=ensemble_config['exp_list'],
        source_list=ensemble_config['source_list'],
        ensemble_dimension_name="ensemble",
        outputdir="./", 
    )
    return ens

@pytest.fixture(scope="module")
def plot_ensemble_instance(ensemble_config):
    """Create a PlotEnsembleLatLon instance."""
    plot_args = {
        "catalog_list": ensemble_config['catalog_list'],
        "model_list": ensemble_config['model_list'],
        "exp_list": ensemble_config['exp_list'],
        "source_list": ensemble_config['source_list'],
    }
    return PlotEnsembleLatLon(**plot_args, outputdir="./")


class TestEnsembleLatLon:
    """Test suite for EnsembleLatLon diagnostic."""

    def test_initialization(self, dataset_instance):
        """Test if data retrieval was successful."""
        assert dataset_instance is not None
        assert isinstance(dataset_instance, xr.Dataset)

    def test_run(self, ensemble_latlon_instance, ensemble_config, tmp_path_str):
        """Test the computation and NetCDF output generation."""
        ens = ensemble_latlon_instance
        conf = ensemble_config
        
        # execution
        ens.run()

        # Check attributes
        assert hasattr(ens, "dataset_mean")
        assert hasattr(ens, "dataset_std")

        # Construct filenames based on the first element of the config lists (as per original logic)
        cat, mod, exp = conf['catalog_list'][0], conf['model_list'][0], conf['exp_list'][0]
        var = conf['var']

        # Check NetCDF outputs
        nc_mean = os.path.join(tmp_path_str, 'netcdf', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.mean.nc')
        assert os.path.exists(nc_mean)

        nc_std = os.path.join(tmp_path_str, 'netcdf', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.std.nc')
        assert os.path.exists(nc_std)

    def test_statistics(self, ensemble_latlon_instance):
        """Test the statistical correctness of the ensemble."""
        ens = ensemble_latlon_instance
        
        # Ensure run() has been called (handled by module scope order, but safe to check)
        if not hasattr(ens, 'dataset_mean'):
            ens.run()

        # test if mean is non-zero and variance is zero (since inputs are identical)
        assert ens.dataset_mean is not None
        assert ens.dataset_std.all() == 0

    def test_plotting(self, ensemble_latlon_instance, plot_ensemble_instance, ensemble_config, tmp_path_str):
        """Test the plotting functionality."""
        ens = ensemble_latlon_instance
        plot_ens = plot_ensemble_instance
        conf = ensemble_config
        
        if not hasattr(ens, 'dataset_mean'):
            ens.run()

        # STD values are zero. Using mean value as std to test implementation visualization
        plot_arguments = {
            "var": conf['var'],
            "dpi": 50, # Low DPI for testing speed
            "save_pdf": True,
            "save_png": True,
            "title_mean": "Test data",
            "title_std": "Test data",
            "cbar_label": "Test Label",
            "dataset_mean": ens.dataset_mean,
            "dataset_std": ens.dataset_mean, 
        }

        plot_dict = plot_ens.plot(**plot_arguments)
        
        assert plot_dict['mean_plot'][0] is not None

        # Construct filenames
        cat, mod, exp = conf['catalog_list'][0], conf['model_list'][0], conf['exp_list'][0]
        var = conf['var']

        # Check PNGs
        png_mean = os.path.join(tmp_path_str, 'png', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.mean.png')
        assert os.path.exists(png_mean)
        
        png_std = os.path.join(tmp_path_str, 'png', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.std.png')
        assert os.path.exists(png_std)

        # Check PDFs
        pdf_mean = os.path.join(tmp_path_str, 'pdf', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.mean.pdf')
        assert os.path.exists(pdf_mean)

        pdf_std = os.path.join(tmp_path_str, 'pdf', f'ensemble.ensemblelatlon.{cat}.{mod}.{exp}.r1.{var}.std.pdf')
        assert os.path.exists(pdf_std)
