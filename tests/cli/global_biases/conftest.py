"""Fixtures for Global Biases CLI tests."""

import pytest
from types import SimpleNamespace
from aqua.diagnostics.base.util import deep_update

# ============================================================================
# Set up Data Config
# ============================================================================

def build_config(tool_overrides=None, output_dir='/tmp/output'):
    """
    Builds a full config dict for Global Biases with optional overrides.
    
    Args:
        tool_overrides (dict): Overrides for the 'globalbiases' section.
        output_dir (str): Output directory path. Default is '/tmp/output'.
    """
    tool_defaults = {
        'run': True, 
        'diagnostic_name': 'globalbiases', 
        'variables': ['2t'], 
        'formulae': [],
        'params': {
            'default': {
                'plev': None, 
                'seasons': False, 
                'seasons_stat': 'mean',
                'vertical': False
            },
            '2t': {
                'units': 'K', 
                'long_name': '2-meter temperature'
            }
        },
        'plot_params': {
            'default': {
                'projection': 'robinson',
                'projection_params': {}
            },
            '2t': {
                'vmin': -5, 
                'vmax': 5, 
                'cmap': 'RdBu_r'
            }
        }
    }
    
    # Merge tool overrides recursively
    if tool_overrides:
        deep_update(tool_defaults, tool_overrides)

    config = {
        'datasets': [{
            'catalog': 'ci',  
            'model': 'ERA5', 
            'exp': 'era5-hpz3', 
            'source': 'monthly', 
            'regrid': 'r100',
            'startdate': '1990-01-01',
            'enddate': '1990-12-31'
        }],
        'references': [{
            'catalog': 'ci', 
            'model': 'ERA5', 
            'exp': 'era5-hpz3', 
            'source': 'monthly',
            'startdate': '1990-01-01',
            'enddate': '1990-12-31'
        }],
        'setup': {'loglevel': 'DEBUG'},
        'output': {
            'outputdir': output_dir, 
            'save_netcdf': True,
            'save_pdf': True,
            'save_png': True,
            'dpi': 50,
            'create_catalog_entry': False
        },
        'diagnostics': {'globalbiases': tool_defaults}
    }
    
    return config

# ============================================================================
# Mock GlobalBiases and PlotGlobalBiases classes
# ============================================================================

@pytest.fixture
def mock_gb(mocker):
    """
    Patches GlobalBiases classes and returns a namespace with access to mocks.
    Pre-configures the instance with standard successful defaults.
    """
    # Patch the classes of the current diagnostic
    gb_cls = mocker.patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    gb_plot_cls = mocker.patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    
    # Configure the default instance to avoid repetition in tests
    gb_instance = gb_cls.return_value
    
    # Default data structure
    gb_instance.data = {'2t': mocker.Mock(dims=['lat', 'lon'])}
    gb_instance.climatology = mocker.Mock()
    gb_instance.seasonal_climatology = None
    
    # Mock methods that don't return anything but need to be called
    gb_instance.retrieve = mocker.Mock()
    gb_instance.compute_climatology = mocker.Mock()
    
    # Return a SimpleNamespace for clean dot-notation access in tests
    return SimpleNamespace(cls=gb_cls, plot_cls=gb_plot_cls, instance=gb_instance)
