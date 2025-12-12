"""Fixtures for Global Biases CLI tests."""

import pytest
from types import SimpleNamespace
from aqua.core.util import dump_yaml
from aqua.diagnostics.core.cli_base import DiagnosticCLI

# ============================================================================
# Test Data Builders & Constants
# ============================================================================

DEFAULT_DATASET = {
    'catalog': 'ci',
    'model': 'ERA5',
    'exp': 'era5-hpz3',
    'source': 'monthly',
    'regrid': 'r100',
    'startdate': '1990-01-01',
    'enddate': '1990-12-31',
}

DEFAULT_OUTPUT = {
    'outputdir': '/tmp/output',
    'rebuild': True,
    'save_netcdf': True,
    'save_pdf': True,
    'save_png': True,
    'dpi': 50,
    'create_catalog_entry': False,
}

def build_dataset_config(**overrides):
    """Build a dataset configuration dict."""
    return {**DEFAULT_DATASET, **overrides}

def build_tool_config(**overrides):
    """Build the 'globalbiases' portion of the config."""
    defaults = {
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
    
    # Merge for params and plot_params if provided
    if 'params' in overrides:
        defaults['params'].update(overrides.pop('params'))
    if 'plot_params' in overrides:
        defaults['plot_params'].update(overrides.pop('plot_params'))
        
    defaults.update(overrides)
    return defaults

def build_full_config(tool_config=None, datasets=None, references=None, output_dir='/tmp/output'):
    """Build a full configuration dictionary."""
    if tool_config is None:
        tool_config = build_tool_config()
    if datasets is None:
        datasets = [build_dataset_config()]
    if references is None:
        references = [build_dataset_config()]
        
    output = {**DEFAULT_OUTPUT, 'outputdir': output_dir}
        
    return {
        'datasets': datasets,
        'references': references,
        'setup': {'loglevel': 'DEBUG'},
        'output': output,
        'diagnostics': {
            'globalbiases': tool_config
        }
    }

# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def gb_config(tmp_path):
    """Create a standard Global Biases configuration."""
    config = build_full_config(output_dir=str(tmp_path))
    return config

@pytest.fixture
def gb_config_file(gb_config, tmp_path):
    """Write the standard configuration to a YAML file."""
    config_file = tmp_path / "config.yaml"
    dump_yaml(outfile=str(config_file), cfg=gb_config)
    return str(config_file)

@pytest.fixture
def gb_config_seasonal(tmp_path):
    """Create a seasonal Global Biases configuration."""
    tool_config = build_tool_config(params={'default': {'seasons': True}})
    config = build_full_config(tool_config=tool_config, output_dir=str(tmp_path))
    return config

@pytest.fixture
def gb_config_seasonal_file(gb_config_seasonal, tmp_path):
    """Write the seasonal configuration to a YAML file."""
    config_file = tmp_path / "config_seasonal.yaml"
    dump_yaml(outfile=str(config_file), cfg=gb_config_seasonal)
    return str(config_file)

@pytest.fixture
def gb_config_formula(tmp_path):
    """Create a Global Biases configuration with formula."""
    tool_config = build_tool_config(
        variables=[],
        formulae=['tnlwrf+tnswrf'],
        params={
            'tnlwrf+tnswrf': {'short_name': 'tnr', 'long_name': 'Top net radiation'}
        }
    )
    config = build_full_config(tool_config=tool_config, output_dir=str(tmp_path))
    return config

@pytest.fixture
def gb_config_formula_file(gb_config_formula, tmp_path):
    """Write the formula configuration to a YAML file."""
    config_file = tmp_path / "config_formula.yaml"
    dump_yaml(outfile=str(config_file), cfg=gb_config_formula)
    return str(config_file)

# ============================================================================
# Mock CLI (for Executor Tests)
# ============================================================================

@pytest.fixture
def mock_cli_global_biases(mock_cli, gb_config):
    """
    Extend the generic mock_cli with Global Biases-specific configuration.
    This mimics a DiagnosticCLI that has already loaded the config.
    """
    mock_cli.config_dict = gb_config
    
    # Simple implementation of dataset_args matching build_dataset_config defaults
    def _dataset_args(dataset_dict):
        # In a real CLI, this merges default args, but here we just return the dict
        # plus ensuring default keys exist if they were omitted in the dict
        defaults = build_dataset_config()
        return {**defaults, **dataset_dict}
        
    mock_cli.dataset_args = _dataset_args
    return mock_cli

@pytest.fixture
def patched_global_biases_classes(patch_diagnostic_classes):
    """Patch GlobalBiases and PlotGlobalBiases classes."""
    return patch_diagnostic_classes(
        'aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases',
        'aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases'
    )

@pytest.fixture
def setup_mock_gb(setup_mock_diagnostic_base, mocker):
    """Setup mock GlobalBiases instance with data."""
    def _setup(vars_dict=None, has_plev=False, with_seasonal=False):
        if vars_dict is None:
            vars_dict = {'2t': ['lat', 'lon']}
        
        if has_plev:
            vars_dict = {var: ['plev', 'lat', 'lon'] for var in vars_dict}
            
        extra_attrs = {
            'climatology': mocker.Mock(),
            'seasonal_climatology': mocker.Mock() if with_seasonal else None,
            # Add retrieve method explicitly if needed, though usually masked by the class mock
            'retrieve': mocker.Mock(),
            'compute_climatology': mocker.Mock()
        }
        
        return setup_mock_diagnostic_base(vars_dict, **extra_attrs)
    return _setup

# ============================================================================
# CLI Integration Helpers
# ============================================================================

@pytest.fixture
def cli_args_maker():
    """Factory for CLI arguments."""
    def _make_args(config_file, **overrides):
        defaults = {
            'config': config_file,
            'loglevel': 'DEBUG',
            'catalog': None,
            'model': None,
            'exp': None,
            'source': None,
            'regrid': None,
            'outputdir': None,
            'startdate': None,
            'enddate': None,
            'realization': None,
            'nworkers': None,
            'cluster': None
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)
    return _make_args

@pytest.fixture
def prepare_cli(cli_args_maker):
    """Factory to create and prepare a DiagnosticCLI instance."""
    def _prepare(config_file, **overrides):
        args = cli_args_maker(config_file, **overrides)
        cli = DiagnosticCLI(
            args=args,
            diagnostic_name='globalbiases',
            default_config='config_global_biases.yaml'
        )
        cli.prepare()
        return cli
    return _prepare
