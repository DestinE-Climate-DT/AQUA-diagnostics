"""Fixtures for Global Biases CLI tests.

This module extends the generic fixtures from tests/cli/conftest.py
with Global Biases-specific configuration and mocks.

Fixtures are organized into:
1. Mock CLI extension (for executor tests)
2. Mock diagnostic instances (for executor tests)
3. Tool configuration (for executor tests)
4. Config YAML files (for CLI tests)
5. CLI setup helpers (for CLI tests)
"""

import pytest
from aqua.core.util import load_yaml, dump_yaml
from aqua.diagnostics.core.cli_base import DiagnosticCLI
from types import SimpleNamespace

# ============================================================================
# Mock CLI Extension
# ============================================================================

@pytest.fixture
def mock_cli_global_biases(mock_cli):
    """
    Extend the generic mock_cli with Global Biases-specific configuration.
    """
    mock_cli.config_dict = {
        'datasets': [{
            'catalog': 'ci',
            'model': 'ERA5',
            'exp': 'era5-hpz3',
            'source': 'monthly',
            'regrid': 'r100',
        }],
        'references': [{
            'catalog': 'ci',
            'model': 'ERA5',
            'exp': 'era5-hpz3',
            'source': 'monthly',
            'regrid': 'r100',
        }]
    }
    # Global Biases-specific dataset_args implementation
    mock_cli.dataset_args = lambda x: {
        'catalog': x.get('catalog', 'ci'),
        'model': x.get('model', 'ERA5'),
        'exp': x.get('exp', 'era5-hpz3'),
        'source': x.get('source', 'monthly'),
        'regrid': x.get('regrid', 'r100')
    }
    return mock_cli


# ============================================================================
# Mock Diagnostic Instances
# ============================================================================

@pytest.fixture
def patched_global_biases_classes(patch_diagnostic_classes):
    """
    Patch GlobalBiases and PlotGlobalBiases classes for testing.
    
    """
    return patch_diagnostic_classes(
        'aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases',
        'aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases'
    )


@pytest.fixture
def setup_mock_gb(setup_mock_diagnostic_base, mocker):
    """
    Setup mock GlobalBiases instance with data.

    """
    def _setup(vars_dict=None, has_plev=False, with_seasonal=False):
        """Setup mock GlobalBiases with specified variables.
        
        Args:
            vars_dict: Dict mapping var names to dims lists.
                      Defaults to {'2t': ['lat', 'lon']} if None.
            has_plev: If True, all vars get ['plev', 'lat', 'lon'] dimensions.
            with_seasonal: If True, sets seasonal_climatology to a mock object.
        
        Returns:
            Mock GlobalBiases instance with data, climatology, etc.
        """
        # Default to Global Biases common variable
        if vars_dict is None:
            vars_dict = {'2t': ['lat', 'lon']}
        
        # Apply has_plev convenience: override all dims to include plev
        if has_plev:
            vars_dict = {
                var: ['plev', 'lat', 'lon']
                for var in vars_dict.keys()
            }
        
        # Build extra attributes
        extra_attrs = {
            'climatology': mocker.Mock(),
            'seasonal_climatology': mocker.Mock() if with_seasonal else None
        }
        
        return setup_mock_diagnostic_base(vars_dict, **extra_attrs)
    return _setup


# ============================================================================
# Tool Configuration
# ============================================================================

@pytest.fixture
def tool_dict_minimal():
    """
    Minimal tool configuration for Global Biases testing.
    """
    return {
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


# ============================================================================
# Config YAML Files (for test_cli_global_biases.py)
# ============================================================================

def _base_dataset_config():
    """Base dataset/reference configuration shared across fixtures."""
    return {
        'catalog': 'ci',
        'model': 'ERA5',
        'exp': 'era5-hpz3',
        'source': 'monthly',
        'regrid': 'r100',
        'startdate': '1990-01-01',
        'enddate': '1990-12-31',
    }


def _base_output_config(tmp_path):
    """Base output configuration shared across fixtures."""
    return {
        'outputdir': str(tmp_path),
        'rebuild': True,
        'save_netcdf': True,
        'save_pdf': True,
        'save_png': True,
        'dpi': 50,
        'create_catalog_entry': False,
    }


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """
    Create a minimal config YAML file for Global Biases CLI testing.
    """
    config = {
        'datasets': [_base_dataset_config()],
        'references': [_base_dataset_config()],
        'setup': {'loglevel': 'DEBUG'},
        'output': _base_output_config(tmp_path),
        'diagnostics': {
            'globalbiases': {
                'run': True,
                'diagnostic_name': 'globalbiases',
                'variables': ['2t'],
                'formulae': [],
                'params': {
                    'default': {
                        'seasons': False,
                        'vertical': False,
                    },
                    '2t': {
                        'standard_name': '2t',
                        'long_name': '2-meter temperature',
                        'units': 'K',
                    }
                },
                'plot_params': {
                    'default': {
                        'projection': 'robinson',
                        'projection_params': {},
                    },
                    '2t': {
                        'vmin': -5,
                        'vmax': 5,
                        'cmap': 'RdBu_r',
                    }
                }
            }
        }
    }
    
    config_file = tmp_path / "test_global_biases_config.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def minimal_config_yaml_with_seasons(minimal_config_yaml, tmp_path):
    """Create config with seasonal analysis enabled (derived from minimal config)."""
    config = load_yaml(minimal_config_yaml)
    config['diagnostics']['globalbiases']['params']['default']['seasons'] = True
    config['diagnostics']['globalbiases']['params']['default']['seasons_stat'] = 'mean'
    
    config_file = tmp_path / "test_global_biases_config_seasonal.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def minimal_config_yaml_with_formula(minimal_config_yaml, tmp_path):
    """Create config with formula for testing formula handling (derived from minimal config)."""
    config = load_yaml(minimal_config_yaml)
    
    # Update diagnostic config for formula
    diag_config = config['diagnostics']['globalbiases']
    diag_config['variables'] = []
    diag_config['formulae'] = ['tnlwrf+tnswrf']
    
    # Update params
    diag_config['params'] = {
        'default': {
            'seasons': False,
            'vertical': False,
        },
        'tnlwrf+tnswrf': {
            'short_name': 'tnr',
            'long_name': 'Top net radiation',
        }
    }
    
    # Update plot_params
    diag_config['plot_params'] = {
        'default': {
            'projection': 'robinson',
            'projection_params': {},
        },
        'tnlwrf+tnswrf': {
            'vmin': -50,
            'vmax': 50,
            'cmap': 'RdBu_r',
        }
    }
    
    config_file = tmp_path / "test_global_biases_config_formula.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


# ============================================================================
# CLI Setup Helpers (for test_cli_global_biases.py)
# ============================================================================

@pytest.fixture
def cli_args_maker():
    """
    Factory fixture to create SimpleNamespace args for CLI tests.
    Creates args with defaults that can be overridden.
    """
    def _create_args(config, **overrides):
        defaults = {
            'config': config,
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
            'cluster': None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)
    
    return _create_args


@pytest.fixture
def prepared_cli(cli_args_maker, minimal_config_yaml):
    """
    Create and prepare a DiagnosticCLI instance for testing.
    """
    def _prepared_cli(args=None):
        """Create and prepare a DiagnosticCLI instance.
        
        Args:
            args: Optional SimpleNamespace args. If None, uses defaults.
        
        Returns:
            Prepared DiagnosticCLI instance.
        """
        if args is None:
            args = cli_args_maker(minimal_config_yaml)
        
        cli = DiagnosticCLI(
            args=args,
            diagnostic_name='globalbiases',
            default_config='config_global_biases.yaml'
        )
        cli.prepare()
        return cli
    
    return _prepared_cli
