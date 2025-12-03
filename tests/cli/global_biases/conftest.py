"""Optimized fixtures for Global Biases CLI tests."""
import pytest
from aqua.core.util import dump_yaml


@pytest.fixture
def minimal_config_yaml(tmp_path):
    """Create a minimal config YAML file for Global Biases CLI testing.
    
    Uses minimal data loading:
    - Single variable: '2t' (2-meter temperature)
    - 1 year of data (1990)
    - Seasons disabled by default
    - Low DPI for faster testing
    """
    config = {
        'datasets': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
                'startdate': '1990-01-01',
                'enddate': '1990-12-31',
            }
        ],
        'references': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
                'startdate': '1990-01-01',
                'enddate': '1990-12-31',
            }
        ],
        'setup': {
            'loglevel': 'DEBUG',
        },
        'output': {
            'outputdir': str(tmp_path),
            'rebuild': True,
            'save_netcdf': True,
            'save_pdf': True,
            'save_png': True,
            'dpi': 50,
            'create_catalog_entry': False,
        },
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
    from aqua.core.util import load_yaml
    
    config = load_yaml(minimal_config_yaml)
    config['diagnostics']['globalbiases']['params']['default']['seasons'] = True
    config['diagnostics']['globalbiases']['params']['default']['seasons_stat'] = 'mean'
    
    config_file = tmp_path / "test_global_biases_config_seasonal.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def minimal_config_yaml_with_formula(tmp_path):
    """Create config with formula for testing formula handling."""
    config = {
        'datasets': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
                'startdate': '1990-01-01',
                'enddate': '1990-12-31',
            }
        ],
        'references': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
                'startdate': '1990-01-01',
                'enddate': '1990-12-31',
            }
        ],
        'setup': {'loglevel': 'DEBUG'},
        'output': {
            'outputdir': str(tmp_path),
            'rebuild': True,
            'save_netcdf': True,
            'save_pdf': True,
            'save_png': True,
            'dpi': 50,
            'create_catalog_entry': False,
        },
        'diagnostics': {
            'globalbiases': {
                'run': True,
                'diagnostic_name': 'globalbiases',
                'variables': [],
                'formulae': ['tnlwrf+tnswrf'],
                'params': {
                    'default': {
                        'seasons': False,
                        'vertical': False,
                    },
                    'tnlwrf+tnswrf': {
                        'short_name': 'tnr',
                        'long_name': 'Top net radiation',
                    }
                },
                'plot_params': {
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
            }
        }
    }
    
    config_file = tmp_path / "test_global_biases_config_formula.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)

