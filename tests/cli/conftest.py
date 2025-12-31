"""Generic fixtures for diagnostic CLI testing.

These fixtures are reusable across all diagnostic CLI tests.
Diagnostic-specific fixtures should extend these in their own conftest.py files.
"""
import pytest
from types import SimpleNamespace
from aqua.core.util import dump_yaml
from aqua.diagnostics.core.cli_base import DiagnosticCLI


@pytest.fixture
def mock_cli(mocker):
    """Create a minimal generic mock CLI object with common attributes.
    
    This fixture provides only the truly common attributes used by all diagnostics.
    Diagnostic-specific fixtures should extend this with their own config_dict
    and dataset_args implementations.
    """
    cli = mocker.Mock()
    cli.logger = mocker.Mock()
    cli.loglevel = 'DEBUG'
    cli.outputdir = '/tmp/test_output'
    cli.save_pdf = True
    cli.save_png = True
    cli.dpi = 50
    cli.create_catalog_entry = False
    cli.reader_kwargs = {}
    # config_dict and dataset_args should be set by diagnostic-specific fixtures
    cli.config_dict = {}
    
    # Default behavior: return dataset dict as-is (identity) but allow overriding
    def default_dataset_args(dataset):
        return dataset
        
    cli.dataset_args = default_dataset_args
    return cli


def make_cli(config_dict, tmp_path, **args_overrides):
    """
    Function that creates a real prepared DiagnosticCLI from a config dict.
    Handles configuration file creation automatically.
    
    Usage:
        def test_something(tmp_path):
            config = {'...'}
            cli = make_cli(config, tmp_path)
            # cli is now a fully prepared DiagnosticCLI instance
    """
    config_file = tmp_path / "config.yaml"
    dump_yaml(outfile=str(config_file), cfg=config_dict)
    
    # Create args with defaults
    defaults = {
        'config': str(config_file), 
        'loglevel': 'DEBUG', 
        'catalog': None, 
        'model': None, 
        'exp': None, 
        'source': None,
        'regrid': None, 
        'outputdir': str(tmp_path), 
        'startdate': None, 
        'enddate': None, 
        'realization': None,
        'nworkers': None,
        'cluster': None
    }
    defaults.update(args_overrides)
    
    # We use a generic name here; the specific diagnostic name is passed 
    # but for the base class it mostly affects logging and config loading keys
    diagnostic_name = args_overrides.get('diagnostic_name', 'globalbiases')
    
    cli = DiagnosticCLI(
        SimpleNamespace(**defaults), 
        diagnostic_name=diagnostic_name, 
        default_config='config_defaults.yaml' # Dummy default
    )
    cli.prepare()
    return cli
