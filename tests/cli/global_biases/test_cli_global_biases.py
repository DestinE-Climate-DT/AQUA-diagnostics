"""Optimized tests for Global Biases CLI.

These tests verify the CLI interface for the Global Biases diagnostic,
ensuring proper integration of command-line arguments, config files, and
diagnostic execution without duplicating the underlying diagnostic tests.
"""
import pytest
from types import SimpleNamespace
from aqua.diagnostics.global_biases.cli_global_biases import parse_arguments
from aqua.diagnostics.core.cli_base import DiagnosticCLI

# Mark all tests in this module
pytestmark = [
    pytest.mark.diagnostics,
    pytest.mark.cli
]


class TestGlobalBiasesCLI:
    """Test suite for Global Biases CLI interface."""

    def test_parse_arguments(self):
        """Test argument parsing with minimal and full arguments."""
        # Minimal arguments
        args = parse_arguments(['--config', 'test_config.yaml'])
        assert args.config == 'test_config.yaml'
        assert hasattr(args, 'loglevel')
        assert hasattr(args, 'catalog')
        
        # Full arguments
        args_full = parse_arguments([
            '--config', 'test.yaml',
            '--loglevel', 'DEBUG',
            '--catalog', 'test-catalog',
            '--model', 'TestModel',
            '--exp', 'test-exp',
            '--source', 'test-source',
            '--regrid', 'r100',
            '--outputdir', '/tmp/output'
        ])
        assert args_full.loglevel == 'DEBUG'
        assert args_full.catalog == 'test-catalog'
        assert args_full.regrid == 'r100'

    def test_cli_config_loading_and_extraction(self, minimal_config_yaml, tmp_path):
        """Test CLI loads config and extracts parameters correctly."""
        args = SimpleNamespace(
            config=minimal_config_yaml,
            loglevel='DEBUG',
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(
            args=args,
            diagnostic_name='globalbiases',
            default_config='config_global_biases.yaml'
        )
        cli.prepare()
        
        # Verify config structure
        assert cli.config_dict is not None
        assert 'diagnostics' in cli.config_dict
        assert 'globalbiases' in cli.config_dict['diagnostics']
        assert cli.config_dict['diagnostics']['globalbiases']['run'] is True
        
        # Verify dataset configuration
        assert len(cli.config_dict['datasets']) == 1
        dataset = cli.config_dict['datasets'][0]
        assert dataset['catalog'] == 'ci'
        assert dataset['model'] == 'ERA5'
        
        # Verify dataset_args extraction
        dataset_args = cli.dataset_args(dataset)
        assert dataset_args['catalog'] == 'ci'
        assert dataset_args['model'] == 'ERA5'
        assert dataset_args['exp'] == 'era5-hpz3'
        assert dataset_args['source'] == 'monthly'
        assert dataset_args['regrid'] == 'r100'
        assert dataset_args['startdate'] == '1990-01-01'
        assert dataset_args['enddate'] == '1990-12-31'
        
        # Verify output settings
        assert cli.outputdir == str(tmp_path)
        assert cli.save_pdf is True
        assert cli.save_png is True
        assert cli.dpi == 50

    def test_cli_parameter_override(self, minimal_config_yaml, tmp_path):
        """Test that CLI args override config file settings."""
        custom_output = str(tmp_path / 'custom_output')
        
        args = SimpleNamespace(
            config=minimal_config_yaml,
            loglevel='INFO',
            catalog='override-catalog',
            model='OverrideModel',
            exp='override-exp',
            source='override-source',
            regrid='r200',
            outputdir=custom_output,
            startdate='1991-01-01',
            enddate='1991-12-31',
            realization='r2i1p1f1',
            nworkers=None,
            cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        # Check CLI args override config
        dataset = cli.config_dict['datasets'][0]
        assert dataset['catalog'] == 'override-catalog'
        assert dataset['model'] == 'OverrideModel'
        assert dataset['exp'] == 'override-exp'
        assert dataset['source'] == 'override-source'
        assert cli.outputdir == custom_output
        assert cli.regrid == 'r200'
        assert cli.realization == 'r2i1p1f1'
        assert cli.reader_kwargs == {'realization': 'r2i1p1f1'}

    def test_diagnostic_parameters(self, minimal_config_yaml):
        """Test diagnostic-specific parameters are correctly loaded."""
        args = SimpleNamespace(
            config=minimal_config_yaml, loglevel='DEBUG',
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        
        # Check diagnostic parameters
        assert tool_dict['run'] is True
        assert tool_dict['diagnostic_name'] == 'globalbiases'
        assert '2t' in tool_dict['variables']
        assert tool_dict['formulae'] == []
        
        # Check params
        default_params = tool_dict['params']['default']
        assert default_params['seasons'] is False
        assert default_params['vertical'] is False
        
        # Check plot params
        plot_params = tool_dict['plot_params']['2t']
        assert plot_params['vmin'] == -5
        assert plot_params['vmax'] == 5
        assert plot_params['cmap'] == 'RdBu_r'

    @pytest.mark.parametrize("config_type,expected_seasons,expected_formulae", [
        ('minimal', False, []),
        ('seasonal', True, []),
        ('formula', False, ['tnlwrf+tnswrf']),
    ])
    def test_config_variations(self, config_type, expected_seasons, expected_formulae, request):
        """Test different configuration variations (seasonal, formula)."""
        # Get the appropriate fixture
        if config_type == 'minimal':
            config = request.getfixturevalue('minimal_config_yaml')
        elif config_type == 'seasonal':
            config = request.getfixturevalue('minimal_config_yaml_with_seasons')
        else:  # formula
            config = request.getfixturevalue('minimal_config_yaml_with_formula')
        
        args = SimpleNamespace(
            config=config, loglevel='DEBUG',
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        default_params = tool_dict['params']['default']
        
        assert default_params['seasons'] == expected_seasons
        assert tool_dict['formulae'] == expected_formulae
        
        if 'tnlwrf+tnswrf' in expected_formulae:
            formula_params = tool_dict['params']['tnlwrf+tnswrf']
            assert formula_params['short_name'] == 'tnr'
            assert formula_params['long_name'] == 'Top net radiation'

    def test_multiple_datasets_handling(self, minimal_config_yaml, tmp_path):
        """Test CLI handles multiple datasets correctly."""
        from aqua.core.util import load_yaml, dump_yaml
        
        config = load_yaml(minimal_config_yaml)
        config['datasets'].append({
            'catalog': 'ci', 'model': 'ERA5', 'exp': 'era5-hpz3',
            'source': 'monthly', 'regrid': 'r100',
            'startdate': '1990-01-01', 'enddate': '1990-12-31',
        })
        
        multi_config_file = tmp_path / "multi_dataset_config.yaml"
        dump_yaml(outfile=str(multi_config_file), cfg=config)
        
        args = SimpleNamespace(
            config=str(multi_config_file), loglevel='DEBUG',
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        # Verify config has multiple datasets
        assert len(cli.config_dict['datasets']) == 2

    @pytest.mark.parametrize("loglevel", ['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    def test_loglevel_configuration(self, minimal_config_yaml, loglevel):
        """Test CLI correctly sets different log levels."""
        args = SimpleNamespace(
            config=minimal_config_yaml, loglevel=loglevel,
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        assert cli.loglevel == loglevel

    def test_cli_integration(self, minimal_config_yaml):
        """Integration test: Verify complete CLI setup workflow."""
        args = SimpleNamespace(
            config=minimal_config_yaml, loglevel='DEBUG',
            catalog=None, model=None, exp=None, source=None,
            regrid=None, outputdir=None, startdate=None, enddate=None,
            realization=None, nworkers=None, cluster=None
        )
        
        cli = DiagnosticCLI(args=args, diagnostic_name='globalbiases',
                           default_config='config_global_biases.yaml')
        cli.prepare()
        
        # Verify all components are ready for diagnostic execution
        assert cli.logger is not None
        assert cli.config_dict is not None
        assert cli.outputdir is not None
        
        # Verify diagnostic config
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        assert tool_dict['run'] is True
        
        # Verify dataset and reference are configured
        dataset_args = cli.dataset_args(cli.config_dict['datasets'][0])
        reference_args = cli.dataset_args(cli.config_dict['references'][0])
        
        required_keys = ['catalog', 'model', 'exp', 'source', 'regrid']
        assert all(key in dataset_args for key in required_keys)
        assert all(key in reference_args for key in required_keys)
        
        # Verify output settings
        assert cli.save_pdf is True
        assert cli.save_png is True
        assert cli.save_netcdf is True

