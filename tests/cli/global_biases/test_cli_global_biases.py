"""Tests for Global Biases CLI."""

import pytest
from types import SimpleNamespace
from aqua.diagnostics.global_biases.cli_global_biases import run_global_biases_diagnostic
from .conftest import build_config
from ..conftest import make_cli

pytestmark = [pytest.mark.diagnostics, pytest.mark.cli]

class TestGlobalBiasesConfig:
    """Integration tests for Global Biases CLI configuration loading using real DiagnosticCLI."""

    def test_config_loading(self, tmp_path):
        """Verify standard config loading."""
        config = build_config(output_dir=str(tmp_path))
        cli = make_cli(config, tmp_path)
        
        # Check tool config
        tool_conf = cli.config_dict['diagnostics']['globalbiases']
        assert tool_conf['run'] is True
        assert tool_conf['variables'] == ['2t']
        
        # Check dataset parsing
        dataset = cli.config_dict['datasets'][0]
        dataset_args = cli.dataset_args(dataset)
        assert dataset_args['model'] == 'ERA5'
        
        # Check output dir
        assert cli.outputdir == str(tmp_path)

    def test_seasonal_config(self, tmp_path):
        """Verify seasonal configuration loading."""
        config = build_config(tool_overrides={'params': {'default': {'seasons': True}}})
        cli = make_cli(config, tmp_path)
        
        assert cli.config_dict['diagnostics']['globalbiases']['params']['default']['seasons'] is True

    def test_formula_config(self, tmp_path):
        """Verify formula configuration loading."""
        overrides = {
            'variables': [],
            'formulae': ['tnlwrf+tnswrf'],
            'params': {'tnlwrf+tnswrf': {'short_name': 'tnr'}}
        }
        config = build_config(tool_overrides=overrides)
        cli = make_cli(config, tmp_path)
        
        tool_conf = cli.config_dict['diagnostics']['globalbiases']
        assert tool_conf['formulae'] == ['tnlwrf+tnswrf']
        assert tool_conf['params']['tnlwrf+tnswrf']['short_name'] == 'tnr'

    def test_cli_overrides(self, tmp_path):
        """Verify that CLI arguments override config file settings."""
        config = build_config()
        config['datasets'][0].pop('regrid', None)
        cli = make_cli(config, tmp_path, model='OverriddenModel', regrid='r200')
        
        assert cli.config_dict['datasets'][0]['model'] == 'OverriddenModel'
        assert cli.regrid == 'r200'
        assert cli.dataset_args(cli.config_dict['datasets'][0])['regrid'] == 'r200'


class TestGlobalBiasesRun:
    """Unit tests for Global Biases execution logic using mocks."""

    def test_execution_success(self, mock_cli, mock_gb):
        """Test successful execution flow."""
        config = build_config()
        mock_cli.config_dict = config
        
        tool_dict = config['diagnostics']['globalbiases']
        result = run_global_biases_diagnostic(mock_cli, tool_dict)
        
        assert result is True
        assert mock_gb.cls.call_count == 2
        mock_gb.instance.retrieve.assert_called()
        mock_gb.instance.compute_climatology.assert_called()
        mock_gb.plot_cls.return_value.plot_bias.assert_called()

    def test_disabled_diagnostic(self, mock_cli):
        """Test that disabled diagnostic returns False."""
        assert run_global_biases_diagnostic(mock_cli, {'run': False}) is False
        mock_cli.logger.info.assert_called_with("GlobalBiases diagnostic is disabled.")

    def test_missing_variable(self, mock_cli, mock_gb):
        """Test graceful handling of missing variables."""
        config = build_config()
        mock_cli.config_dict = config
        mock_gb.instance.retrieve.side_effect = KeyError("Variable not found")
        
        result = run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        assert result is True
        mock_cli.logger.warning.assert_called()

    def test_seasonal_execution(self, mock_cli, mock_gb):
        """Test execution with seasonal plots."""
        config = build_config(tool_overrides={'params': {'default': {'seasons': True}}})
        mock_cli.config_dict = config
        mock_gb.instance.seasonal_climatology = "exists"
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        mock_gb.plot_cls.return_value.plot_seasonal_bias.assert_called()

    def test_formula_execution(self, mock_cli, mock_gb):
        """Test execution with formula."""
        overrides = {
            'variables': [],
            'formulae': ['tnlwrf+tnswrf'],
            'params': {'tnlwrf+tnswrf': {'short_name': 'tnr'}}
        }
        config = build_config(tool_overrides=overrides)
        mock_cli.config_dict = config
        # Ensure mock data has the formula variable with proper structure
        mock_gb.instance.data = {'tnr': SimpleNamespace(dims=['lat', 'lon'])}
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        # Verify retrieve was called with formula=True
        calls = mock_gb.instance.retrieve.call_args_list
        assert any(call.kwargs.get('formula') is True for call in calls)

    def test_vertical_execution(self, mock_cli, mock_gb):
        """Test execution with vertical plots."""
        overrides = {
            'params': {'default': {'vertical': True}},
            'plot_params': {'2t': {'vmin_v': -1, 'vmax_v': 1}}
        }
        config = build_config(tool_overrides=overrides)
        mock_cli.config_dict = config
        
        # Mock data having plev dim
        mock_gb.instance.data['2t'].dims = ['plev', 'lat', 'lon']
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        mock_gb.plot_cls.return_value.plot_vertical_bias.assert_called()

    def test_multiple_pressure_levels(self, mock_cli, mock_gb):
        """Test execution over multiple pressure levels."""
        config = build_config(tool_overrides={'params': {'default': {'plev': [85000, 50000]}}})
        mock_cli.config_dict = config
        mock_gb.instance.data['2t'].dims = ['plev', 'lat', 'lon']
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        # Should plot twice (once for each level)
        assert mock_gb.plot_cls.return_value.plot_bias.call_count == 2

    def test_configuration_propagation(self, mock_cli, mock_gb):
        """Verify propagation of configuration parameters to methods."""
        config = build_config(tool_overrides={
            'params': {
                'default': {'seasons': True, 'seasons_stat': 'std'},
                '2t': {'units': 'K', 'long_name': '2-meter temperature'}
            },
            'plot_params': {
                'default': {
                    'projection': 'mollweide', 
                    'projection_params': {'central_longitude': 180},
                    'cmap': 'magma'
                },
                '2t': {'vmin': -10, 'vmax': 10}
            }
        })
        mock_cli.config_dict = config
        mock_cli.create_catalog_entry = True
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])

        retrieve_calls = mock_gb.instance.retrieve.call_args_list
        assert any(call.kwargs.get('units') == 'K' for call in retrieve_calls)
        assert any(call.kwargs.get('long_name') == '2-meter temperature' for call in retrieve_calls)

        compute_calls = mock_gb.instance.compute_climatology.call_args_list
        assert compute_calls[0].kwargs['seasonal'] is True
        assert compute_calls[0].kwargs['seasons_stat'] == 'std'
        assert compute_calls[0].kwargs['create_catalog_entry'] is True

        plot_kwargs = mock_gb.plot_cls.return_value.plot_bias.call_args.kwargs
        assert plot_kwargs['proj'] == 'mollweide'
        assert plot_kwargs['proj_params'] == {'central_longitude': 180}
        assert plot_kwargs['vmin'] == -10
        assert plot_kwargs['vmax'] == 10

    def test_variable_renaming_flow(self, mock_cli, mock_gb):
        """Verify that short_name is used for plotting after retrieval."""
        overrides = {
            'variables': [],
            'formulae': ['varA+varB'],
            'params': {'varA+varB': {'short_name': 'result_var'}}
        }
        config = build_config(tool_overrides=overrides)
        mock_cli.config_dict = config
        mock_gb.instance.data = {'result_var': SimpleNamespace(dims=['lat', 'lon'])}
        
        run_global_biases_diagnostic(mock_cli, config['diagnostics']['globalbiases'])
        
        assert mock_gb.plot_cls.return_value.plot_bias.call_args.kwargs['var'] == 'result_var'

    def test_variable_failure_is_handled(self, mock_cli, mock_gb):
        config = build_config(tool_overrides={"variables": ["bad_var"]})
        mock_cli.config_dict = config

        mock_gb.instance.retrieve.side_effect = ValueError("Missing data")

        assert run_global_biases_diagnostic(mock_cli, config["diagnostics"]["globalbiases"]) is True
        mock_cli.logger.warning.assert_called()

    def test_good_variable_is_plotted(self, mock_cli, mock_gb):
        config = build_config(tool_overrides={"variables": ["good_var"]})
        mock_cli.config_dict = config
        mock_gb.instance.data = {"good_var": SimpleNamespace(dims=["lat", "lon"])}

        run_global_biases_diagnostic(mock_cli, config["diagnostics"]["globalbiases"])

        plot_bias = mock_gb.plot_cls.return_value.plot_bias
        assert any(call.kwargs.get("var") == "good_var" for call in plot_bias.call_args_list)
