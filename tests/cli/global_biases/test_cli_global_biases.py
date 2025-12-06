"""Tests for Global Biases CLI."""

import pytest
from aqua.diagnostics.global_biases.cli_global_biases import run_global_biases_diagnostic
from aqua.diagnostics.core import DiagnosticCLI
from aqua.core.exceptions import NoDataError

pytestmark = [pytest.mark.diagnostics, pytest.mark.cli]

class TestGlobalBiasesConfig:
    """Integration tests for Global Biases CLI configuration loading."""

    def test_config_loading(self, prepare_cli, gb_config_file, tmp_path):
        """Verify that configuration is correctly loaded into DiagnosticCLI."""
        cli = prepare_cli(gb_config_file)

        # Check basic structure
        assert 'globalbiases' in cli.config_dict['diagnostics']
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        assert tool_dict['run'] is True
        assert tool_dict['variables'] == ['2t']
        
        # Check dataset parsing
        dataset = cli.config_dict['datasets'][0]
        dataset_args = cli.dataset_args(dataset)
        assert dataset_args['model'] == 'ERA5'
        
        # Check output dir
        assert cli.outputdir == str(tmp_path)

    def test_seasonal_config(self, prepare_cli, gb_config_seasonal_file):
        """Verify loading of seasonal configuration."""
        cli = prepare_cli(gb_config_seasonal_file)
        
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        assert tool_dict['params']['default']['seasons'] is True

    def test_formula_config(self, prepare_cli, gb_config_formula_file):
        """Verify loading of formula configuration."""
        cli = prepare_cli(gb_config_formula_file)
        
        tool_dict = cli.config_dict['diagnostics']['globalbiases']
        assert tool_dict['formulae'] == ['tnlwrf+tnswrf']
        assert tool_dict['params']['tnlwrf+tnswrf']['short_name'] == 'tnr'

    def test_cli_overrides(self, prepare_cli, gb_config_file):
        """Verify that CLI arguments override config file settings."""
        cli = prepare_cli(gb_config_file, model='OverriddenModel', regrid='r200')
        
        dataset = cli.config_dict['datasets'][0]
        assert dataset['model'] == 'OverriddenModel'
        assert dataset['regrid'] == 'r200'


class TestGlobalBiasesRun:
    """Unit tests for Global Biases execution logic (run_global_biases_diagnostic)."""

    def test_disabled_diagnostic(self, mock_cli_global_biases):
        """Test that disabled diagnostic returns False."""
        tool_dict = {'run': False}
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        assert result is False
        mock_cli_global_biases.logger.info.assert_called_with("GlobalBiases diagnostic is disabled.")

    def test_execution_success(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb):
        """Test successful execution flow."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb_class.return_value = mock_gb

        tool_dict = mock_cli_global_biases.config_dict['diagnostics']['globalbiases']
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        assert result is True
        # Dataset and reference initialized
        assert mock_gb_class.call_count == 2
        mock_gb.retrieve.assert_called()
        mock_gb.compute_climatology.assert_called()
        mock_plot_class.return_value.plot_bias.assert_called()

    def test_missing_variable(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb):
        """Test graceful handling of missing variables."""
        mock_gb_class, _ = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        # Simulate retrieval failure
        mock_gb.retrieve.side_effect = NoDataError("Missing var")
        mock_gb_class.return_value = mock_gb

        tool_dict = mock_cli_global_biases.config_dict['diagnostics']['globalbiases']
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        # Should still return True (diagnostic ran, even if one var failed)
        assert result is True
        mock_cli_global_biases.logger.warning.assert_called()

    def test_formula_execution(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb, gb_config_formula):
        """Test execution with formula."""
        mock_cli_global_biases.config_dict = gb_config_formula
        tool_dict = gb_config_formula['diagnostics']['globalbiases']
        
        mock_gb_class, _ = patched_global_biases_classes
        mock_gb = setup_mock_gb({'tnr': []})
        mock_gb_class.return_value = mock_gb
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        # Check that retrieve was called with formula=True
        calls = mock_gb.retrieve.call_args_list
        assert any(call.kwargs.get('formula') is True for call in calls)

    def test_seasonal_execution(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb, gb_config_seasonal):
        """Test execution with seasonal plots."""
        mock_cli_global_biases.config_dict = gb_config_seasonal
        tool_dict = gb_config_seasonal['diagnostics']['globalbiases']
        
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []}, with_seasonal=True)
        mock_gb_class.return_value = mock_gb
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        mock_plot_class.return_value.plot_seasonal_bias.assert_called()

    def test_vertical_execution(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb):
        """Test execution with vertical plots."""
        tool_dict = mock_cli_global_biases.config_dict['diagnostics']['globalbiases']
        tool_dict['params']['default']['vertical'] = True
        tool_dict['plot_params']['2t']['vmin_v'] = -1
        tool_dict['plot_params']['2t']['vmax_v'] = 1
        
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []}, has_plev=True)
        mock_gb_class.return_value = mock_gb
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        mock_plot_class.return_value.plot_vertical_bias.assert_called()

    def test_multiple_pressure_levels(self, mock_cli_global_biases, patched_global_biases_classes, setup_mock_gb):
        """Test execution over multiple pressure levels."""
        tool_dict = mock_cli_global_biases.config_dict['diagnostics']['globalbiases']
        tool_dict['params']['default']['plev'] = [85000, 50000]
        
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []}, has_plev=True)
        mock_gb_class.return_value = mock_gb
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        
        # Should plot twice
        assert mock_plot_class.return_value.plot_bias.call_count == 2
