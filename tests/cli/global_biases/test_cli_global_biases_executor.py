"""Tests for Global Biases CLI execution function.

These tests verify the extracted run_global_biases_diagnostic() function
with high granularity, using mocks to avoid slow data operations.

Fixtures are defined in conftest.py for reusability across diagnostic tests.
"""
import pytest
from aqua.core.exceptions import NoDataError
from aqua.diagnostics.global_biases.cli_global_biases import run_global_biases_diagnostic

pytestmark = [
    pytest.mark.diagnostics,
    pytest.mark.cli
]


class TestRunGlobalBiasesDiagnostic:
    """Test suite for run_global_biases_diagnostic function."""
    
    @pytest.mark.parametrize("tool_dict", [
        {'run': False},
        {},
        None,
    ])
    def test_disabled_or_empty_diagnostic(self, mock_cli_global_biases, tool_dict):
        """Test that disabled, empty, or None diagnostic returns False."""
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict)
        assert result is False
        
        if tool_dict == {'run': False}:
            mock_cli_global_biases.logger.info.assert_called_with(
                "GlobalBiases diagnostic is disabled."
            )
    
    def test_basic_execution_success(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test basic successful execution flow."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        assert mock_gb_class.call_count == 2  # dataset + reference
        mock_gb.retrieve.assert_called()
        mock_gb.compute_climatology.assert_called()
        mock_plot_class.return_value.plot_bias.assert_called()
    
    @pytest.mark.parametrize("config_key", ['datasets', 'references'])
    def test_warns_about_multiple_entries(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, config_key):
        """Test warning when multiple datasets or references provided."""
        mock_cli_global_biases.config_dict[config_key].append({
            'catalog': 'ci',
            'model': 'ERA5',
            'exp': 'era5-hpz3',
            'source': 'monthly',
        })
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        warning_calls = mock_cli_global_biases.logger.warning.call_args_list
        assert any('Only the first entry' in str(call) for call in warning_calls)
    
    def test_handles_missing_variable_gracefully(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that missing variables are skipped with warning."""
        mock_gb_class, _ = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb.retrieve.side_effect = NoDataError("Variable not found")
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        mock_cli_global_biases.logger.warning.assert_called()
        # Check the actual warning message
        warning_args = mock_cli_global_biases.logger.warning.call_args[0]
        assert 'not found' in warning_args[0].lower() or 'skipping' in warning_args[0].lower()
    
    def test_processes_multiple_variables(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test processing of multiple variables."""
        mock_gb_class, _ = patched_global_biases_classes
        tool_dict_minimal['variables'] = ['2t', 'msl', 'tprate']
        
        mock_gb = setup_mock_gb({'2t': [], 'msl': [], 'tprate': []})
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        # retrieve should be called 6 times (3 vars × 2 instances)
        assert mock_gb.retrieve.call_count == 6
    
    def test_processes_formula_variables(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test processing of formula variables."""
        mock_gb_class, _ = patched_global_biases_classes
        tool_dict_minimal['variables'] = []
        tool_dict_minimal['formulae'] = ['10u^2+10v^2']
        tool_dict_minimal['params']['10u^2+10v^2'] = {
            'short_name': 'windspeed',
            'long_name': 'Wind Speed'
        }
        
        mock_gb = setup_mock_gb({'windspeed': []})
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        # Check that retrieve was called with formula=True
        retrieve_calls = mock_gb.retrieve.call_args_list
        assert any(call.kwargs.get('formula') is True for call in retrieve_calls)
    
    def test_seasonal_plotting(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that seasonal plots are generated when enabled."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['params']['default']['seasons'] = True
        
        mock_gb = setup_mock_gb({'2t': []}, with_seasonal=True)
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        mock_plot_class.return_value.plot_seasonal_bias.assert_called()
    
    def test_multiple_pressure_levels(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test plotting at multiple pressure levels."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['params']['default']['plev'] = [85000, 50000, 20000]
        
        mock_gb = setup_mock_gb({'2t': []}, has_plev=True)
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        # plot_bias should be called once per pressure level
        assert mock_plot_class.return_value.plot_bias.call_count == 3
    
    def test_vertical_plotting(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test vertical bias plotting when enabled."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['params']['default']['vertical'] = True
        tool_dict_minimal['plot_params']['2t']['vmin_v'] = -1
        tool_dict_minimal['plot_params']['2t']['vmax_v'] = 1
        
        mock_gb = setup_mock_gb({'2t': []}, has_plev=True)
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        mock_plot_class.return_value.plot_vertical_bias.assert_called()
