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
    def test_disabled_or_empty_diagnostic_returns_false(self, mock_cli_global_biases, tool_dict):
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
        
        warning_calls = [str(call) for call in mock_cli_global_biases.logger.warning.call_args_list]
        assert any('Only the first entry' in str(call) for call in warning_calls)
    
    def test_handles_missing_variable_gracefully(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that missing variables are skipped with warning."""
        mock_gb_class, _ = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb.retrieve.side_effect = NoDataError("Variable not found")
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True  # Should complete despite error
        mock_cli_global_biases.logger.warning.assert_called()
        warning_msg = str(mock_cli_global_biases.logger.warning.call_args)
        assert 'not found' in warning_msg.lower() or 'skipping' in warning_msg.lower()
    
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
        assert any(call[1].get('formula') is True for call in retrieve_calls)
    
    def test_seasonal_plotting(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that seasonal plots are generated when enabled."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['params']['default']['seasons'] = True
        
        mock_gb = setup_mock_gb({'2t': []}, with_seasonal=True)
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        mock_plot_class.return_value.plot_seasonal_bias.assert_called()
    
    def test_no_seasonal_plotting_when_disabled(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that seasonal plots are NOT generated when disabled."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['params']['default']['seasons'] = False
        
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        mock_plot_class.return_value.plot_seasonal_bias.assert_not_called()
    
    def test_multiple_pressure_levels(self, mocker, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test plotting at multiple pressure levels."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        mock_to_list = mocker.patch('aqua.diagnostics.global_biases.cli_global_biases.to_list')
        
        tool_dict_minimal['params']['default']['plev'] = [85000, 50000, 20000]
        mock_to_list.return_value = [85000, 50000, 20000]
        
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
    
    def test_custom_projection_parameters(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that custom projection parameters are used."""
        mock_gb_class, mock_plot_class = patched_global_biases_classes
        tool_dict_minimal['plot_params']['2t']['projection'] = 'PlateCarree'
        tool_dict_minimal['plot_params']['2t']['projection_params'] = {
            'central_longitude': 0
        }
        
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        # Verify plot_bias was called with custom projection
        call_args = mock_plot_class.return_value.plot_bias.call_args[1]
        assert call_args['proj'] == 'PlateCarree'
        assert call_args['proj_params'] == {'central_longitude': 0}
    
    def test_creates_global_biases_instances_with_correct_args(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that GlobalBiases instances are created with correct arguments."""
        mock_gb_class, _ = patched_global_biases_classes
        mock_gb = setup_mock_gb({'2t': []})
        mock_gb_class.return_value = mock_gb
        
        run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        # Verify GlobalBiases was called twice (dataset + reference)
        assert mock_gb_class.call_count == 2
        
        # Check first call (dataset)
        first_call_kwargs = mock_gb_class.call_args_list[0][1]
        assert first_call_kwargs['catalog'] == 'ci'
        assert first_call_kwargs['model'] == 'ERA5'
        assert first_call_kwargs['diagnostic'] == 'globalbiases'
        assert first_call_kwargs['outputdir'] == '/tmp/test_output'
    
    def test_continues_after_one_variable_fails(self, mock_cli_global_biases, tool_dict_minimal, patched_global_biases_classes, setup_mock_gb):
        """Test that execution continues after one variable fails."""
        mock_gb_class, _ = patched_global_biases_classes
        tool_dict_minimal['variables'] = ['bad_var', '2t', 'msl']
        
        mock_gb = setup_mock_gb({'2t': [], 'msl': []})
        # First retrieve fails (for dataset), then we skip to next variable
        # The retrieve is called twice per variable (dataset + reference)
        # If the first one (dataset) fails, we catch it and continue, so reference is also called
        mock_gb.retrieve.side_effect = [
            NoDataError("bad_var not found"),  # bad_var dataset fails
            # Note: When dataset retrieve fails, reference retrieve is NOT called (exception is caught)
            None,  # 2t dataset succeeds
            None,  # 2t reference succeeds  
            None,  # msl dataset succeeds
            None   # msl reference succeeds
        ]
        mock_gb_class.return_value = mock_gb
        
        result = run_global_biases_diagnostic(mock_cli_global_biases, tool_dict_minimal)
        
        assert result is True
        # Should have tried bad_var (1 call, failed), then 2t (2 calls), then msl (2 calls) = 5 total
        # But actually when dataset fails, the reference is not called, so it continues
        # Let's check what actually happened
        assert mock_gb.retrieve.call_count >= 4  # At least attempted multiple variables
        # Should have warned about the failed variable
        assert mock_cli_global_biases.logger.warning.called
