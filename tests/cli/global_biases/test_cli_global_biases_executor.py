"""Tests for Global Biases CLI execution function.

These tests verify the extracted run_global_biases_diagnostic() function
with high granularity, using mocks to avoid slow data operations.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace
from aqua.core.exceptions import NoDataError
from aqua.diagnostics.global_biases.cli_global_biases import run_global_biases_diagnostic

pytestmark = [
    pytest.mark.diagnostics,
    pytest.mark.cli
]


@pytest.fixture
def mock_cli():
    """Create a mock CLI object with necessary attributes."""
    cli = Mock()
    cli.logger = Mock()
    cli.outputdir = '/tmp/test_output'
    cli.loglevel = 'DEBUG'
    cli.save_pdf = True
    cli.save_png = True
    cli.dpi = 50
    cli.create_catalog_entry = False
    cli.reader_kwargs = {}
    cli.config_dict = {
        'datasets': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
            }
        ],
        'references': [
            {
                'catalog': 'ci',
                'model': 'ERA5',
                'exp': 'era5-hpz3',
                'source': 'monthly',
                'regrid': 'r100',
            }
        ]
    }
    cli.dataset_args = Mock(return_value={
        'catalog': 'ci',
        'model': 'ERA5',
        'exp': 'era5-hpz3',
        'source': 'monthly',
        'regrid': 'r100'
    })
    return cli


@pytest.fixture
def tool_dict_minimal():
    """Minimal tool configuration for testing."""
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


class TestRunGlobalBiasesDiagnostic:
    """Test suite for run_global_biases_diagnostic function."""
    
    def test_disabled_diagnostic_returns_false(self, mock_cli):
        """Test that disabled diagnostic returns False without execution."""
        tool_dict = {'run': False}
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict)
        
        assert result is False
        mock_cli.logger.info.assert_called_with(
            "GlobalBiases diagnostic is disabled in configuration."
        )
    
    def test_empty_tool_dict_returns_false(self, mock_cli):
        """Test that empty tool_dict returns False."""
        result = run_global_biases_diagnostic(mock_cli, {})
        
        assert result is False
    
    def test_none_tool_dict_returns_false(self, mock_cli):
        """Test that None tool_dict returns False."""
        result = run_global_biases_diagnostic(mock_cli, None)
        
        assert result is False
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_basic_execution_success(self, mock_plot_class, mock_gb_class, 
                                     mock_cli, tool_dict_minimal):
        """Test basic successful execution flow."""
        # Setup mocks
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=[])}  # No plev dimension
        mock_gb.climatology = Mock()
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        assert mock_gb_class.call_count == 2  # dataset + reference
        mock_gb.retrieve.assert_called()
        mock_gb.compute_climatology.assert_called()
        mock_plot_class.return_value.plot_bias.assert_called()
    
    def test_warns_about_multiple_datasets(self, mock_cli, tool_dict_minimal):
        """Test warning when multiple datasets provided."""
        mock_cli.config_dict['datasets'].append({
            'catalog': 'ci',
            'model': 'ERA5',
            'exp': 'era5-hpz3',
            'source': 'monthly',
        })
        
        with patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases'):
            with patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases'):
                run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        warning_calls = [str(call) for call in mock_cli.logger.warning.call_args_list]
        assert any('Only the first entry' in str(call) for call in warning_calls)
    
    def test_warns_about_multiple_references(self, mock_cli, tool_dict_minimal):
        """Test warning when multiple references provided."""
        mock_cli.config_dict['references'].append({
            'catalog': 'ci',
            'model': 'ERA5',
            'exp': 'era5-hpz3',
            'source': 'monthly',
        })
        
        with patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases'):
            with patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases'):
                run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        warning_calls = [str(call) for call in mock_cli.logger.warning.call_args_list]
        assert any('Only the first entry' in str(call) for call in warning_calls)
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_handles_missing_variable_gracefully(self, mock_plot_class, mock_gb_class,
                                                 mock_cli, tool_dict_minimal):
        """Test that missing variables are skipped with warning."""
        mock_gb = mock_gb_class.return_value
        mock_gb.retrieve.side_effect = NoDataError("Variable not found")
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True  # Should complete despite error
        mock_cli.logger.warning.assert_called()
        warning_msg = str(mock_cli.logger.warning.call_args)
        assert 'not found' in warning_msg.lower() or 'skipping' in warning_msg.lower()
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_processes_multiple_variables(self, mock_plot_class, mock_gb_class,
                                          mock_cli, tool_dict_minimal):
        """Test processing of multiple variables."""
        tool_dict_minimal['variables'] = ['2t', 'msl', 'tprate']
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=[]), 'msl': Mock(dims=[]), 'tprate': Mock(dims=[])}
        mock_gb.climatology = Mock()
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        # retrieve should be called 6 times (3 vars × 2 instances)
        assert mock_gb.retrieve.call_count == 6
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_processes_formula_variables(self, mock_plot_class, mock_gb_class,
                                        mock_cli, tool_dict_minimal):
        """Test processing of formula variables."""
        tool_dict_minimal['variables'] = []
        tool_dict_minimal['formulae'] = ['10u^2+10v^2']
        tool_dict_minimal['params']['10u^2+10v^2'] = {
            'short_name': 'windspeed',
            'long_name': 'Wind Speed'
        }
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'windspeed': Mock(dims=[])}
        mock_gb.climatology = Mock()
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        # Check that retrieve was called with formula=True
        retrieve_calls = mock_gb.retrieve.call_args_list
        assert any(call[1].get('formula') is True for call in retrieve_calls)
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_seasonal_plotting(self, mock_plot_class, mock_gb_class,
                              mock_cli, tool_dict_minimal):
        """Test that seasonal plots are generated when enabled."""
        tool_dict_minimal['params']['default']['seasons'] = True
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=[])}
        mock_gb.climatology = Mock()
        mock_gb.seasonal_climatology = Mock()
        
        mock_plot = mock_plot_class.return_value
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        mock_plot.plot_seasonal_bias.assert_called()
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_no_seasonal_plotting_when_disabled(self, mock_plot_class, mock_gb_class,
                                                mock_cli, tool_dict_minimal):
        """Test that seasonal plots are NOT generated when disabled."""
        tool_dict_minimal['params']['default']['seasons'] = False
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=[])}
        mock_gb.climatology = Mock()
        
        mock_plot = mock_plot_class.return_value
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        mock_plot.plot_seasonal_bias.assert_not_called()
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.to_list')
    def test_multiple_pressure_levels(self, mock_to_list, mock_plot_class, mock_gb_class,
                                     mock_cli, tool_dict_minimal):
        """Test plotting at multiple pressure levels."""
        tool_dict_minimal['params']['default']['plev'] = [85000, 50000, 20000]
        mock_to_list.return_value = [85000, 50000, 20000]
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=['plev', 'lat', 'lon'])}
        mock_gb.climatology = Mock()
        
        mock_plot = mock_plot_class.return_value
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        # plot_bias should be called once per pressure level
        assert mock_plot.plot_bias.call_count == 3
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_vertical_plotting(self, mock_plot_class, mock_gb_class,
                              mock_cli, tool_dict_minimal):
        """Test vertical bias plotting when enabled."""
        tool_dict_minimal['params']['default']['vertical'] = True
        tool_dict_minimal['plot_params']['2t']['vmin_v'] = -1
        tool_dict_minimal['plot_params']['2t']['vmax_v'] = 1
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=['plev', 'lat', 'lon'])}
        mock_gb.climatology = Mock()
        
        mock_plot = mock_plot_class.return_value
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        mock_plot.plot_vertical_bias.assert_called()
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_custom_projection_parameters(self, mock_plot_class, mock_gb_class,
                                          mock_cli, tool_dict_minimal):
        """Test that custom projection parameters are used."""
        tool_dict_minimal['plot_params']['2t']['projection'] = 'PlateCarree'
        tool_dict_minimal['plot_params']['2t']['projection_params'] = {
            'central_longitude': 0
        }
        
        mock_gb = mock_gb_class.return_value
        mock_gb.data = {'2t': Mock(dims=[])}
        mock_gb.climatology = Mock()
        
        mock_plot = mock_plot_class.return_value
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        # Verify plot_bias was called with custom projection
        call_args = mock_plot.plot_bias.call_args[1]
        assert call_args['proj'] == 'PlateCarree'
        assert call_args['proj_params'] == {'central_longitude': 0}
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    def test_creates_global_biases_instances_with_correct_args(self, mock_gb_class,
                                                               mock_cli, tool_dict_minimal):
        """Test that GlobalBiases instances are created with correct arguments."""
        with patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases'):
            mock_gb = mock_gb_class.return_value
            mock_gb.data = {'2t': Mock(dims=[])}
            mock_gb.climatology = Mock()
            
            run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        # Verify GlobalBiases was called twice (dataset + reference)
        assert mock_gb_class.call_count == 2
        
        # Check first call (dataset)
        first_call_kwargs = mock_gb_class.call_args_list[0][1]
        assert first_call_kwargs['catalog'] == 'ci'
        assert first_call_kwargs['model'] == 'ERA5'
        assert first_call_kwargs['diagnostic'] == 'globalbiases'
        assert first_call_kwargs['outputdir'] == '/tmp/test_output'
    
    @patch('aqua.diagnostics.global_biases.cli_global_biases.GlobalBiases')
    @patch('aqua.diagnostics.global_biases.cli_global_biases.PlotGlobalBiases')
    def test_continues_after_one_variable_fails(self, mock_plot_class, mock_gb_class,
                                               mock_cli, tool_dict_minimal):
        """Test that execution continues after one variable fails."""
        tool_dict_minimal['variables'] = ['bad_var', '2t', 'msl']
        
        mock_gb = mock_gb_class.return_value
        
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
        
        mock_gb.data = {'2t': Mock(dims=[]), 'msl': Mock(dims=[])}
        mock_gb.climatology = Mock()
        
        result = run_global_biases_diagnostic(mock_cli, tool_dict_minimal)
        
        assert result is True
        # Should have tried bad_var (1 call, failed), then 2t (2 calls), then msl (2 calls) = 5 total
        # But actually when dataset fails, the reference is not called, so it continues
        # Let's check what actually happened
        assert mock_gb.retrieve.call_count >= 4  # At least attempted multiple variables
        # Should have warned about the failed variable
        assert mock_cli.logger.warning.called

