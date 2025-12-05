"""Generic fixtures for diagnostic CLI testing.

These fixtures are reusable across all diagnostic CLI tests.
Diagnostic-specific fixtures should extend these in their own conftest.py files.
"""
import pytest


@pytest.fixture
def mock_cli(mocker):
    """Create a minimal generic mock CLI object with common attributes.
    
    This fixture provides only the truly common attributes used by all diagnostics.
    Diagnostic-specific fixtures should extend this with their own config_dict
    and dataset_args implementations.
    
    Common attributes:
    - logger: Mock logger
    - loglevel: Logging level
    - outputdir: Output directory
    - save_pdf, save_png: Plot saving flags
    - dpi: Plot resolution
    - create_catalog_entry: Catalog entry flag
    - reader_kwargs: Reader keyword arguments
    - config_dict: Should be set by diagnostic-specific fixtures
    - dataset_args: Should be set by diagnostic-specific fixtures
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
    cli.dataset_args = lambda x: {}
    return cli


@pytest.fixture
def patch_diagnostic_classes(mocker):
    """Generic fixture to patch diagnostic and plot classes.
    
    Returns a factory function that patches classes at the given module path.
    This makes it easy to patch classes for any diagnostic.
    
    Usage:
        patch_classes = patch_diagnostic_classes
        mock_diag_class, mock_plot_class = patch_classes(
            'aqua.diagnostics.my_diag.cli_my_diag.MyDiagnostic',
            'aqua.diagnostics.my_diag.cli_my_diag.PlotMyDiagnostic'
        )
    """
    def _patch(diagnostic_class_path, plot_class_path):
        """Patch diagnostic and plot classes at given paths.
        
        Args:
            diagnostic_class_path: Full module path to diagnostic class
            plot_class_path: Full module path to plot class
        
        Returns:
            tuple: (mock_diagnostic_class, mock_plot_class)
        """
        mock_diag_class = mocker.patch(diagnostic_class_path)
        mock_plot_class = mocker.patch(plot_class_path)
        return mock_diag_class, mock_plot_class
    return _patch


@pytest.fixture
def setup_mock_diagnostic_instance(mocker):
    """Generic helper to set up a mock diagnostic instance with data.
    
    This fixture returns a factory function that creates mock diagnostic
    instances with the standard interface (data, climatology, etc.).
    
    Usage:
        setup_mock = setup_mock_diagnostic_instance
        mock_diag = setup_mock({'var1': [], 'var2': ['plev', 'lat', 'lon']}, has_plev=False)
    """
    def _setup(vars_dict=None, has_plev=False):
        """Setup mock diagnostic instance with specified variables.
        
        Args:
            vars_dict: Dict mapping var names to dims lists, or None for default {'2t': []}
            has_plev: If True, all vars get ['plev', 'lat', 'lon'], else use provided dims
        
        Returns:
            Mock diagnostic instance with data, climatology, etc.
        """
        mock_diag = mocker.Mock()
        if vars_dict is None:
            vars_dict = {'2t': []}
        
        if has_plev:
            mock_diag.data = {
                var: mocker.Mock(dims=['plev', 'lat', 'lon'])
                for var in vars_dict.keys()
            }
        else:
            mock_diag.data = {
                var: mocker.Mock(dims=dims)
                for var, dims in vars_dict.items()
            }
        
        mock_diag.climatology = mocker.Mock()
        mock_diag.seasonal_climatology = None
        return mock_diag
    return _setup

