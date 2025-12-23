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
def setup_mock_diagnostic_base(mocker):
    """Generic helper to set up a minimal mock diagnostic instance with data.
    
    This fixture returns a factory function that creates mock diagnostic
    instances with a minimal standard interface. Diagnostic-specific fixtures
    should extend this for their needs.
    
    Usage:
        setup_mock = setup_mock_diagnostic_base
        # Basic usage with variables and their dimensions
        mock_diag = setup_mock({
            'var': ['plev', 'lat', 'lon']
        })
        
        # With additional attributes
        mock_diag = setup_mock(
            {'var2': ['lat', 'lon']},
            climatology=mocker.Mock(),
            seasonal_climatology=None
        )
    """
    def _setup(vars_dict, **extra_attrs):
        """Setup mock diagnostic instance with specified variables.
        
        Args:
            vars_dict: Dict mapping variable names to their dimension lists.
                      Each variable will be a mock with a 'dims' attribute.
                      Example: {'2t': ['lat', 'lon'], 't': ['plev', 'lat', 'lon']}
            **extra_attrs: Additional attributes to set on the mock diagnostic.
                          Common examples: climatology, seasonal_climatology, logger, etc.
        
        Returns:
            Mock diagnostic instance with:
            - data: Dict of mock variables with 'dims' attributes
            - Any additional attributes specified via **extra_attrs
        """
        mock_diag = mocker.Mock()
        
        # Create data dict with mock variables that have dims attributes
        mock_diag.data = {
            var: mocker.Mock(dims=dims)
            for var, dims in vars_dict.items()
        }
        
        # Set any additional attributes provided
        for attr_name, attr_value in extra_attrs.items():
            setattr(mock_diag, attr_name, attr_value)
        
        return mock_diag
    return _setup

