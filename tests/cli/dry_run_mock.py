"""Dry-run factory for CLI testing.

This module provides mock classes that mimic the interface of diagnostic
and plot classes, enabling dry-run testing of CLI functions without
performing real data operations.
"""


class MockDiagnostic:
    """Mock diagnostic class that mimics the interface of diagnostic classes.
    
    This class provides the same interface as diagnostic classes (e.g., GlobalBiases)
    but doesn't perform any real data operations. Used for testing CLI execution
    paths without loading data or performing computations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize mock diagnostic with stored arguments."""
        self.args = args
        self.kwargs = kwargs
        self.data = {}
        self.climatology = None
        self.seasonal_climatology = None
        self.logger_calls = []
    
    def retrieve(self, var, **kwargs):
        """Mock retrieve - creates a mock data object with dims attribute.
        
        Args:
            var: Variable name
            **kwargs: Additional arguments (ignored in mock)
        
        Returns:
            Mock data object with dims attribute
        """
        # Create a simple mock data object that has a 'dims' attribute
        # This mimics xarray DataArray behavior for dimension checking
        mock_data = type('MockData', (), {'dims': []})()
        self.data[var] = mock_data
        return mock_data
    
    def compute_climatology(self, **kwargs):
        """Mock compute_climatology - creates mock climatology objects.
        
        Args:
            **kwargs: Arguments including 'seasonal' flag
        
        Returns:
            Mock climatology object
        """
        self.climatology = type('MockClim', (), {})()
        if kwargs.get('seasonal'):
            self.seasonal_climatology = type('MockSeasonalClim', (), {})()
        return self.climatology


class MockPlot:
    """Mock plot class that mimics the interface of plot classes.
    
    This class provides the same interface as plot classes (e.g., PlotGlobalBiases)
    but doesn't generate actual plots. Used for testing CLI execution paths
    without creating plot files.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize mock plot with stored arguments."""
        self.args = args
        self.kwargs = kwargs
        self.plot_calls = []
    
    def plot_bias(self, **kwargs):
        """Mock plot_bias - records the call without generating a plot.
        
        Args:
            **kwargs: Plot arguments (recorded but not used)
        """
        self.plot_calls.append(('plot_bias', kwargs))
    
    def plot_seasonal_bias(self, **kwargs):
        """Mock plot_seasonal_bias - records the call without generating a plot.
        
        Args:
            **kwargs: Plot arguments (recorded but not used)
        """
        self.plot_calls.append(('plot_seasonal_bias', kwargs))
    
    def plot_vertical_bias(self, **kwargs):
        """Mock plot_vertical_bias - records the call without generating a plot.
        
        Args:
            **kwargs: Plot arguments (recorded but not used)
        """
        self.plot_calls.append(('plot_vertical_bias', kwargs))

