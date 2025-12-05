"""Helper functions for diagnostic CLI modules."""

def get_classes(cli, real_diagnostic, real_plot):
    """Get diagnostic classes (real or mock) based on cli.dry_run.
    
    Convenience wrapper for cli.get_diagnostic_classes() that provides
    a clean, reusable pattern for all diagnostic CLI modules.
    
    Args:
        cli: DiagnosticCLI instance with dry_run attribute
        real_diagnostic: Real diagnostic class (e.g., GlobalBiases)
        real_plot: Real plot class (e.g., PlotGlobalBiases)
    
    Returns:
        tuple: (DiagnosticClass, PlotClass) - mocks if cli.dry_run, real otherwise
    
    Example:
        from aqua.diagnostics.core.cli_helpers import get_classes
        from aqua.diagnostics import GlobalBiases, PlotGlobalBiases
        
        GlobalBiases, PlotGlobalBiases = get_classes(cli, GlobalBiases, PlotGlobalBiases)
    """
    return cli.get_diagnostic_classes(real_diagnostic, real_plot)

