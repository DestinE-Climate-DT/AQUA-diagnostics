"""Command-line interface for GlobalBiases diagnostic."""

import argparse
import sys

from aqua.core.util import to_list
from aqua.core.exceptions import NoDataError
from aqua.diagnostics import GlobalBiases, PlotGlobalBiases
from aqua.diagnostics.core import template_parse_arguments
from aqua.diagnostics.core import DiagnosticCLI
from aqua.diagnostics.core.cli_helpers import get_classes

TOOLNAME='GlobalBiases'
TOOLNAME_KEY = TOOLNAME.lower()

def parse_arguments(args):
    """Parse command-line arguments for GlobalBiases diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description=f'{TOOLNAME} CLI')
    parser = template_parse_arguments(parser)
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        help='Run in dry-run mode (for testing, no real data operations)')
    return parser.parse_args(args)


def run_global_biases_diagnostic(cli, tool_dict):
    """Execute the Global Biases diagnostic workflow.
    
    This function contains the main execution logic for the Global Biases diagnostic,
    extracted from the __main__ block for better testability.
    
    Args:
        cli (DiagnosticCLI): Prepared DiagnosticCLI instance with loaded configuration
        tool_dict (dict): Diagnostic-specific configuration dictionary from config file
    
    Returns:
        bool: True if diagnostic ran successfully, False if disabled or skipped
    
    Raises:
        NoDataError: If required data cannot be retrieved (caught and logged)
        KeyError: If required configuration keys are missing (caught and logged)
        ValueError: If invalid parameters are provided (caught and logged)
    """
    if not tool_dict or not tool_dict.get('run', False):
        cli.logger.info(f"{TOOLNAME} diagnostic is disabled.")
        return False
    
    cli.logger.info(f"{TOOLNAME} diagnostic is enabled.")
    
    # Get classes (real or mock based on cli.dry_run)
    GlobalBiases, PlotGlobalBiases = get_classes(cli, GlobalBiases, PlotGlobalBiases)
    
    # Warn about multiple datasets/references
    if len(cli.config_dict['datasets']) > 1:
        cli.logger.warning(
            "Only the first entry in 'datasets' will be used.\n"
            "Multiple datasets are not supported by this diagnostic."
        )
    if len(cli.config_dict['references']) > 1:
        cli.logger.warning(
            "Only the first entry in 'references' will be used.\n"
            "Multiple references are not supported by this diagnostic."
        )
    
    # Extract configuration
    diagnostic_name = tool_dict.get('diagnostic_name', TOOLNAME_KEY)
    dataset = cli.config_dict['datasets'][0]
    reference = cli.config_dict['references'][0]
    dataset_args = cli.dataset_args(dataset)
    reference_args = cli.dataset_args(reference)
    
    variables = tool_dict.get('variables', [])
    formulae = tool_dict.get('formulae', [])
    plev = tool_dict.get('params', {}).get('default', {}).get('plev')
    seasons = tool_dict.get('params', {}).get('default', {}).get('seasons', False)
    seasons_stat = tool_dict.get('params', {}).get('default', {}).get('seasons_stat', 'mean')
    vertical = tool_dict.get('params', {}).get('default', {}).get('vertical', False)
    
    cli.logger.debug("Selected levels for vertical plots: %s", plev)
    
    # Create diagnostic instances
    biases_dataset = GlobalBiases(**dataset_args, diagnostic=diagnostic_name,
                                   outputdir=cli.outputdir, loglevel=cli.loglevel)
    biases_reference = GlobalBiases(**reference_args, diagnostic=diagnostic_name,
                                     outputdir=cli.outputdir, loglevel=cli.loglevel)
    
    all_vars = [(v, False) for v in variables] + [(f, True) for f in formulae]
    
    # Process each variable
    for var, is_formula in all_vars:
        cli.logger.info("Running Global Biases diagnostic for %s: %s",
                       "formula" if is_formula else "variable", var)
        
        # Extract plotting parameters
        all_plot_params = tool_dict.get('plot_params', {})
        default_params = all_plot_params.get('default', {})
        var_params = all_plot_params.get(var, {})
        plot_params = {**default_params, **var_params}
        
        vmin, vmax = plot_params.get('vmin'), plot_params.get('vmax')
        param_dict = tool_dict.get('params', {}).get(var, {})
        units = param_dict.get('units', None)
        long_name = param_dict.get('long_name', None)
        short_name = param_dict.get('short_name', None)
        
        # Retrieve data
        try:
            biases_dataset.retrieve(var=var, units=units, formula=is_formula,
                                   long_name=long_name, short_name=short_name,
                                   reader_kwargs=cli.reader_kwargs)
            biases_reference.retrieve(var=var, units=units, formula=is_formula,
                                     long_name=long_name, short_name=short_name)
        except (NoDataError, KeyError, ValueError) as e:
            cli.logger.warning("Variable '%s' not found in dataset. Skipping. (%s)", var, e)
            continue
        
        # Compute climatology
        biases_dataset.compute_climatology(seasonal=seasons, seasons_stat=seasons_stat,
                                          create_catalog_entry=cli.create_catalog_entry)
        biases_reference.compute_climatology(seasonal=seasons, seasons_stat=seasons_stat)
        
        if short_name is not None:
            var = short_name
        
        # Determine pressure levels
        if 'plev' in biases_dataset.data.get(var, {}).dims and plev:
            plev_list = to_list(plev)
        else:
            plev_list = [None]
        
        # Generate plots for each pressure level
        for p in plev_list:
            cli.logger.info(f"Processing variable: {var} at pressure level: {p}" if p 
                          else f"Processing variable: {var} at surface level")
            
            proj = plot_params.get('projection', 'robinson')
            proj_params = plot_params.get('projection_params', {})
            cmap = plot_params.get('cmap', 'RdBu_r')
            
            cli.logger.debug("Using projection: %s for variable: %s", proj, var)
            plot_biases = PlotGlobalBiases(diagnostic=diagnostic_name,
                                          save_pdf=cli.save_pdf, save_png=cli.save_png,
                                          dpi=cli.dpi, outputdir=cli.outputdir,
                                          cmap=cmap, loglevel=cli.loglevel)
            
            plot_biases.plot_bias(data=biases_dataset.climatology,
                                 data_ref=biases_reference.climatology,
                                 var=var, plev=p,
                                 proj=proj, proj_params=proj_params,
                                 vmin=vmin, vmax=vmax)
            
            if seasons:
                plot_biases.plot_seasonal_bias(data=biases_dataset.seasonal_climatology,
                                              data_ref=biases_reference.seasonal_climatology,
                                              var=var, plev=p,
                                              proj=proj, proj_params=proj_params,
                                              vmin=vmin, vmax=vmax)
        
        # Generate vertical bias plot if requested
        if vertical and 'plev' in biases_dataset.data.get(var, {}).dims:
            cli.logger.debug("Plotting vertical bias for variable:  %s", var)
            vmin_v, vmax_v = plot_params.get('vmin_v'), plot_params.get('vmax_v')
            plot_biases.plot_vertical_bias(data=biases_dataset.climatology,
                                          data_ref=biases_reference.climatology,
                                          var=var, vmin=vmin_v, vmax=vmax_v)
    
    return True


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments(sys.argv[1:])
    
    # Setup CLI
    cli = DiagnosticCLI(
        args,
        diagnostic_name=TOOLNAME_KEY,
        default_config='config_global_biases.yaml',
        dry_run=getattr(args, 'dry_run', False)
    )
    cli.prepare()
    cli.open_dask_cluster()
    
    # Execute diagnostic
    tool_dict = cli.config_dict['diagnostics'].get(TOOLNAME_KEY, {})
    success = run_global_biases_diagnostic(cli, tool_dict)
    
    # Cleanup
    cli.close_dask_cluster()
    
    if success:
        cli.logger.info("Global Biases diagnostic completed successfully.")
    else:
        cli.logger.info("Global Biases diagnostic was not executed.")
