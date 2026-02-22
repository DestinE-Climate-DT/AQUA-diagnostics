#!/usr/bin/env python3
"""Command-line interface for Histogram diagnostic."""

import sys
import argparse
from aqua.diagnostics.base import template_parse_arguments, DiagnosticCLI
from aqua.diagnostics.histogram import Histogram, PlotHistogram

def parse_arguments(args):
    """Parse command-line arguments for Histogram diagnostic."""
    parser = argparse.ArgumentParser(description='Histogram CLI')
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)

def process_dataset(cli, dataset, var_name, var_config, diag_config, region, is_reference=False):
    """
    Process a single dataset for histogram computation.
    
    Args:
        cli: DiagnosticCLI instance with configuration
        dataset (dict): Dataset configuration
        var_config (dict): Variable configuration
        diag_config (dict): Diagnostic configuration
        region (str): Region to process
        is_reference (bool): Whether this is a reference dataset
        
    Returns:
        Histogram: Computed histogram object
    """
    cli.logger.info("Processing %s: %s/%s",
                    'reference' if is_reference else 'dataset', dataset['model'], dataset['exp'])
    
    # Get dataset arguments
    if is_reference:
        dataset_args = cli.reference_args(dataset)
    else:
        dataset_args = cli.dataset_args(dataset)
    
    # Extract variable info from params (already merged in var_config)
    units = var_config.get('units', None)
    long_name = var_config.get('long_name', None)
    standard_name = var_config.get('standard_name', None)
    lon_limits = var_config.get('lon_limits', None)
    lat_limits = var_config.get('lat_limits', None)
    
    # Create histogram object
    histogram = Histogram(
        **dataset_args,
        region=region,
        lon_limits=lon_limits,
        lat_limits=lat_limits,
        bins=diag_config.get('bins', 100),
        range=diag_config.get('range'),
        weighted=diag_config.get('weighted', True),
        diagnostic_name=diag_config.get('diagnostic_name', 'histogram'),
        loglevel=cli.loglevel
    )
    
    # Run the diagnostic
    histogram.run(
        var=var_name,
        formula=var_config.get('is_formula', False),
        long_name=long_name,
        units=units,
        standard_name=standard_name,
        box_brd=diag_config.get('box_brd', True),
        density=diag_config.get('density', True),
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        reader_kwargs=dataset.get('reader_kwargs') or cli.reader_kwargs or {}
    )
    
    return histogram

def create_and_save_plots(cli, histograms, histogram_ref, diag_config):
    """
    Create and save histogram plots.
    
    Args:
        cli: DiagnosticCLI instance with configuration
        histograms (list): List of Histogram objects
        histogram_ref (Histogram or None): Reference histogram
        diag_config (dict): Diagnostic configuration
    """
    if not (cli.save_png or cli.save_pdf):
        cli.logger.debug('No plot output requested, skipping plot generation')
        return
    
    cli.logger.info('Creating histogram plots')
    
    data_list = [h.histogram_data for h in histograms]
    ref_data = histogram_ref.histogram_data if histogram_ref else None
    
    plot = PlotHistogram(
        data=data_list,
        ref_data=ref_data,
        diagnostic_name=diag_config.get('diagnostic_name', 'histogram'),
        density=diag_config.get('density', True),
        loglevel=cli.loglevel
    )
    
    plot_params = {
        'outputdir': cli.outputdir,
        'rebuild': cli.rebuild,
        'dpi': cli.dpi,
        'xlogscale': diag_config.get('xlogscale', False),
        'ylogscale': diag_config.get('ylogscale', True),
        'smooth': diag_config.get('smooth', False),
        'smooth_window': diag_config.get('smooth_window', 5),
        'xmin': diag_config.get('xmin'),
        'xmax': diag_config.get('xmax'),
        'ymin': diag_config.get('ymin'),
        'ymax': diag_config.get('ymax')
    }
    
    if cli.save_png:
        cli.logger.info('Saving PNG plot')
        plot.run(format='png', **plot_params)
    if cli.save_pdf:
        cli.logger.info('Saving PDF plot')
        plot.run(format='pdf', **plot_params)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    
    cli = DiagnosticCLI(
        args,
        diagnostic_name='histogram',
        default_config='config-histogram.yaml',
        log_name='Histogram CLI'
    ).prepare()
    
    cli.open_dask_cluster()
    
    # Get diagnostic configuration
    diag_config = cli.config_dict['diagnostics'].get('histogram', {})
    
    if diag_config and diag_config.get('run', False):
        cli.logger.info("Histogram diagnostic is enabled.")
        
        datasets = cli.config_dict.get('datasets', [])
        references = cli.config_dict.get('references', [])
        
        # Get variables and formulae
        variables = diag_config.get('variables', [])
        formulae = diag_config.get('formulae', [])
        all_vars = [(v, False) for v in variables] + [(f, True) for f in formulae]
        
        for var, is_formula in all_vars:
            # Handle both dict and string formats
            if isinstance(var, dict):
                var_name = var.get('name')
                var_config = var.copy()  # Use the dict directly
            else:
                var_name = var
                var_config = {}
            
            cli.logger.info("Running Histogram diagnostic for %s: %s",
                          "formula" if is_formula else "variable", var_name)
            
            # Get params for this variable and merge with var_config
            param_dict = diag_config.get('params', {}).get(var_name, {})
            var_config = {**var_config, **param_dict}
            var_config['is_formula'] = is_formula
            
            # Get regions from merged config
            regions = var_config.get('regions', [None])
            
            for region in regions:
                cli.logger.info("Region: %s", region if region else 'global')
                
                try:
                    histograms = []
                    for dataset in datasets:
                        hist = process_dataset(cli, dataset, var_name, var_config, 
                                             diag_config, region, is_reference=False)
                        histograms.append(hist)
                    
                    histogram_ref = None
                    if references:
                        histogram_ref = process_dataset(cli, references[0], var_name, var_config,
                                                      diag_config, region, is_reference=True)
                    
                    create_and_save_plots(cli, histograms, histogram_ref, diag_config)
                    
                except Exception as e:
                    cli.logger.error("Error for variable %s in region %s: %s",
                                   var_name, region if region else 'global', e)
    
    cli.close_dask_cluster()
    
    cli.logger.info("Histogram diagnostic completed.")