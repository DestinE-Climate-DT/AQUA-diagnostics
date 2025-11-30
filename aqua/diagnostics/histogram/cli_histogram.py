#!/usr/bin/env python3
"""
Command-line interface for Histogram diagnostic.

This CLI allows to run the Histogram diagnostic to compute histograms and PDFs
of variables over specified regions. Details of the run are defined in a yaml 
configuration file for single or multiple experiments.
"""

import sys
import argparse
from aqua.diagnostics.core import template_parse_arguments
from aqua.diagnostics.core.cli_base import DiagnosticCLI
from aqua.diagnostics.histogram import Histogram, PlotHistogram
from aqua.diagnostics.histogram.util_cli import load_var_config


def parse_arguments(args):
    """Parse command-line arguments for Histogram diagnostic."""
    parser = argparse.ArgumentParser(description='Histogram CLI')
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def get_dataset_args(cli, dataset):
    """
    Helper to extract dataset arguments for Histogram diagnostic.
    
    Args:
        cli: DiagnosticCLI instance
        dataset (dict): Dataset configuration dictionary
        
    Returns:
        dict: Arguments for Histogram class initialization
    """
    return {
        'catalog': dataset.get('catalog'),
        'model': dataset.get('model'),
        'exp': dataset.get('exp'),
        'source': dataset.get('source'),
        'regrid': dataset.get('regrid', cli.regrid),
        'startdate': dataset.get('startdate'),
        'enddate': dataset.get('enddate')
    }


def process_dataset(cli, dataset, var_config, diag_config, region, is_reference=False):
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
    cli.logger.info(f"Processing {'reference' if is_reference else 'dataset'}: "
                   f"{dataset['model']}/{dataset['exp']}")
    
    # Get dataset arguments
    dataset_args = get_dataset_args(cli, dataset)
    
    # Extract variable info
    var_name = var_config.get('name')
    var_units = var_config.get('units')
    var_long_name = var_config.get('long_name')
    var_standard_name = var_config.get('standard_name')
    
    # Get lon/lat limits from variable config if specified
    lon_limits = var_config.get('lon_limits')
    lat_limits = var_config.get('lat_limits')
    
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
        long_name=var_long_name,
        units=var_units,
        standard_name=var_standard_name,
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
    # Check to see if any output is requested
    if not (cli.save_png or cli.save_pdf):
        cli.logger.debug('No plot output requested, skipping plot generation')
        return
    
    cli.logger.info('Creating histogram plots')
    
    # Collect histogram data
    data_list = [h.histogram_data for h in histograms]
    ref_data = histogram_ref.histogram_data if histogram_ref else None
    
    # Create plot object
    plot = PlotHistogram(
        data=data_list,
        ref_data=ref_data,
        diagnostic_name=diag_config.get('diagnostic_name', 'histogram'),
        loglevel=cli.loglevel
    )
    
    # Plot parameters
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
    
    # Save PNG if requested
    if cli.save_png:
        cli.logger.info('Saving PNG plot')
        plot.run(format='png', **plot_params)
    
    # Save PDF if requested
    if cli.save_pdf:
        cli.logger.info('Saving PDF plot')
        plot.run(format='pdf', **plot_params)


def process_variable(cli, var_config, regions, datasets, references, diag_config):
    """
    Process a single variable or formula across all datasets and regions.
    
    Args:
        cli: DiagnosticCLI instance with configuration
        var_config (dict): Variable configuration
        regions (list): List of regions to process
        datasets (list): List of dataset configurations
        references (list): List of reference dataset configurations
        diag_config (dict): Diagnostic configuration
    """
    var_name = var_config.get('name')
    is_formula = var_config.get('is_formula', False)
    
    cli.logger.info(
        "Running Histogram diagnostic for %s: %s",
        "formula" if is_formula else "variable", var_name)
    
    # Loop over regions
    for region in regions:
        cli.logger.info(f"Region: {region if region else 'global'}")
        
        # try-except block to catch errors per region
        try:
            # Process all datasets
            histograms = []
            for dataset in datasets:
                hist = process_dataset(cli, dataset, var_config, diag_config, 
                                     region, is_reference=False)
                histograms.append(hist)
            
            # Process reference dataset if present
            histogram_ref = None
            if references:
                ref_dataset = references[0]
                histogram_ref = process_dataset(cli, ref_dataset, var_config, 
                                              diag_config, region, is_reference=True)
            
            # Create and save plots
            create_and_save_plots(cli, histograms, histogram_ref, diag_config)
            
        except Exception as e:
            cli.logger.error(
                f"Error running Histogram diagnostic for variable {var_name} "
                f"in region {region if region else 'global'}: {e}")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    
    # Initialize and prepare CLI
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
        
        # Get datasets and references
        datasets = cli.config_dict.get('datasets', [])
        references = cli.config_dict.get('references', [])
        
        # unification of variables and formulae
        variables = diag_config.get('variables', [])
        formulae = diag_config.get('formulae', [])
        all_vars = [(v, False) for v in variables] + [(f, True) for f in formulae]
        
        for var, is_formula in all_vars:
            var_config, regions = load_var_config(cli.config_dict, var)
            var_config['is_formula'] = is_formula
            process_variable(cli, var_config, regions, datasets, references, diag_config)
    
    cli.close_dask_cluster()
    
    cli.logger.info("Histogram diagnostic completed.")