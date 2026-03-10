#!/usr/bin/env python3
"""
Command-line interface for Teleconnections diagnostic.

This CLI allows to run the NAO and ENSO diagnostics.
Details of the run are defined in a yaml configuration file for a
single or multiple experiments.
"""
import argparse
import sys

from aqua.diagnostics.base import template_parse_arguments, DiagnosticCLI
from aqua.diagnostics.teleconnections import NAO, ENSO
from aqua.diagnostics.teleconnections import PlotNAO, PlotENSO


def parse_arguments(args):
    """Parse command-line arguments for Teleconnections diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description='Teleconnections CLI')
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def get_season_product(base_name: str, season: str) -> str:
    """
    Generate diagnostic product name based on season.

    Args:
        base_name (str): The base name of the diagnostic product.
        season (str): The season for which the product is generated.

    Returns:
        str: The generated diagnostic product name.
    """
    return f'{base_name}_{season}' if season != 'annual' else base_name


def process_teleconnection_data(
        diagnostic_class, data_sources, data_type, tc_config,
        diagnostic_name, cli, is_reference=False):
    """
    Process teleconnection data for datasets or references.

    Args:
        diagnostic_class: The diagnostic class (NAO or ENSO)
        data_sources: List of datasets or references to process
        data_type: 'datasets' or 'references'
        tc_config: Teleconnection configuration dictionary
        diagnostic_name: Name of diagnostic ('nao' or 'enso')
        cli: CLI object with configuration
        is_reference: Whether processing references (default: False)

    Returns:
        tuple: (diagnostics, regressions, correlations) dictionaries
    """
    seasons = tc_config.get('seasons', 'annual')
    init_args = {'loglevel': cli.loglevel}
    
    diagnostics = [None] * len(data_sources)
    regressions = {season: [None] * len(data_sources) for season in seasons}
    correlations = {season: [None] * len(data_sources) for season in seasons}
    
    for i, source in enumerate(data_sources):
        source_args = cli.reference_args(source) if is_reference else cli.dataset_args(source)
        cli.logger.info(f'Running {data_type}: {source_args}')
        
        diagnostics[i] = diagnostic_class(**source_args, **init_args)
        diagnostics[i].retrieve(reader_kwargs=cli.reader_kwargs if not is_reference else {})
        diagnostics[i].compute_index(months_window=tc_config.get('months_window', 3), rebuild=cli.rebuild)
        diagnostics[i].save_netcdf(
            diagnostics[i].index, diagnostic=diagnostic_name,
            diagnostic_product='index',
            outputdir=cli.outputdir, rebuild=cli.rebuild)
        
        for season in seasons:
            regressions[season][i] = diagnostics[i].compute_regression(season=season)
            correlations[season][i] = diagnostics[i].compute_correlation(season=season)
            
            reg_product = get_season_product('regression', season)
            cor_product = get_season_product('correlation', season)
            
            diagnostics[i].save_netcdf(
                regressions[season][i], diagnostic=diagnostic_name,
                diagnostic_product=reg_product,
                outputdir=cli.outputdir, rebuild=cli.rebuild)
            diagnostics[i].save_netcdf(
                correlations[season][i], diagnostic=diagnostic_name,
                diagnostic_product=cor_product,
                outputdir=cli.outputdir, rebuild=cli.rebuild)

    return diagnostics, regressions, correlations


def save_plot_formats(plotter, fig, diagnostic_product, description, cli):
    """Save plot in requested formats."""
    if cli.save_pdf:
        plotter.save_plot(
            fig, diagnostic_product=diagnostic_product, format='pdf',
            metadata={'description': description}, dpi=cli.dpi)
    if cli.save_png:
        plotter.save_plot(
            fig, diagnostic_product=diagnostic_product, format='png',
            metadata={'description': description}, dpi=cli.dpi)


def plot_teleconnection(
        plot_class, diagnostics, ref_diagnostics, regressions,
        ref_regressions, correlations, ref_correlations, seasons,
        cli, diagnostic_name):
    """Generate and save teleconnection plots.

    Args:
        plot_class: Plot class (PlotNAO or PlotENSO)
        diagnostics: List of diagnostic objects
        ref_diagnostics: List of reference diagnostic objects
        regressions: Dictionary of regression data by season
        ref_regressions: Dictionary of reference regression data by season
        correlations: Dictionary of correlation data by season
        ref_correlations: Dictionary of reference correlation data by season
        seasons: List of seasons to plot
        cli: CLI object with configuration
        diagnostic_name: Name of diagnostic ('NAO' or 'ENSO')
    """
    cli.logger.info(f'Plotting {diagnostic_name}')

    plotter = plot_class(
        indexes=[d.index for d in diagnostics],
        ref_indexes=[d.index for d in ref_diagnostics],
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        loglevel=cli.loglevel
    )

    # Plot index
    fig_index, _ = plotter.plot_index()
    index_description = plotter.set_index_description()
    save_plot_formats(plotter, fig_index, 'index', index_description, cli)

    # Plot regressions and correlations for each season
    for season in seasons:
        # Load data
        for i in range(len(diagnostics)):
            regressions[season][i].load(keep_attrs=True)
            ref_regressions[season][i].load(keep_attrs=True)
            correlations[season][i].load(keep_attrs=True)
            ref_correlations[season][i].load(keep_attrs=True)

        # Plot and save regressions
        fig_reg = plotter.plot_maps(
            maps=regressions[season], ref_maps=ref_regressions[season],
            statistic='regression')
        regression_description = plotter.set_map_description(
            maps=regressions[season], ref_maps=ref_regressions[season],
            statistic='regression')
        save_plot_formats(
            plotter, fig_reg, get_season_product('regression', season),
            regression_description, cli)

        # Plot and save correlations
        fig_cor = plotter.plot_maps(
            maps=correlations[season], ref_maps=ref_correlations[season],
            statistic='correlation')
        correlation_description = plotter.set_map_description(
            maps=correlations[season], ref_maps=ref_correlations[season],
            statistic='correlation')
        save_plot_formats(
            plotter, fig_cor, get_season_product('correlation', season),
            correlation_description, cli)


def run_teleconnection_diagnostic(tc_name: str, tc_class, plot_class, cli):
    """Run a teleconnection diagnostic (NAO or ENSO).

    Args:
        tc_name: Name of teleconnection ('NAO' or 'ENSO')
        tc_class: Diagnostic class (NAO or ENSO)
        plot_class: Plot class (PlotNAO or PlotENSO)
        cli: CLI object with configuration
    """
    config_dict = cli.config_dict
    tc_config = config_dict['diagnostics']['teleconnections'][tc_name]

    if not tc_config['run']:
        cli.logger.debug(f'Skipping {tc_name} teleconnections diagnostic as it is set to not run in the configuration.')
        return

    cli.logger.info(f'Running {tc_name} teleconnections diagnostic')
    diagnostic_name = tc_name.lower()
    seasons = tc_config.get('seasons', 'annual')

    # Process datasets
    diagnostics, regressions, correlations = process_teleconnection_data(
        tc_class, config_dict['datasets'], 'datasets', tc_config, diagnostic_name, cli, is_reference=False
    )

    # Process references
    ref_diagnostics, ref_regressions, ref_correlations = process_teleconnection_data(
        tc_class, config_dict['references'], 'references', tc_config, diagnostic_name, cli, is_reference=True
    )

    # Generate plots
    if cli.save_pdf or cli.save_png:
        plot_teleconnection(
            plot_class, diagnostics, ref_diagnostics, regressions,
            ref_regressions, correlations, ref_correlations, seasons, cli,
            tc_name)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    cli = DiagnosticCLI(
        args,
        diagnostic_name='teleconnections',
        default_config='config_teleconnections.yaml',
        log_name='Teleconnections CLI',
    ).prepare()
    cli.open_dask_cluster()

    if 'teleconnections' in cli.config_dict['diagnostics']:
        # Run NAO diagnostic if configured
        if 'NAO' in cli.config_dict['diagnostics']['teleconnections']:
            run_teleconnection_diagnostic('NAO', NAO, PlotNAO, cli)

        # Run ENSO diagnostic if configured
        if 'ENSO' in cli.config_dict['diagnostics']['teleconnections']:
            run_teleconnection_diagnostic('ENSO', ENSO, PlotENSO, cli)

    cli.close_dask_cluster()
    cli.logger.info('Teleconnections diagnostic completed.')
