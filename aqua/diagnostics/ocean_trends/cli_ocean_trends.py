#!/usr/bin/env python3
"""
Command-line interface for Ocean trends diagnostic.

This CLI allows to run the trends, OceanTrends diagnostics.
Details of the run are defined in a yaml configuration file for a
single or multiple experiments.
"""
import argparse
import sys

from aqua.core.util import get_arg
from aqua.diagnostics.base import template_parse_arguments
from aqua.diagnostics.ocean_trends import Trends
from aqua.diagnostics.ocean_trends import PlotTrends
from aqua.diagnostics.base import DiagnosticCLI


def parse_arguments(args):
    """Parse command-line arguments for OceanTrends diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description='OceanTrends CLI')
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    
    cli = DiagnosticCLI(args, 'ocean3d', 'config-ocean3d-en4-trend-drift.yaml', log_name='OceanTrends CLI').prepare()
    cli.open_dask_cluster()
    
    logger = cli.logger
    config_dict = cli.config_dict

    dataset = config_dict['datasets'][0]
    dataset_args = cli.dataset_args(dataset)

    #logger.info("Catalog: %s, Model: %s, Experiment: %s, Source: %s, Regrid: %s", catalog, model, exp, source, regrid)

    # Output options (from cli_base)
    reader_kwargs = cli.reader_kwargs
    outputdir = cli.outputdir
    rebuild = cli.rebuild
    save_pdf = cli.save_pdf
    save_png = cli.save_png
    dpi = cli.dpi

    if 'multilevel' in config_dict['diagnostics']['ocean_trends']:
        trends_config = config_dict['diagnostics']['ocean_trends']['multilevel']
        cli.logger.info("Ocean Trends diagnostic is set to %s", trends_config['run'])
        if trends_config['run']:
            regions = trends_config.get('regions', [None])
            diagnostic_name = trends_config.get('diagnostic_name', 'ocean_trends')
            var = trends_config.get('var', None)
            dim_mean = trends_config.get('dim_mean', None)
            vert_coord = trends_config.get('vert_coord', None)
            # Add the global region if not present
            # if regions != [None] or 'go' not in regions:
            #     regions.append('go')
            
            # Calculating Trend on whole dataset
            
            data_trends = Trends(
                **dataset_args,
                diagnostic_name=diagnostic_name,
                vert_coord=vert_coord,
                loglevel=cli.loglevel
            )
            data_trends.run(
                # region=region,
                var=var,
                # dim_mean=dim_mean,
                outputdir=outputdir,
                rebuild=rebuild,
                reader_kwargs=reader_kwargs
            )
            
            for region in regions:
                try:
                    cli.logger.info("Processing region: %s", region)
                    data_trends_region, region = data_trends.select_region(data=data_trends.trend_coef, region=region)

                    trends_plot = PlotTrends(
                        data=data_trends_region,
                        diagnostic_name=diagnostic_name,
                        vert_coord=vert_coord,
                        outputdir=outputdir,
                        rebuild=rebuild,
                        loglevel=cli.loglevel
                    )
                    trends_plot.plot_multilevel(save_pdf=save_pdf, save_png=save_png, dpi=dpi)

                    zonal_trend_plot = PlotTrends(
                        data=data_trends_region.mean('lon'),
                        diagnostic_name=diagnostic_name,
                        vert_coord=vert_coord,
                        outputdir=outputdir,
                        rebuild=rebuild,
                        loglevel=cli.loglevel
                    )
                    zonal_trend_plot.plot_zonal(save_pdf=save_pdf, save_png=save_png, dpi=dpi)
                except Exception as e:
                    cli.logger.error("Error processing region %s: %s", region, e)

    cli.close_dask_cluster()

    cli.logger.info("Ocean Trends diagnostic completed.")
