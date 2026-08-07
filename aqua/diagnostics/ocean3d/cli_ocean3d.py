#!/usr/bin/env python3
"""Command-line interface for Ocean 3D diagnostics.

Runs Hovmöller (drift), multilevel trends, stratification, and MLD sections
based on which blocks are enabled in the YAML configuration.
"""

import argparse
import sys

from aqua.core.util import to_list
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.base.defaults import DEFAULT_OCEAN_VERT_COORD
from aqua.diagnostics.ocean_drift.hovmoller import Hovmoller
from aqua.diagnostics.ocean_drift.plot_hovmoller import PlotHovmoller
from aqua.diagnostics.ocean_stratification import PlotMLD, PlotStratification
from aqua.diagnostics.ocean_stratification.stratification import Stratification
from aqua.diagnostics.ocean_trends import PlotTrends, Trends

TOOLNAME = "Ocean3D"
TOOLNAME_KEY = "ocean3d"


def parse_arguments(args):
    """Parse command-line arguments for Ocean3D diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description=f"{TOOLNAME} CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def _run_hovmoller(cli, dataset_args, hovmoller_config):
    """Run Hovmöller drift diagnostic and plots for configured regions."""
    logger = cli.logger
    logger.info("Hovmoller diagnostic is set to %s", hovmoller_config["run"])
    if not hovmoller_config["run"]:
        return

    regions = to_list(hovmoller_config.get("regions", None))
    diagnostic_name = hovmoller_config.get("diagnostic_name", "ocean_drift")
    var = hovmoller_config.get("var", None)
    dim_mean = hovmoller_config.get("dim_mean", ["lat", "lon"])
    vert_coord = hovmoller_config.get("vert_coord", DEFAULT_OCEAN_VERT_COORD)

    for region in regions:
        logger.info("Processing region: %s", region)
        data_hovmoller = None
        try:
            data_hovmoller = Hovmoller(
                **dataset_args, diagnostic_name=diagnostic_name, vert_coord=vert_coord, loglevel=cli.loglevel
            )
            data_hovmoller.run(
                region=region,
                var=var,
                dim_mean=dim_mean,
                anomaly_ref="t0",
                outputdir=cli.outputdir,
                reader_kwargs=cli.reader_kwargs,
                rebuild=cli.rebuild,
            )
        except Exception as e:
            logger.error("Error processing region %s: %s", region, e)
            continue

        try:
            if not cli.save_format:
                logger.debug("No plot output requested, skipping plot generation for region %s", region)
                continue

            hov_plot = PlotHovmoller(
                diagnostic_name=diagnostic_name,
                data=data_hovmoller.processed_data_list,
                vert_coord=vert_coord,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
            )
            logger.info("Saving Hovmoller plots for region %s with formats: %s", region, cli.save_format)
            hov_plot.plot_hovmoller(rebuild=cli.rebuild, save_format=cli.save_format, dpi=cli.dpi)
            hov_plot.plot_timeseries(rebuild=cli.rebuild, save_format=cli.save_format, dpi=cli.dpi)
        except Exception as e:
            logger.error("Error plotting region %s: %s", region, e)


def _run_trends(cli, dataset_args, trends_config):
    """Run multilevel ocean trends diagnostic and plots."""
    logger = cli.logger
    logger.info("Ocean Trends diagnostic is set to %s", trends_config["run"])
    if not trends_config["run"]:
        return

    regions = trends_config.get("regions", [None])
    diagnostic_name = trends_config.get("diagnostic_name", "ocean_trends")
    var = trends_config.get("var", None)
    vert_coord = trends_config.get("vert_coord", DEFAULT_OCEAN_VERT_COORD)

    data_trends = Trends(**dataset_args, diagnostic_name=diagnostic_name, vert_coord=vert_coord, loglevel=cli.loglevel)
    data_trends.run(
        var=var,
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        reader_kwargs=cli.reader_kwargs,
    )

    for region in regions:
        try:
            logger.info("Processing region: %s", region)
            data_trends_region, region = data_trends.select_region(data=data_trends.trend_coef, region=region)

            trends_plot = PlotTrends(
                data=data_trends_region,
                diagnostic_name=diagnostic_name,
                vert_coord=vert_coord,
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
                loglevel=cli.loglevel,
            )
            trends_plot.plot_multilevel(
                levels=[10, 100, 500, 1000],
                cbar_limits={
                    "thetao": {"vmin": -0.7, "vmax": 0.7},
                    "so": {"vmin": -0.12, "vmax": 0.12},
                },
                sym=True,
                save_format=cli.save_format,
                dpi=cli.dpi,
            )

            zonal_trend_plot = PlotTrends(
                data=data_trends_region.mean("lon"),
                diagnostic_name=diagnostic_name,
                vert_coord=vert_coord,
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
                loglevel=cli.loglevel,
            )
            zonal_trend_plot.plot_zonal(save_format=cli.save_format, dpi=cli.dpi)
        except Exception as e:
            logger.error("Error processing region %s: %s", region, e)


def _run_stratification(cli, dataset_args, reference_args, stratification_config):
    """Run stratification profiles for model (and optional reference)."""
    logger = cli.logger
    logger.info("Stratification diagnostic is set to %s", stratification_config["run"])
    if not stratification_config["run"]:
        return

    regions = to_list(stratification_config.get("regions", None))
    diagnostic_name = stratification_config.get("diagnostic_name", "ocean_stratification")
    climatologies = stratification_config.get("climatology", None)
    vert_coord = stratification_config.get("vert_coord", DEFAULT_OCEAN_VERT_COORD)
    var = stratification_config.get("var", None)
    dim_mean = ["lat", "lon"]

    for region, climatology in zip(regions, climatologies):
        logger.info("Processing region: %s, climatology: %s", region, climatology)
        model_stratification = Stratification(
            **dataset_args,
            diagnostic_name=diagnostic_name,
            vert_coord=vert_coord,
            loglevel=cli.loglevel,
        )
        model_stratification.run(
            region=region,
            var=var,
            dim_mean=dim_mean,
            mld=False,
            climatology=climatology,
            outputdir=cli.outputdir,
            reader_kwargs=cli.reader_kwargs,
            rebuild=cli.rebuild,
        )

        obs_stratification = None
        if reference_args is not None:
            logger.info("Processing reference data")
            obs_stratification = Stratification(
                **reference_args,
                diagnostic_name=diagnostic_name,
                vert_coord=vert_coord,
                loglevel=cli.loglevel,
            )
            obs_stratification.run(
                region=region,
                var=var,
                dim_mean=dim_mean,
                mld=False,
                climatology=climatology,
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
            )

        strat_plot = PlotStratification(
            data=model_stratification.data[["thetao", "so", "rho"]],
            obs=(obs_stratification.data[["thetao", "so", "rho"]] if obs_stratification is not None else None),
            diagnostic_name=diagnostic_name,
            vert_coord=vert_coord,
            outputdir=cli.outputdir,
            loglevel=cli.loglevel,
        )
        strat_plot.plot_stratification(save_format=cli.save_format, dpi=cli.dpi)


def _run_mld(cli, dataset_args, reference_args, mld_config):
    """Run mixed-layer depth diagnostic for model (and optional reference)."""
    logger = cli.logger
    logger.info("MLD diagnostic is set to %s", mld_config["run"])
    if not mld_config["run"]:
        return

    regions = to_list(mld_config.get("regions", None))
    diagnostic_name = mld_config.get("diagnostic_name", "ocean_stratification")
    climatologies = mld_config.get("climatology", None)
    vert_coord = mld_config.get("vert_coord", None)
    var = mld_config.get("var", None)

    for region, climatology in zip(regions, climatologies):
        logger.info("Processing region: %s, climatology: %s", region, climatology)
        model_stratification = Stratification(
            **dataset_args,
            diagnostic_name=diagnostic_name,
            vert_coord=vert_coord,
            loglevel=cli.loglevel,
        )
        model_stratification.run(
            region="go",
            var=var,
            mld=True,
            climatology=climatology,
            outputdir=cli.outputdir,
            reader_kwargs=cli.reader_kwargs,
            rebuild=cli.rebuild,
        )

        obs_stratification = None
        if reference_args is not None:
            logger.info("Processing reference data")
            obs_stratification = Stratification(
                **reference_args,
                diagnostic_name=diagnostic_name,
                vert_coord=vert_coord,
                loglevel=cli.loglevel,
            )
            obs_stratification.run(
                region="go",
                var=var,
                mld=True,
                climatology=climatology,
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
            )

        mld_plot = PlotMLD(
            data=model_stratification.data[["mld"]],
            obs=(obs_stratification.data[["mld"]] if obs_stratification is not None else None),
            diagnostic_name=diagnostic_name,
            outputdir=cli.outputdir,
            loglevel=cli.loglevel,
        )
        mld_plot.plot_mld(region=region, proj_name="Orthographic", save_format=cli.save_format, dpi=cli.dpi)


def main(argv=None):
    """Run the Ocean3D diagnostic CLI.

    Sections are activated from config under diagnostics.ocean_drift,
    diagnostics.ocean_trends, and diagnostics.ocean_stratification.
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    cli = DiagnosticCLI(
        args,
        diagnostic_name=TOOLNAME_KEY,
        default_config="config-ocean3d-en4-drift.yaml",
        log_name=f"{TOOLNAME} CLI",
    ).prepare()
    cli.open_dask_cluster()

    logger = cli.logger
    config_dict = cli.config_dict
    diags = config_dict.get("diagnostics", {})

    dataset = config_dict["datasets"][0]
    dataset_args = cli.dataset_args(dataset)
    logger.debug("Dataset args: %s", dataset_args)

    reference_args = None
    if "references" in config_dict and config_dict["references"]:
        references = config_dict["references"]
        logger.info("References found: %s", references)
        reference_args = cli.reference_args(references[0])
        logger.debug("Reference args: %s", reference_args)

    drift_cfg = diags.get("ocean_drift", {})
    if "hovmoller" in drift_cfg:
        _run_hovmoller(cli, dataset_args, drift_cfg["hovmoller"])

    trends_cfg = diags.get("ocean_trends", {})
    if "multilevel" in trends_cfg:
        _run_trends(cli, dataset_args, trends_cfg["multilevel"])

    strat_cfg = diags.get("ocean_stratification", {})
    if "stratification" in strat_cfg:
        _run_stratification(cli, dataset_args, reference_args, strat_cfg["stratification"])
    if "mld" in strat_cfg:
        _run_mld(cli, dataset_args, reference_args, strat_cfg["mld"])

    cli.close_dask_cluster()
    logger.info("%s diagnostic completed.", TOOLNAME)


if __name__ == "__main__":
    main()
