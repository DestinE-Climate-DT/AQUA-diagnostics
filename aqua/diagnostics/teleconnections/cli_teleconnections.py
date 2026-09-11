#!/usr/bin/env python3
"""
Command-line interface for Teleconnections diagnostic.

This CLI allows to run the NAO and ENSO diagnostics.
Details of the run are defined in a yaml configuration file for a
single or multiple experiments.
"""

import argparse
import sys

from aqua.core.util import to_list
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.teleconnections import ENSO, NAO, PlotENSO, PlotNAO


def parse_arguments(args):
    """Parse command-line arguments for Teleconnections diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description="Teleconnections CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    """Run the Teleconnections diagnostic CLI.

    Args:
        argv (list, optional): command-line arguments. Defaults to sys.argv[1:].
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    cli = DiagnosticCLI(
        args,
        diagnostic_name="teleconnections",
        default_config="config_teleconnections.yaml",
        log_name="Teleconnections CLI",
    ).prepare()
    cli.open_dask_cluster()

    logger = cli.logger
    config_dict = cli.config_dict

    if "teleconnections" in config_dict["diagnostics"]:
        # NAO
        if "NAO" in config_dict["diagnostics"]["teleconnections"]:
            if config_dict["diagnostics"]["teleconnections"]["NAO"]["run"]:
                logger.info("Running NAO teleconnections diagnostic")

                nao = [None] * len(config_dict["datasets"])

                nao_config = config_dict["diagnostics"]["teleconnections"]["NAO"]

                # We create lists for the seasons and statistics_var to handle multiple seasons and variables.
                seasons = nao_config.get("seasons", ["annual"])
                statistics_var = to_list(nao_config.get("statistics_var", "msl"))

                # We prepare dictionaries to store the regression and correlation results.
                # For each dataset we can have multiple seasons and multiple variables (statistics_var).
                # The nested dictionaries will have the structure:
                # {statistics_var: {season: [None] * len(config_dict["datasets"])}}.
                nao_regressions = {
                    var: {season: [None] * len(config_dict["datasets"]) for season in seasons} for var in statistics_var
                }
                nao_correlations = {
                    var: {season: [None] * len(config_dict["datasets"]) for season in seasons} for var in statistics_var
                }

                init_args = {"loglevel": cli.loglevel}

                for i, dataset in enumerate(config_dict["datasets"]):
                    dataset_args = cli.dataset_args(dataset)
                    logger.info(f"Running dataset: {dataset_args}")

                    nao[i] = NAO(**dataset_args, **init_args)
                    nao[i].retrieve(reader_kwargs=dataset.get("reader_kwargs") or {})
                    nao[i].compute_index(months_window=nao_config.get("months_window", 3), rebuild=cli.rebuild)

                    nao[i].save_netcdf(
                        nao[i].index,
                        diagnostic="nao",
                        diagnostic_product="index",
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                    )

                    # Loop over each variable in statistics_var and each season to compute regressions and correlations.
                    for var in statistics_var:
                        for season in seasons:
                            nao_regressions[var][season][i] = nao[i].compute_regression(var=var, season=season)
                            nao_correlations[var][season][i] = nao[i].compute_correlation(var=var, season=season)

                            extra_keys = {"var": var, "season": season} if season != "annual" else {"var": var}

                            nao[i].save_netcdf(
                                nao_regressions[var][season][i],
                                diagnostic="nao",
                                diagnostic_product="regression",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )
                            nao[i].save_netcdf(
                                nao_correlations[var][season][i],
                                diagnostic="nao",
                                diagnostic_product="correlation",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )

                nao_ref = [None] * len(config_dict["references"])

                nao_ref_regressions = {
                    var: {season: [None] * len(config_dict["references"]) for season in seasons} for var in statistics_var
                }
                nao_ref_correlations = {
                    var: {season: [None] * len(config_dict["references"]) for season in seasons} for var in statistics_var
                }

                for i, reference in enumerate(config_dict["references"]):
                    reference_args = cli.reference_args(reference)
                    logger.info(f"Running reference: {reference_args}")
                    nao_ref[i] = NAO(**reference_args, **init_args)
                    nao_ref[i].retrieve(reader_kwargs=reference.get("reader_kwargs") or {})
                    nao_ref[i].compute_index(months_window=nao_config.get("months_window", 3), rebuild=cli.rebuild)

                    nao_ref[i].save_netcdf(
                        nao_ref[i].index,
                        diagnostic="nao",
                        diagnostic_product="index",
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                    )

                    for var in statistics_var:
                        for season in seasons:
                            nao_ref_regressions[var][season][i] = nao_ref[i].compute_regression(var=var, season=season)
                            nao_ref_correlations[var][season][i] = nao_ref[i].compute_correlation(var=var, season=season)

                            extra_keys = {"var": var, "season": season} if season != "annual" else {"var": var}

                            nao_ref[i].save_netcdf(
                                nao_ref_regressions[var][season][i],
                                diagnostic="nao",
                                diagnostic_product="regression",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )
                            nao_ref[i].save_netcdf(
                                nao_ref_correlations[var][season][i],
                                diagnostic="nao",
                                diagnostic_product="correlation",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )

                # Plot NAO regressions
                if cli.save_format:
                    logger.info("Plotting NAO with formats: %s", cli.save_format)
                    plot_args = {
                        "indexes": [nao[i].index for i in range(len(nao))],
                        "ref_indexes": [nao_ref[i].index for i in range(len(nao_ref))],
                        "outputdir": cli.outputdir,
                        "rebuild": cli.rebuild,
                        "loglevel": cli.loglevel,
                    }

                    plot_nao = PlotNAO(**plot_args)

                    # Plot the NAO index
                    fig_index, _ = plot_nao.plot_index()
                    index_description = plot_nao.set_index_description()
                    plot_nao.save_plot(
                        fig_index,
                        diagnostic_product="index",
                        format=cli.save_format,
                        metadata={"description": index_description},
                        dpi=cli.dpi,
                    )

                    # Plot regressions and correlations
                    for var in statistics_var:
                        if var == "default":
                            var = nao[
                                i
                            ].var  # The default variable is the one used defined in the configuration of the NAO diagnostic.
                        for season in seasons:
                            for i in range(len(nao)):
                                nao_regressions[var][season][i].load(keep_attrs=True)
                                nao_ref_regressions[var][season][i].load(keep_attrs=True)
                                nao_correlations[var][season][i].load(keep_attrs=True)
                                nao_ref_correlations[var][season][i].load(keep_attrs=True)

                            fig_reg = plot_nao.plot_maps(
                                maps=nao_regressions[var][season],
                                ref_maps=nao_ref_regressions[var][season],
                                statistic="regression",
                            )
                            fig_cor = plot_nao.plot_maps(
                                maps=nao_correlations[var][season],
                                ref_maps=nao_ref_correlations[var][season],
                                statistic="correlation",
                            )

                            regression_description = plot_nao.set_map_description(
                                maps=nao_regressions[var][season],
                                ref_maps=nao_ref_regressions[var][season],
                                statistic="regression",
                            )
                            correlation_description = plot_nao.set_map_description(
                                maps=nao_correlations[var][season],
                                ref_maps=nao_ref_correlations[var][season],
                                statistic="correlation",
                            )

                            reg_product = f"regression_{season}" if season != "annual" else "regression"
                            cor_product = f"correlation_{season}" if season != "annual" else "correlation"

                            plot_nao.save_plot(
                                fig_reg,
                                diagnostic_product=reg_product,
                                format=cli.save_format,
                                metadata={"description": regression_description},
                                dpi=cli.dpi,
                            )
                            plot_nao.save_plot(
                                fig_cor,
                                diagnostic_product=cor_product,
                                format=cli.save_format,
                                metadata={"description": correlation_description},
                                dpi=cli.dpi,
                            )

        # ENSO
        if "ENSO" in config_dict["diagnostics"]["teleconnections"]:
            if config_dict["diagnostics"]["teleconnections"]["ENSO"]["run"]:
                logger.info("Running ENSO teleconnections diagnostic")

                enso = [None] * len(config_dict["datasets"])

                enso_config = config_dict["diagnostics"]["teleconnections"]["ENSO"]
                # We create lists for the seasons and statistics_var to handle multiple seasons and variables.
                seasons = enso_config.get("seasons", ["annual"])
                statistics_var = to_list(enso_config.get("statistics_var", "tos"))

                # We prepare dictionaries to store the regression and correlation results.
                # For each dataset we can have multiple seasons and multiple variables (statistics_var).
                # The nested dictionaries will have the structure:
                # {statistics_var: {season: [None] * len(config_dict["datasets"])}}.
                enso_regressions = {
                    var: {season: [None] * len(config_dict["datasets"]) for season in seasons} for var in statistics_var
                }
                enso_correlations = {
                    var: {season: [None] * len(config_dict["datasets"]) for season in seasons} for var in statistics_var
                }

                init_args = {"loglevel": cli.loglevel}

                for i, dataset in enumerate(config_dict["datasets"]):
                    dataset_args = cli.dataset_args(dataset)
                    logger.info(f"Running dataset: {dataset_args}")

                    enso[i] = ENSO(**dataset_args, **init_args)
                    enso[i].retrieve(reader_kwargs=dataset.get("reader_kwargs") or {})
                    enso[i].compute_index(months_window=enso_config.get("months_window", 3), rebuild=cli.rebuild)
                    enso[i].save_netcdf(
                        enso[i].index,
                        diagnostic="enso",
                        diagnostic_product="index",
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                    )

                    # Loop over each variable in statistics_var and each season to compute regressions and correlations.
                    for var in statistics_var:
                        for season in seasons:
                            enso_regressions[var][season][i] = enso[i].compute_regression(var=var, season=season)
                            enso_correlations[var][season][i] = enso[i].compute_correlation(var=var, season=season)

                            extra_keys = {"var": var, "season": season} if season != "annual" else {"var": var}

                            enso[i].save_netcdf(
                                enso_regressions[var][season][i],
                                diagnostic="enso",
                                diagnostic_product="regression",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )
                            enso[i].save_netcdf(
                                enso_correlations[var][season][i],
                                diagnostic="enso",
                                diagnostic_product="correlation",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )

                enso_ref = [None] * len(config_dict["references"])

                enso_ref_regressions = {
                    var: {season: [None] * len(config_dict["references"]) for season in seasons} for var in statistics_var
                }
                enso_ref_correlations = {
                    var: {season: [None] * len(config_dict["references"]) for season in seasons} for var in statistics_var
                }

                for i, reference in enumerate(config_dict["references"]):
                    reference_args = cli.reference_args(reference)
                    logger.info(f"Running reference: {reference_args}")

                    enso_ref[i] = ENSO(**reference_args, **init_args)
                    enso_ref[i].retrieve(reader_kwargs=reference.get("reader_kwargs") or {})
                    enso_ref[i].compute_index(months_window=enso_config.get("months_window", 3), rebuild=cli.rebuild)

                    enso_ref[i].save_netcdf(
                        enso_ref[i].index,
                        diagnostic="enso",
                        diagnostic_product="index",
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                    )

                    for var in statistics_var:
                        for season in seasons:
                            enso_ref_regressions[var][season][i] = enso_ref[i].compute_regression(var=var, season=season)
                            enso_ref_correlations[var][season][i] = enso_ref[i].compute_correlation(var=var, season=season)

                            extra_keys = {"var": var, "season": season} if season != "annual" else {"var": var}

                            enso_ref[i].save_netcdf(
                                enso_ref_regressions[var][season][i],
                                diagnostic="enso",
                                diagnostic_product="regression",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )
                            enso_ref[i].save_netcdf(
                                enso_ref_correlations[var][season][i],
                                diagnostic="enso",
                                diagnostic_product="correlation",
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                                extra_keys=extra_keys,
                            )

                # Plot ENSO regressions
                if cli.save_format:
                    logger.info("Plotting ENSO with formats: %s", cli.save_format)
                    plot_args = {
                        "indexes": [enso[i].index for i in range(len(enso))],
                        "ref_indexes": [enso_ref[i].index for i in range(len(enso_ref))],
                        "outputdir": cli.outputdir,
                        "rebuild": cli.rebuild,
                        "loglevel": cli.loglevel,
                    }

                    plot_enso = PlotENSO(**plot_args)

                    # Plot the ENSO index
                    fig_index, _ = plot_enso.plot_index()
                    index_description = plot_enso.set_index_description()
                    plot_enso.save_plot(
                        fig_index,
                        diagnostic_product="index",
                        format=cli.save_format,
                        metadata={"description": index_description},
                        dpi=cli.dpi,
                    )

                    # Plot regressions and correlations
                    for var in statistics_var:
                        if var == "default":
                            var = enso[
                                i
                            ].var  # The default variable is the one used defined in the configuration of the ENSO diagnostic.
                        for season in seasons:
                            for i in range(len(enso)):
                                enso_regressions[var][season][i].load(keep_attrs=True)
                                enso_ref_regressions[var][season][i].load(keep_attrs=True)
                                enso_correlations[var][season][i].load(keep_attrs=True)
                                enso_ref_correlations[var][season][i].load(keep_attrs=True)

                            fig_reg = plot_enso.plot_maps(
                                maps=enso_regressions[var][season],
                                ref_maps=enso_ref_regressions[var][season],
                                statistic="regression",
                            )
                            fig_cor = plot_enso.plot_maps(
                                maps=enso_correlations[var][season],
                                ref_maps=enso_ref_correlations[var][season],
                                statistic="correlation",
                            )

                            regression_description = plot_enso.set_map_description(
                                maps=enso_regressions[var][season],
                                ref_maps=enso_ref_regressions[var][season],
                                statistic="regression",
                            )
                            correlation_description = plot_enso.set_map_description(
                                maps=enso_correlations[var][season],
                                ref_maps=enso_ref_correlations[var][season],
                                statistic="correlation",
                            )

                            reg_product = f"regression_{season}" if season != "annual" else "regression"
                            cor_product = f"correlation_{season}" if season != "annual" else "correlation"

                            plot_enso.save_plot(
                                fig_reg,
                                diagnostic_product=reg_product,
                                format=cli.save_format,
                                metadata={"description": regression_description},
                                dpi=cli.dpi,
                            )
                            plot_enso.save_plot(
                                fig_cor,
                                diagnostic_product=cor_product,
                                format=cli.save_format,
                                metadata={"description": correlation_description},
                                dpi=cli.dpi,
                            )

    cli.close_dask_cluster()

    logger.info("Teleconnections diagnostic completed.")


if __name__ == "__main__":
    main()
