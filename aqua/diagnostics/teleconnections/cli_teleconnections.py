#!/usr/bin/env python3
"""
Command-line interface for Teleconnections diagnostic.

This CLI allows to run the NAO and ENSO diagnostics.
Details of the run are defined in a yaml configuration file for a
single or multiple experiments.
"""

import argparse
import sys

from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.teleconnections import ENSO, NAO, PlotENSO, PlotNAO
from aqua.diagnostics.teleconnections.definitions import ENSO_DEFINITIONS, NAO_DEFINITIONS  # noqa: F401


def parse_arguments(args):
    """Parse command-line arguments for Teleconnections diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description="Teleconnections CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def produce_corr_reg_products(statistic: str, season: str = None, var: str = None):
    """Produce the diagnostic product name for correlation or regression.

    Args:
        statistic (str): 'correlation' or 'regression'.
        season (str, optional): season name. Defaults to None.
        var (str, optional): variable name. Defaults to None.

    Returns:
        str: diagnostic product name.
    """
    if var is not None and var != "default":
        product_name = f"{statistic}_{var}"
    else:
        product_name = statistic

    if season is not None and season != "annual":
        product_name = f"{product_name}_{season}"

    return product_name


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
                statistics_var = nao_config.get("statistics_var", ["default"])
                statistics_var = [var if var != "default" else NAO_DEFINITIONS["field"] for var in statistics_var]

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
                        if var == "default":
                            var = nao[
                                i
                            ].var  # The default variable is the one used defined in the configuration of the NAO diagnostic.
                        for season in seasons:
                            nao_regressions[var][season][i] = nao[i].compute_regression(var=var, season=season)
                            nao_correlations[var][season][i] = nao[i].compute_correlation(var=var, season=season)

                            diagnostic_product_reg = produce_corr_reg_products("regression", season, var)
                            diagnostic_product_cor = produce_corr_reg_products("correlation", season, var)

                            nao[i].save_netcdf(
                                nao_regressions[var][season][i],
                                diagnostic="nao",
                                diagnostic_product=diagnostic_product_reg,
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                            )
                            nao[i].save_netcdf(
                                nao_correlations[var][season][i],
                                diagnostic="nao",
                                diagnostic_product=diagnostic_product_cor,
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
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
                        if var == "default":
                            var = nao_ref[
                                i
                            ].var  # The default variable is the one used defined in the configuration of the NAO diagnostic.
                        for season in seasons:
                            nao_ref_regressions[var][season][i] = nao_ref[i].compute_regression(var=var, season=season)
                            nao_ref_correlations[var][season][i] = nao_ref[i].compute_correlation(var=var, season=season)

                            diagnostic_product_reg = produce_corr_reg_products("regression", season, var)
                            diagnostic_product_cor = produce_corr_reg_products("correlation", season, var)

                            nao_ref[i].save_netcdf(
                                nao_ref_regressions[var][season][i],
                                diagnostic="nao",
                                diagnostic_product=diagnostic_product_reg,
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
                            )
                            nao_ref[i].save_netcdf(
                                nao_ref_correlations[var][season][i],
                                diagnostic="nao",
                                diagnostic_product=diagnostic_product_cor,
                                outputdir=cli.outputdir,
                                rebuild=cli.rebuild,
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
                seasons = enso_config.get("seasons", "annual")

                enso_regressions = {season: [None] * len(config_dict["datasets"]) for season in seasons}
                enso_correlations = {season: [None] * len(config_dict["datasets"]) for season in seasons}

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

                    for season in seasons:
                        enso_regressions[season][i] = enso[i].compute_regression(season=season)
                        enso_correlations[season][i] = enso[i].compute_correlation(season=season)

                        diagnostic_product_reg = f"regression_{season}" if season != "annual" else "regression"
                        diagnostic_product_cor = f"correlation_{season}" if season != "annual" else "correlation"

                        enso[i].save_netcdf(
                            enso_regressions[season][i],
                            diagnostic="enso",
                            diagnostic_product=diagnostic_product_reg,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
                        )
                        enso[i].save_netcdf(
                            enso_correlations[season][i],
                            diagnostic="enso",
                            diagnostic_product=diagnostic_product_cor,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
                        )

                enso_ref = [None] * len(config_dict["references"])

                enso_ref_regressions = {season: [None] * len(config_dict["references"]) for season in seasons}
                enso_ref_correlations = {season: [None] * len(config_dict["references"]) for season in seasons}

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

                    for season in seasons:
                        enso_ref_regressions[season][i] = enso_ref[i].compute_regression(season=season)
                        enso_ref_correlations[season][i] = enso_ref[i].compute_correlation(season=season)

                        diagnostic_product_reg = f"regression_{season}" if season != "annual" else "regression"
                        diagnostic_product_cor = f"correlation_{season}" if season != "annual" else "correlation"

                        enso_ref[i].save_netcdf(
                            enso_ref_regressions[season][i],
                            diagnostic="enso",
                            diagnostic_product=diagnostic_product_reg,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
                        )
                        enso_ref[i].save_netcdf(
                            enso_ref_correlations[season][i],
                            diagnostic="enso",
                            diagnostic_product=diagnostic_product_cor,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
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
                    for season in seasons:
                        for i in range(len(enso)):
                            enso_regressions[season][i].load(keep_attrs=True)
                            enso_ref_regressions[season][i].load(keep_attrs=True)
                            enso_correlations[season][i].load(keep_attrs=True)
                            enso_ref_correlations[season][i].load(keep_attrs=True)

                        fig_reg = plot_enso.plot_maps(
                            maps=enso_regressions[season], ref_maps=enso_ref_regressions[season], statistic="regression"
                        )
                        fig_cor = plot_enso.plot_maps(
                            maps=enso_correlations[season], ref_maps=enso_ref_correlations[season], statistic="correlation"
                        )

                        regression_description = plot_enso.set_map_description(
                            maps=enso_regressions[season], ref_maps=enso_ref_regressions[season], statistic="regression"
                        )
                        correlation_description = plot_enso.set_map_description(
                            maps=enso_correlations[season], ref_maps=enso_ref_correlations[season], statistic="correlation"
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
