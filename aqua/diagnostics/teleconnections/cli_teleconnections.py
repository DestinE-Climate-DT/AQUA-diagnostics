#!/usr/bin/env python3
"""
Command-line interface for Teleconnections diagnostic.

This CLI allows to run the NAO, ENSO and DMI diagnostics.
Details of the run are defined in a yaml configuration file for a
single or multiple experiments.
"""

import argparse
import sys

from aqua.core.util import to_list
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.teleconnections import DMI, ENSO, NAO, PlotDMI, PlotENSO, PlotNAO

STATISTICS = ("regression", "correlation")


def parse_arguments(args):
    """Parse command-line arguments for Teleconnections diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description="Teleconnections CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def get_extra_keys(var: str, season: str) -> dict:
    """Filename keys of a regression or correlation map, the annual season is not written."""
    return {"var": var, "season": season} if season != "annual" else {"var": var}


def run_teleconnection(cli, tc_name: str, tc_class, plot_class, default_var: str):
    """Compute, save and plot the index and the regression and correlation maps of a teleconnection.

    Args:
        cli (DiagnosticCLI): prepared CLI object with configuration and output options.
        tc_name (str): name of the teleconnection block in the configuration ('NAO', 'ENSO' or 'DMI').
        tc_class: diagnostic class (NAO, ENSO or DMI).
        plot_class: plot class (PlotNAO, PlotENSO or PlotDMI).
        default_var (str): variable of the maps when 'statistics_var' is not set.
    """
    tc_config = cli.config_dict["diagnostics"]["teleconnections"][tc_name]
    diagnostic = tc_name.lower()
    seasons = to_list(tc_config.get("seasons", ["annual"]))
    statistics_var = to_list(tc_config.get("statistics_var", default_var))
    plot_params = tc_config.get("plot_params", {})
    # DMI has no months_window: it is passed only when configured, otherwise the class default is used
    index_kwargs = {"months_window": tc_config["months_window"]} if "months_window" in tc_config else {}

    # For each dataset and reference: (diagnostic, {(var, season, statistic): map})
    results = {"datasets": [], "references": []}
    for kind, get_args in (("datasets", cli.dataset_args), ("references", cli.reference_args)):
        for entry in cli.config_dict[kind]:
            args = get_args(entry)
            cli.logger.info("Running %s on %s: %s", tc_name, kind, args)

            tc = tc_class(**args, loglevel=cli.loglevel)
            tc.retrieve(reader_kwargs=entry.get("reader_kwargs") or {})
            tc.compute_index(rebuild=cli.rebuild, **index_kwargs)
            tc.save_netcdf(
                tc.index, diagnostic=diagnostic, diagnostic_product="index", outputdir=cli.outputdir, rebuild=cli.rebuild
            )

            maps = {}
            for var in statistics_var:
                # Units of the maps from the plot_params of the variable, if defined
                units = plot_params.get(var, {}).get("units")
                for season in seasons:
                    maps[var, season, "regression"] = tc.compute_regression(var=var, season=season, units=units)
                    maps[var, season, "correlation"] = tc.compute_correlation(var=var, season=season)
                    for statistic in STATISTICS:
                        tc.save_netcdf(
                            maps[var, season, statistic],
                            diagnostic=diagnostic,
                            diagnostic_product=statistic,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
                            extra_keys=get_extra_keys(var, season),
                        )
            results[kind].append((tc, maps))

    if not cli.save_format:
        return

    cli.logger.info("Plotting %s with formats: %s", tc_name, cli.save_format)
    plot = plot_class(
        indexes=[tc.index for tc, _ in results["datasets"]],
        ref_indexes=[tc.index for tc, _ in results["references"]],
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        loglevel=cli.loglevel,
    )

    fig, _ = plot.plot_index()
    plot.save_plot(
        fig,
        diagnostic_product="index",
        format=cli.save_format,
        metadata={"description": plot.set_index_description()},
        dpi=cli.dpi,
    )

    for var in statistics_var:
        # The default plot parameters are overridden field by field by the ones of the variable
        var_params = {**plot_params.get("default", {}), **plot_params.get(var, {})}
        map_params = {key: var_params.get(key) for key in ("vmin", "vmax", "vmin_diff", "vmax_diff")}
        map_params["cmap"] = var_params.get("cmap", "RdBu_r")

        for season in seasons:
            for statistic in STATISTICS:
                # Loaded once here, as both the plot and its description read them
                maps = [m[var, season, statistic].load() for _, m in results["datasets"]]
                ref_maps = [m[var, season, statistic].load() for _, m in results["references"]]

                fig = plot.plot_maps(maps=maps, ref_maps=ref_maps, statistic=statistic, **map_params)
                description = plot.set_map_description(maps=maps, ref_maps=ref_maps, statistic=statistic)
                plot.save_plot(
                    fig,
                    diagnostic_product=statistic,
                    format=cli.save_format,
                    metadata={"description": description},
                    extra_keys=get_extra_keys(var, season),
                    dpi=cli.dpi,
                )


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

    tc_dict = cli.config_dict["diagnostics"].get("teleconnections", {})
    # Built here and not at module level, so that the classes can be patched in the tests
    for tc_name, tc_class, plot_class, default_var in (
        ("NAO", NAO, PlotNAO, "msl"),
        ("ENSO", ENSO, PlotENSO, "tos"),
        ("DMI", DMI, PlotDMI, "tos"),
    ):
        if tc_dict.get(tc_name, {}).get("run", False):
            cli.logger.info("Running %s teleconnections diagnostic", tc_name)
            run_teleconnection(cli, tc_name, tc_class, plot_class, default_var)

    cli.close_dask_cluster()
    cli.logger.info("Teleconnections diagnostic completed.")


if __name__ == "__main__":
    main()
