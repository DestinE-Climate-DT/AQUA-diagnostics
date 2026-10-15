#!/usr/bin/env python3
"""
Command-line interface for the Taylor diagnostic.

For each variable, the climatology of the first reference is computed once and every dataset is
compared with it. The statistics of all the variables end up in a single normalised Taylor diagram
per region, and, if seasons are enabled, a seasonal Taylor diagram is produced for each variable.
Details of the run are defined in a yaml configuration file for a single or multiple experiments.
"""

import argparse
import sys

from aqua.core.exceptions import NotEnoughDataError
from aqua.diagnostics.base import DiagnosticCLI, load_var_config, template_parse_arguments
from aqua.diagnostics.taylor import PlotTaylor, Taylor

TOOLNAME = "Taylor"
TOOLNAME_KEY = TOOLNAME.lower()


def parse_arguments(args):
    """Parse command-line arguments for the Taylor diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description=f"{TOOLNAME} CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def process_variable(cli, var_config, is_formula, region, datasets, reference, diagnostic_name, seasonal):
    """
    Compute, or load in a plot only run, the Taylor statistics of one variable for all the datasets.

    Args:
        cli: DiagnosticCLI instance with the prepared configuration
        var_config (dict): Variable configuration, merged with the params
        is_formula (bool): Whether the variable is a formula
        region (str): Region to select, None for the whole domain
        datasets (list): List of dataset configurations
        reference (dict): Reference configuration
        diagnostic_name (str): Name of the diagnostic, used in the filenames
        seasonal (bool): Whether to compute the seasonal statistics as well

    Returns:
        list: The statistics datasets, one per dataset for which they are available.
    """
    var_name = var_config.get("name")
    short_name = var_config.get("short_name")
    plev = var_config.get("plev")
    run_kwargs = {
        "var": var_name,
        "formula": is_formula,
        "long_name": var_config.get("long_name"),
        "short_name": short_name,
        "units": var_config.get("units"),
        "plev": plev,
        "seasonal": seasonal,
    }

    taylor_ref = Taylor(**cli.reference_args(reference), region=region, diagnostic_name=diagnostic_name, loglevel=cli.loglevel)
    if not cli.plot_only:
        try:
            taylor_ref.run(**run_kwargs, reader_kwargs=reference.get("reader_kwargs") or {})
        except NotEnoughDataError:
            cli.logger.warning(
                "Skipping %s: not enough data for reference %s %s", var_name, reference["model"], reference["exp"]
            )
            return []
        except Exception as e:
            cli.logger.error(
                "Skipping %s: unexpected error for reference %s %s: %s", var_name, reference["model"], reference["exp"], e
            )
            return []

    statistics = []
    for dataset in datasets:
        reader_kwargs = dataset.get("reader_kwargs") or {}
        taylor = Taylor(**cli.dataset_args(dataset), region=region, diagnostic_name=diagnostic_name, loglevel=cli.loglevel)

        if not cli.plot_only:
            try:
                taylor.run(**run_kwargs, reader_kwargs=reader_kwargs)
                taylor.compute_statistics(
                    reference=taylor_ref, save_netcdf=cli.save_netcdf, outputdir=cli.outputdir, rebuild=cli.rebuild
                )
            except NotEnoughDataError:
                cli.logger.warning(
                    "Skipping %s (%s, %s): not enough data", dataset["model"], dataset["exp"], dataset["source"]
                )
                continue
            except Exception as e:
                cli.logger.error(
                    "Unexpected error for %s (%s, %s): %s", dataset["model"], dataset["exp"], dataset["source"], e
                )
                continue

        # Populate from the netcdf files, written just now or by a previous run. Results computed
        # without saving them are already in memory, and older files must not replace them.
        if cli.plot_only or cli.save_netcdf:
            taylor.load(
                var=short_name or var_name,
                reference=taylor_ref,
                plev=plev,
                outputdir=cli.outputdir,
                reader_kwargs=reader_kwargs,
                expect_netcdf=True,
            )

        if taylor.statistics is not None:
            statistics.append(taylor.statistics)

    return statistics


def plot(cli, statistics, diagnostic_name, plot_params, **kwargs):
    """
    Plot a Taylor diagram, if any statistics and output format are available.

    Args:
        cli: DiagnosticCLI instance with the prepared configuration
        statistics (list): The statistics datasets to plot
        diagnostic_name (str): Name of the diagnostic, used in the filenames
        plot_params (dict): Plot options from the configuration file
        **kwargs: Additional arguments for PlotTaylor.plot_taylor
    """
    if not statistics:
        reason = ", and plot_only is true so nothing was computed" if cli.plot_only else ""
        cli.logger.warning("No Taylor statistics found in %s%s, skipping the plot", cli.outputdir, reason)
        return
    if not cli.save_format:
        cli.logger.debug("No plot output requested, skipping plot generation")
        return

    plot_taylor = PlotTaylor(
        diagnostic_name=diagnostic_name,
        save_format=cli.save_format,
        dpi=cli.dpi,
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        loglevel=cli.loglevel,
    )
    try:
        plot_taylor.plot_taylor(
            statistics=statistics,
            std_max=plot_params.get("std_max"),
            annotate=plot_params.get("annotate", False),
            **kwargs,
        )
    except Exception as e:
        cli.logger.error("Error plotting the Taylor diagram: %s", e)


def main(argv=None):
    """Run the Taylor diagnostic CLI.

    Args:
        argv (list, optional): command-line arguments. Defaults to sys.argv[1:].
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    cli = DiagnosticCLI(args, diagnostic_name=TOOLNAME_KEY, default_config="config-taylor.yaml").prepare()
    cli.open_dask_cluster()

    tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME_KEY, {})

    if tool_dict and tool_dict.get("run", False):
        cli.logger.info("%s diagnostic is enabled.", TOOLNAME)

        diagnostic_name = tool_dict.get("diagnostic_name", TOOLNAME_KEY)
        seasonal = tool_dict.get("seasons", True)
        plot_params = tool_dict.get("plot_params", {}) or {}
        datasets = cli.config_dict["datasets"]
        references = cli.config_dict.get("references") or []

        if not references:
            raise ValueError("The Taylor diagnostic needs a reference dataset, add one to 'references'")
        if len(references) > 1:
            cli.logger.warning("Only the first entry in 'references' will be used, multiple references are not supported")
        reference = references[0]

        variables = tool_dict.get("variables", []) or []
        formulae = tool_dict.get("formulae", []) or []
        all_vars = [(v, False) for v in variables] + [(f, True) for f in formulae]

        # Statistics of all the variables for each region, plotted together at the end
        statistics = {}
        for var, is_formula in all_vars:
            var_config, regions = load_var_config(cli.config_dict, var, diagnostic=TOOLNAME_KEY)
            cli.logger.info("Running %s for %s: %s", TOOLNAME, "formula" if is_formula else "variable", var_config["name"])

            for region in regions:
                cli.logger.info("Region: %s", region if region else "global")
                stats = process_variable(
                    cli=cli,
                    var_config=var_config,
                    is_formula=is_formula,
                    region=region,
                    datasets=datasets,
                    reference=reference,
                    diagnostic_name=diagnostic_name,
                    seasonal=seasonal,
                )
                statistics.setdefault(region, []).extend(stats)

                if seasonal:
                    plot(cli, stats, diagnostic_name, plot_params, period="seasonal", color_by="period", marker_by="dataset")

        for region, stats in statistics.items():
            cli.logger.info("Plotting the Taylor diagram of all the variables, region: %s", region if region else "global")
            plot(cli, stats, diagnostic_name, plot_params, period="annual", color_by="var", marker_by="dataset")

    cli.close_dask_cluster()

    cli.logger.info("%s diagnostic completed.", TOOLNAME)


if __name__ == "__main__":
    main()
