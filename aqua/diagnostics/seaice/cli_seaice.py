#!/usr/bin/env python3
"""
Command-line interface for seaice diagnostic.

This CLI allows to perform multiple plots such as the timeseries
of integrated sea ice volume and extent from a yaml configuration
file for a single or multiple experiments with the possibility to
add reference data.
"""

import argparse
import sys

from aqua.core.util import get_arg
from aqua.diagnostics import Plot2DSeaIce, PlotSeaIce, SeaIce
from aqua.diagnostics.base import DiagnosticCLI, load_var_config, template_parse_arguments
from aqua.diagnostics.seaice.util import filter_region_list


def parse_arguments(args):
    """Parse command-line arguments for SeaIce diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description="SeaIce CLI")
    parser = template_parse_arguments(parser)

    # Add extra arguments
    parser.add_argument(
        "--proj",
        type=str,
        choices=["orthographic", "azimuthal_equidistant"],
        default="orthographic",
        help="Projection type for 2D plots (default: orthographic)",
    )
    return parser.parse_args(args)


def run_seaice(cli, tool, method, **run_kwargs):
    """
    Run SeaIce on the datasets and on the references of one method of a tool block.

    A reference is used for the methods listed in its ``methods`` (all of them if missing) and for the regions of
    its ``domain`` ('nh' or 'sh', all of them if missing).

    Args:
        cli (DiagnosticCLI): The prepared CLI.
        tool (str): The key of the tool block in the configuration.
        method (str): The sea ice method to compute.
        **run_kwargs: Additional keyword arguments for SeaIce.run().

    Returns:
        tuple: The SeaIce instances of the datasets and those of the references, with their results.
    """
    config = cli.config_dict
    params, regions = load_var_config(config, method, diagnostic=tool)
    regions = [region for region in regions if region is not None]  # [None] when no regions are configured
    common = {"diagnostic_name": config["diagnostics"][tool].get("diagnostic_name", "seaice"), "loglevel": cli.loglevel}
    run_kwargs = {
        "method": method,
        "var": params.get("var"),
        "annual": params.get("annual", False),
        "outputdir": cli.outputdir,
        "rebuild": cli.rebuild,
        "save_netcdf": cli.save_netcdf,
        **run_kwargs,
    }

    models = []
    for dataset in config["datasets"]:
        seaice = SeaIce(**cli.dataset_args(dataset), regions=regions, **common)
        seaice.run(reader_kwargs=dataset.get("reader_kwargs") or {}, **run_kwargs)
        models.append(seaice)

    refs = []
    for reference in config.get("references") or []:
        if method not in reference.get("methods", [method]):
            cli.logger.info("Skipping reference %s %s: not meant for %s", reference["model"], reference["exp"], method)
            continue
        ref_regions = models[0].regions
        if reference.get("domain"):
            ref_regions = filter_region_list(
                {"regions": models[0].regions_definition}, ref_regions, reference["domain"], cli.logger
            )
        if not ref_regions:
            cli.logger.info("Skipping reference %s %s: no region in its domain", reference["model"], reference["exp"])
            continue
        reference_args = cli.reference_args(reference)
        # a tool can set its own analysis window for the references
        reference_args["startdate"] = params.get("ref_startdate") or reference_args["startdate"]
        reference_args["enddate"] = params.get("ref_enddate") or reference_args["enddate"]
        seaice = SeaIce(
            **reference_args,
            std_startdate=reference.get("std_startdate") or params.get("std_startdate"),
            std_enddate=reference.get("std_enddate") or params.get("std_enddate"),
            regions=ref_regions,
            **common,
        )
        seaice.run(reader_kwargs=reference.get("reader_kwargs") or {}, std=params.get("std", False), **run_kwargs)
        refs.append(seaice)

    return models, refs


def main(argv=None):
    """Run the SeaIce diagnostic CLI.

    Args:
        argv (list, optional): command-line arguments. Defaults to sys.argv[1:].
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(
        args=args, diagnostic_name="seaice", default_config="config_seaice.yaml", log_name="SeaIce CLI"
    ).prepare()
    cli.open_dask_cluster()

    # Diagnostic-specific options
    projection = get_arg(args, "proj", "orthographic")

    diagnostics = cli.config_dict["diagnostics"]

    # ============= Sea Ice diagnostic - Timeseries and Seasonal Cycle ============
    # =============================================================================
    for tool, plot_type in (("seaice_timeseries", "timeseries"), ("seaice_seasonal_cycle", "seasonalcycle")):
        tool_dict = diagnostics.get(tool) or {}
        if not tool_dict.get("run", False):
            continue
        cli.logger.info("Executing Sea ice %s diagnostic for loaded config_dict.", plot_type)
        cycle = plot_type == "seasonalcycle"

        for method in tool_dict["methods"]:
            cli.logger.info("Method: %s", method)
            models, refs = run_seaice(cli, tool, method, seasonal_cycle=cycle)

            psi = PlotSeaIce(
                monthly_models=[m.seasonalcycle if cycle else m.monthly for m in models],
                annual_models=[m.annual for m in models],
                monthly_ref=[r.seasonalcycle if cycle else r.monthly for r in refs],
                annual_ref=[r.annual for r in refs],
                monthly_std_ref=[r.std_monthly for r in refs],
                annual_std_ref=[r.std_annual for r in refs],
                diagnostic_name=tool_dict.get("diagnostic_name", "seaice"),
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
                dpi=cli.dpi,
                loglevel=cli.loglevel,
            )
            psi.plot_seaice(plot_type=plot_type, save_format=cli.save_format)

    # ================ Sea Ice diagnostic - 2D Bias Maps ================
    # ===================================================================
    tool = "seaice_2d_bias"
    tool_dict = diagnostics.get(tool) or {}
    if tool_dict.get("run", False):
        cli.logger.info("Executing Sea ice 2D bias diagnostic for loaded config_dict.")
        plot_params = (tool_dict.get("plot_params") or {}).get("default") or {}

        # Loop over the methods (fraction and thickness)
        for method in tool_dict["methods"]:
            cli.logger.info("Method: %s", method)
            models, refs = run_seaice(cli, tool, method)
            if not refs:
                cli.logger.warning("No reference for sea ice %s: the difference maps need one, skipping", method)
                continue

            psi = Plot2DSeaIce(
                ref=[r.climatology for r in refs],
                models=[m.climatology for m in models],
                diagnostic_name=tool_dict.get("diagnostic_name", "seaice"),
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
                dpi=cli.dpi,
                loglevel=cli.loglevel,
            )
            psi.plot_2d_seaice(
                method=method,
                months=plot_params.get("months", [3, 9]),
                projection=projection,
                plot_ref_contour=method == "fraction",
                save_format=cli.save_format,
            )

    cli.close_dask_cluster()

    cli.logger.info("Sea Ice diagnostic completed.")


if __name__ == "__main__":
    main()
