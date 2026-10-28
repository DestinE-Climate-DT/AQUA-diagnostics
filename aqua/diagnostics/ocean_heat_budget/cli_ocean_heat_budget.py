"""Command-line interface for the OceanHeatBudget diagnostic."""

import argparse
import sys

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.ocean_heat_budget import OceanHeatBudget, PlotOceanHeatBudget

TOOLNAME = "ocean_heat_budget"


def parse_arguments(args):
    """Parse the command-line arguments (the shared AQUA set, nothing more)."""
    parser = argparse.ArgumentParser(description="OceanHeatBudget CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    """Run the OceanHeatBudget tool for every heat content variable of the configuration."""
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name=TOOLNAME, default_config="config-ocean_heat_budget.yaml").prepare()
    cli.open_dask_cluster()
    try:
        tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME) or {}
        if not tool_dict.get("run", False):
            cli.logger.info("OceanHeatBudget is disabled, nothing to do")
            return
        diagnostic_name = tool_dict.get("diagnostic_name", TOOLNAME)
        dataset = cli.config_dict["datasets"][0]
        reader_kwargs = dataset.get("reader_kwargs") or {}

        # One budget per heat content variable (e.g. hc700m, hc300m); there are no per-variable options
        for hc_var in tool_dict.get("variables", ["hc700m"]):
            diag = OceanHeatBudget(**cli.dataset_args(dataset), diagnostic_name=diagnostic_name, loglevel=cli.loglevel)
            try:
                if not cli.plot_only:
                    diag.run(
                        hc_var=hc_var,
                        reader_kwargs=reader_kwargs,
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                        save_netcdf=cli.save_netcdf,
                    )
                if cli.plot_only or cli.save_netcdf:
                    diag.load(hc_var=hc_var, outputdir=cli.outputdir, reader_kwargs=reader_kwargs)
            except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
                cli.logger.warning("Skipping %s: %s", hc_var, e)
                continue
            if diag.result is None:
                cli.logger.warning("No results for %s, skipping the plot", hc_var)
                continue
            PlotOceanHeatBudget(
                data=diag.result, diagnostic_name=diagnostic_name, outputdir=cli.outputdir, loglevel=cli.loglevel
            ).plot(save_format=cli.save_format, dpi=cli.dpi, rebuild=cli.rebuild)
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
