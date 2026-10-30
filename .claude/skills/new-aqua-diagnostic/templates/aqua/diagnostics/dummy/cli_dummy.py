"""Command-line interface for the Dummy diagnostic."""

import argparse
import sys

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.diagnostics.base import DiagnosticCLI, load_var_config, template_parse_arguments
from aqua.diagnostics.dummy import Dummy, PlotDummy

TOOLNAME = "dummy"


def parse_arguments(args):
    """Parse the command-line arguments (the shared AQUA set, nothing more)."""
    parser = argparse.ArgumentParser(description="Dummy CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    """Run the Dummy tool for every variable and region of the configuration."""
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name=TOOLNAME, default_config="config-dummy.yaml").prepare()
    cli.open_dask_cluster()
    try:
        tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME) or {}
        if not tool_dict.get("run", False):
            cli.logger.info("Dummy is disabled, nothing to do")
            return
        diagnostic_name = tool_dict.get("diagnostic_name", TOOLNAME)
        dataset = cli.config_dict["datasets"][0]
        reader_kwargs = dataset.get("reader_kwargs") or {}

        for var in tool_dict.get("variables", []):
            var_config, regions = load_var_config(cli.config_dict, var, diagnostic=TOOLNAME)
            for region in regions:
                diag = Dummy(
                    **cli.dataset_args(dataset), region=region, diagnostic_name=diagnostic_name, loglevel=cli.loglevel
                )
                try:
                    if not cli.plot_only:
                        diag.run(
                            var=var_config["name"],
                            reader_kwargs=reader_kwargs,
                            outputdir=cli.outputdir,
                            rebuild=cli.rebuild,
                            save_netcdf=cli.save_netcdf,
                        )
                    if cli.plot_only or cli.save_netcdf:
                        diag.load(var=var_config["name"], outputdir=cli.outputdir, reader_kwargs=reader_kwargs)
                except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
                    cli.logger.warning("Skipping %s in region %s: %s", var, region, e)
                    continue
                if diag.result is None:
                    cli.logger.warning("No results for %s in region %s, skipping the plot", var, region)
                    continue
                PlotDummy(
                    data=diag.result, diagnostic_name=diagnostic_name, outputdir=cli.outputdir, loglevel=cli.loglevel
                ).plot(save_format=cli.save_format, dpi=cli.dpi, rebuild=cli.rebuild)
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
