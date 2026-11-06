"""Command-line interface for the Budget diagnostic."""

import argparse
import sys

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.diagnostics.base import DiagnosticCLI, load_var_config, template_parse_arguments
from aqua.diagnostics.budget import Budget, PlotBudget

TOOLNAME = "budget"


def parse_arguments(args):
    """Parse the command-line arguments (the shared AQUA set, nothing more)."""
    parser = argparse.ArgumentParser(description="Budget CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    """Run the Budget tool for every budget of the configuration."""
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name=TOOLNAME, default_config="config-budget.yaml").prepare()
    cli.open_dask_cluster()
    try:
        tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME) or {}
        if not tool_dict.get("run", False):
            cli.logger.info("Budget is disabled, nothing to do")
            return
        diagnostic_name = tool_dict.get("diagnostic_name", TOOLNAME)
        dataset = cli.config_dict["datasets"][0]
        reader_kwargs = dataset.get("reader_kwargs") or {}

        for budget in tool_dict.get("budgets", []):
            # params.default < params.<budget> < inline dict; a budget has no regions, it is global
            config, _ = load_var_config(cli.config_dict, budget, diagnostic=TOOLNAME)
            name = config["name"]
            diag = Budget(**cli.dataset_args(dataset), diagnostic_name=diagnostic_name, loglevel=cli.loglevel)
            try:
                if not cli.plot_only:
                    diag.run(
                        name=name,
                        first=config["first"],
                        second=config["second"],
                        tendency=config.get("tendency", False),
                        mask=config.get("mask"),
                        units=config.get("units"),
                        long_names=config.get("long_names"),
                        reader_kwargs=reader_kwargs,
                        outputdir=cli.outputdir,
                        rebuild=cli.rebuild,
                        save_netcdf=cli.save_netcdf,
                    )
                if cli.plot_only or cli.save_netcdf:
                    diag.load(name=name, outputdir=cli.outputdir, reader_kwargs=reader_kwargs)
            except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
                cli.logger.warning("Skipping budget %s: %s", name, e)
                continue
            if diag.result is None:
                cli.logger.warning("No results for budget %s, skipping the plot", name)
                continue
            PlotBudget(
                data=diag.result,
                name=name,
                expected=config.get("expected", 0.0),
                diagnostic_name=diagnostic_name,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
            ).plot(save_format=cli.save_format, dpi=cli.dpi, rebuild=cli.rebuild)
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
