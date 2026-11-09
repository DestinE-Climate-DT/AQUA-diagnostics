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


def process_dataset(cli, dataset_args, reader_kwargs, var, region, diagnostic_name):
    """
    Run (or just load, in plot-only mode) the analysis of one dataset or reference.

    Args:
        cli (DiagnosticCLI): The prepared CLI.
        dataset_args (dict): From cli.dataset_args() for a model, cli.reference_args() for a reference.
        reader_kwargs (dict): The Reader keyword arguments of the dataset.
        var (str): The variable to analyse.
        region (str or None): The region, None for the global domain.
        diagnostic_name (str): The prefix of the output files.

    Returns:
        Dummy or None: The instance with its result, None if there is nothing to plot.
    """
    diag = Dummy(**dataset_args, region=region, diagnostic_name=diagnostic_name, loglevel=cli.loglevel)
    try:
        if not cli.plot_only:
            diag.run(
                var=var, reader_kwargs=reader_kwargs, outputdir=cli.outputdir, rebuild=cli.rebuild, save_netcdf=cli.save_netcdf
            )
        if cli.plot_only or cli.save_netcdf:
            diag.load(var=var, outputdir=cli.outputdir, reader_kwargs=reader_kwargs)
    except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
        cli.logger.warning(
            "Skipping %s/%s for %s in region %s: %s", dataset_args["model"], dataset_args["exp"], var, region, e
        )
        return None
    if diag.result is None:
        cli.logger.warning("No results for %s in region %s", var, region)
        return None
    return diag


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

        datasets = cli.config_dict["datasets"]
        if len(datasets) > 1:
            cli.logger.warning("Dummy supports one dataset, using only the first of %d", len(datasets))
        dataset = datasets[0]

        # REFERENCE (optional): drop this block, the `ref` lines below and the `references` block of the config
        # if the tool needs no reference; raise instead of continuing if the tool cannot work without one.
        references = cli.config_dict.get("references") or []
        if len(references) > 1:
            cli.logger.warning("Dummy supports one reference, using only the first of %d", len(references))
        reference = references[0] if references else None

        for var in tool_dict.get("variables", []):
            var_config, regions = load_var_config(cli.config_dict, var, diagnostic=TOOLNAME)
            for region in regions:
                diag = process_dataset(
                    cli,
                    cli.dataset_args(dataset),
                    dataset.get("reader_kwargs") or {},
                    var_config["name"],
                    region,
                    diagnostic_name,
                )
                if diag is None:
                    continue

                ref = None  # REFERENCE
                if reference is not None:  # REFERENCE
                    ref = process_dataset(  # REFERENCE
                        cli,
                        cli.reference_args(reference),
                        reference.get("reader_kwargs") or {},
                        var_config["name"],
                        region,
                        diagnostic_name,
                    )
                    if ref is None:  # REFERENCE
                        cli.logger.warning("No reference available for %s, plotting without it", var)  # REFERENCE

                PlotDummy(
                    data=diag.result,
                    ref_data=ref.result if ref is not None else None,  # REFERENCE
                    diagnostic_name=diagnostic_name,
                    outputdir=cli.outputdir,
                    loglevel=cli.loglevel,
                ).plot(save_format=cli.save_format, dpi=cli.dpi, rebuild=cli.rebuild)
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
