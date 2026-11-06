"""Command-line interface for the Overturning diagnostic."""

import argparse
import sys

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.diagnostics.base import DiagnosticCLI, load_var_config, template_parse_arguments
from aqua.diagnostics.overturning import Overturning, PlotOverturning

TOOLNAME = "overturning"
# Computation options read from params, passed to Overturning.run()
RUN_KEYS = ("latitudes", "min_depth", "lat_range", "mass_correction")
# Options of the streamfunction section read from plot_params
PLOT_KEYS = ("vmin", "vmax", "nlevels")


def parse_arguments(args):
    """Parse the command-line arguments (the shared AQUA set, nothing more)."""
    parser = argparse.ArgumentParser(description="Overturning CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def process_dataset(cli, dataset_args, reader_kwargs, var_config, region, diagnostic_name):
    """
    Run (or just load, in plot-only mode) the analysis of one dataset or reference.

    Args:
        cli (DiagnosticCLI): The prepared CLI.
        dataset_args (dict): From cli.dataset_args() for a model, cli.reference_args() for a reference.
        reader_kwargs (dict): The Reader keyword arguments of the dataset.
        var_config (dict): The variable configuration from load_var_config (name and computation options).
        region (str or None): The basin, None for the global ocean.
        diagnostic_name (str): The prefix of the output files.

    Returns:
        Overturning or None: The instance with its results, None if there is nothing to plot.
    """
    var = var_config["name"]
    try:
        diag = Overturning(
            **dataset_args,
            region=region,
            vert_coord=var_config.get("vert_coord"),
            diagnostic_name=diagnostic_name,
            loglevel=cli.loglevel,
        )
        if not cli.plot_only:
            diag.run(
                var=var,
                **{k: var_config[k] for k in RUN_KEYS if var_config.get(k) is not None},
                reader_kwargs=reader_kwargs,
                outputdir=cli.outputdir,
                rebuild=cli.rebuild,
                save_netcdf=cli.save_netcdf,
            )
        if cli.plot_only or cli.save_netcdf:
            diag.load(var=var, outputdir=cli.outputdir, reader_kwargs=reader_kwargs)
    except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
        cli.logger.warning("Skipping %s/%s for %s in basin %s: %s", dataset_args["model"], dataset_args["exp"], var, region, e)
        return None
    if diag.streamfunction is None and diag.index_monthly is None:
        cli.logger.warning("No results for %s in basin %s", var, region)
        return None
    return diag


def main(argv=None):
    """Run the Overturning tool for every variable and basin of the configuration."""
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name=TOOLNAME, default_config="config-overturning.yaml").prepare()
    cli.open_dask_cluster()
    try:
        tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME) or {}
        if not tool_dict.get("run", False):
            cli.logger.info("Overturning is disabled, nothing to do")
            return
        diagnostic_name = tool_dict.get("diagnostic_name", TOOLNAME)
        plot_params = (tool_dict.get("plot_params") or {}).get("default") or {}

        datasets = cli.config_dict["datasets"]
        if len(datasets) > 1:
            cli.logger.warning("Overturning supports one dataset, using only the first of %d", len(datasets))
        dataset = datasets[0]

        # REFERENCE (optional): a dataset with the meridional velocity, analysed like the model
        references = cli.config_dict.get("references") or []
        if len(references) > 1:
            cli.logger.warning("Overturning supports one reference, using only the first of %d", len(references))
        reference = references[0] if references else None

        for var in tool_dict.get("variables", ["von"]):
            var_config, regions = load_var_config(cli.config_dict, var, diagnostic=TOOLNAME)
            for region in regions:
                diag = process_dataset(
                    cli,
                    cli.dataset_args(dataset),
                    dataset.get("reader_kwargs") or {},
                    var_config,
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
                        var_config,
                        region,
                        diagnostic_name,
                    )
                    if ref is None:  # REFERENCE
                        cli.logger.warning("No reference available for %s, plotting without it", var)  # REFERENCE

                PlotOverturning(
                    streamfunction=diag.streamfunction,
                    index_monthly=diag.index_monthly,
                    index_annual=diag.index_annual,
                    ref_index_monthly=ref.index_monthly if ref is not None else None,  # REFERENCE
                    ref_index_annual=ref.index_annual if ref is not None else None,  # REFERENCE
                    diagnostic_name=diagnostic_name,
                    outputdir=cli.outputdir,
                    loglevel=cli.loglevel,
                ).plot(
                    plot_params={k: plot_params[k] for k in PLOT_KEYS if k in plot_params},
                    save_format=cli.save_format,
                    dpi=cli.dpi,
                    rebuild=cli.rebuild,
                )
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
