#!/usr/bin/env python3
"""Run the generic trend diagnostic from a YAML configuration.

Example:
    python -m aqua.diagnostics.trends.cli_trend --config config-trends.yaml

The CLI reads only ``diagnostics.trends``. Its configuration may also contain
``ocean_trends`` for the separate ocean CLI. All variables share the configured
regions and dimensional mean. Without ``--config``, the installed shared
``templates/collections/config-trends.yaml`` is used. Plotting supports horizontal maps;
set ``plot: false`` to save three-dimensional or spatially averaged coefficients.
"""

import argparse
import sys
from pathlib import Path

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.core.util import DEFAULT_REALIZATION, to_list
from aqua.diagnostics.base import DiagnosticCLI, get_diagnostic_configpath, template_parse_arguments
from aqua.diagnostics.trends import PlotTrends, Trends


def parse_arguments(args):
    """Parse the standard diagnostic CLI arguments.

    Args:
        args (list): Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generic Trend CLI")
    return template_parse_arguments(parser).parse_args(args)


def _save_and_plot(cli, diagnostic, coefficients, config):
    """Write coefficients and plot horizontal fields according to output settings."""
    if cli.save_netcdf and not cli.plot_only:
        diagnostic.save_netcdf(data=coefficients, outputdir=cli.outputdir, rebuild=cli.rebuild)

    if not config.get("plot", True) or not cli.save_format:
        return

    for var in coefficients.data_vars:
        field = coefficients[var]
        if "lat" not in field.coords or "lon" not in field.coords:
            raise ValueError(
                f"Trend map for '{var}' requires latitude and longitude coordinates. Set plot: false to save only."
            )
        horizontal_dims = set(field.lat.dims) | set(field.lon.dims)
        if not horizontal_dims or set(field.dims) != horizontal_dims:
            raise ValueError(f"Trend map for '{var}' requires a horizontal field. Set plot: false to save other dimensions.")

    plotter = PlotTrends(
        data=coefficients,
        diagnostic_name=diagnostic.diagnostic_name,
        outputdir=cli.outputdir,
        rebuild=cli.rebuild,
        loglevel=cli.loglevel,
    )
    plotter.plot_trend(
        **(config.get("plot_params") or {}),
        rebuild=cli.rebuild,
        save_format=cli.save_format,
        dpi=cli.dpi,
    )


def _process_dataset(cli, dataset, config):
    """Retrieve once and produce independent regional outputs for one dataset."""
    dataset_args = cli.dataset_args(dataset)
    reader_kwargs = dataset.get("reader_kwargs") or {}
    variables = to_list(config.get("var"))
    if not variables:
        raise ValueError("diagnostics.trends.var must specify at least one variable.")

    regions = to_list(config.get("regions")) or [None]
    dim_mean = config.get("dim_mean")

    if dim_mean is not None and not to_list(dim_mean):
        raise ValueError("dim_mean must be null or specify at least one dimension.")
    diagnostic_name = config.get("diagnostic_name", "trends")
    diagnostic = Trends(**dataset_args, diagnostic_name=diagnostic_name, loglevel=cli.loglevel)

    if cli.plot_only:
        diagnostic.realization = reader_kwargs.get("realization", DEFAULT_REALIZATION)
        for region in regions:
            region_name, _, _ = diagnostic._set_region(region=region)
            extra_keys = {}
            if region_name is not None:
                extra_keys["region"] = region_name
            if dim_mean is not None:
                extra_keys["dim_mean"] = "_".join(to_list(dim_mean))
            coefficients = diagnostic.load_netcdf(
                diagnostic=diagnostic_name,
                diagnostic_product="trend",
                outputdir=cli.outputdir,
                extra_keys=extra_keys,
                startdate=dataset_args["startdate"],
                enddate=dataset_args["enddate"],
                expect_netcdf=True,
            )
            if coefficients is None:
                raise NoDataError(f"No saved trend coefficients found for {dataset['model']} {dataset['exp']} in {region}.")
            _save_and_plot(cli, diagnostic, coefficients[variables], config)
        return

    diagnostic.retrieve(var=variables, reader_kwargs=reader_kwargs)
    if dim_mean is None:
        coefficients = diagnostic.compute_trend()
        # Execute once at output time; regional selection also gets concrete coordinates.
        if cli.save_netcdf or (config.get("plot", True) and cli.save_format):
            coefficients = coefficients.compute()
        for region in regions:
            regional = diagnostic.select_region(data=coefficients, region=region)["data"]
            _save_and_plot(cli, diagnostic, regional, config)
    else:
        # A dimensional mean depends on the region and must precede each fit.
        for region in regions:
            coefficients = diagnostic.compute_trend(region=region, dim_mean=dim_mean)
            _save_and_plot(cli, diagnostic, coefficients, config)


def main(argv=None):
    """Run generic trends for the configured datasets.

    Args:
        argv (list, optional): Arguments. Defaults to sys.argv[1:].

    Raises:
        ValueError: When variables or plot dimensions are invalid.
        NoDataError: When requested saved coefficients are unavailable.
        NotEnoughDataError: When the input does not meet annual coverage requirements.
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    if args.config is None:
        config_dir = get_diagnostic_configpath("trends", folder="templates", loglevel=args.loglevel or "WARNING")
        args.config = str(Path(config_dir) / "config-trends.yaml")
    cli = DiagnosticCLI(args, "trends", "config-trends.yaml", log_name="Trend CLI").prepare()
    config = (cli.config_dict.get("diagnostics") or {}).get("trends") or {}

    if not config.get("run", False):
        cli.logger.info("Generic trend diagnostic is disabled.")
        return

    datasets = cli.config_dict.get("datasets") or []
    if not datasets:
        raise ValueError("At least one dataset must be configured for generic trends.")

    try:
        cli.open_dask_cluster()
        for dataset in datasets:
            cli.logger.info("Processing trends for %s/%s/%s", dataset["model"], dataset["exp"], dataset["source"])
            _process_dataset(cli, dataset, config)
    except (NoDataError, NotEnoughDataError) as exc:
        cli.logger.error("Trend diagnostic failed: %s", exc)
        raise
    finally:
        cli.close_dask_cluster()


if __name__ == "__main__":
    main()
