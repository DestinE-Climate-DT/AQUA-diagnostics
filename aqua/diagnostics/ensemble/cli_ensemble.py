#!/usr/bin/env python3
"""
Command-line interface for ensemble diagnostics.

Runs EnsembleTimeseries and/or EnsembleLatLon diagnostics for a single model.
Each diagnostic reads its own original YAML configuration file:

  - config_single_model_timeseries_ensemble.yaml  (diagnostics.ensemble block)
  - config_single_model_latlon_ensemble.yaml       (diagnostics.ensemble block)

Which diagnostics run is controlled by the ``run`` flag inside each config's
``diagnostics.ensemble`` block, exactly as before.  Both configs are loaded
every time; set ``run: false`` in either one to skip it.

CLI overrides (``--catalog``, ``--model``, ``--exp``, ``--source``,
``--outputdir``, ``--loglevel``) are applied to both configs.
"""

import argparse
import sys

from aqua.core.logger import log_configure
from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleLatLon,
    EnsembleTimeseries,
    PlotEnsembleLatLon,
    PlotEnsembleTimeseries,
    extract_realizations,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    close_cluster,
    load_diagnostic_config,
    merge_config_args,
    open_cluster,
    template_parse_arguments,
)

# Default config filenames (resolved by load_diagnostic_config from the
# package's config/collections/legacy/ensemble/ directory)
_DEFAULT_CONFIG_TIMESERIES = "config_single_model_timeseries_ensemble.yaml"
_DEFAULT_CONFIG_LATLON     = "config_single_model_latlon_ensemble.yaml"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments(args):
    """Parse command-line arguments for the unified ensemble diagnostic CLI.

    Args:
        args (list): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified Ensemble diagnostic CLI. "
            "Runs EnsembleTimeseries and EnsembleLatLon back-to-back, "
            "each driven by its own YAML config file. "
            "Use --config-timeseries / --config-latlon to supply custom paths."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    # Optional per-diagnostic config overrides (in addition to the shared --config
    # already added by template_parse_arguments)
    parser.add_argument(
        "--config-timeseries",
        dest="config_timeseries",
        default=None,
        help=(
            "Path to the timeseries config YAML. "
            f"Defaults to {_DEFAULT_CONFIG_TIMESERIES}."
        ),
    )
    parser.add_argument(
        "--config-latlon",
        dest="config_latlon",
        default=None,
        help=(
            "Path to the latlon config YAML. "
            f"Defaults to {_DEFAULT_CONFIG_LATLON}."
        ),
    )
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_options(config_dict):
    """Extract output options from a configuration dictionary.

    Args:
        config_dict (dict): full configuration dictionary.

    Returns:
        dict: keys outputdir, rebuild, save_netcdf, save_format, dpi.
    """
    out = config_dict.get("output", {})
    return {
        "outputdir":   out.get("outputdir",   "./"),
        "rebuild":     out.get("rebuild",     True),
        "save_netcdf": out.get("save_netcdf", True),
        "save_format": out.get("save_format", SAVE_FORMAT),
        "dpi":         out.get("dpi",         300),
    }


def _resolve_dataset(args, config_dict, logger):
    """Extract dataset identifiers from config, with CLI overrides applied.

    Args:
        args (argparse.Namespace): parsed CLI arguments.
        config_dict (dict): full configuration dictionary.
        logger: configured logger instance.

    Returns:
        tuple: (catalog, model, exp, source, regrid), or raises SystemExit
               if no datasets block is present.
    """
    datasets = config_dict.get("datasets")
    if not datasets:
        logger.error("No datasets configured in config file. Aborting.")
        sys.exit(1)

    first   = datasets[0]
    catalog = get_arg(args, "catalog", first["catalog"])
    model   = get_arg(args, "model",   first["model"])
    exp     = get_arg(args, "exp",     first["exp"])
    source  = get_arg(args, "source",  first["source"])
    regrid  = get_arg(args, "regrid",  first.get("regrid"))
    return catalog, model, exp, source, regrid


def _retrieve_dataset(variable, catalog, model, exp, source,
                      realization_dict, region=None,
                      startdate=None, enddate=None, logger=None):
    """Retrieve and merge ensemble data for one variable.

    Args:
        variable (str): variable name.
        catalog (str): catalog identifier.
        model (str): model identifier.
        exp (str): experiment identifier.
        source (str): source identifier.
        realization_dict (dict): mapping of model name to realization list.
        region (str | None): optional region filter.
        startdate (str | None): optional start date string.
        enddate (str | None): optional end date string.
        logger: configured logger instance.

    Returns:
        xarray.Dataset | None: merged dataset, or None if retrieval failed.
    """
    dataset = reader_retrieve_and_merge(
        variable=variable,
        catalog_list=catalog,
        model_list=model,
        exp_list=exp,
        source_list=source,
        region=region,
        startdate=startdate,
        enddate=enddate,
        realization=realization_dict,
    )
    if dataset is None and logger is not None:
        logger.warning(
            "Ensemble data retrieval returned None for variable '%s'.", variable
        )
    return dataset


# ---------------------------------------------------------------------------
# Diagnostic runners
# ---------------------------------------------------------------------------

def run_timeseries(config_dict, args, loglevel, logger):
    """Execute the EnsembleTimeseries diagnostic and plot loop.

    Reads all parameters directly from *config_dict*, which is the already-
    loaded and CLI-merged timeseries configuration dictionary.

    Args:
        config_dict (dict): merged timeseries configuration dictionary.
        args (argparse.Namespace): parsed CLI arguments (used for output dir
            override only; dataset fields are already merged into config_dict).
        loglevel (str): log level string passed to class constructors.
        logger: configured logger instance.
    """
    diag_config = config_dict["diagnostics"]["ensemble"]
    output_opts = _output_options(config_dict)

    params      = diag_config.get("params", {}).get("default", {})
    plot_params = diag_config.get("plot_params", {}).get("default", {})

    startdate_data        = params.get("startdate_data")
    enddate_data          = params.get("enddate_data")
    title                 = plot_params.get("title")
    plot_ensemble_members = plot_params.get("plot_ensemble_members", True)

    variables = diag_config.get("variable") or []
    regions   = diag_config.get("region") or []

    catalog, model, exp, source, _regrid = _resolve_dataset(
        args, config_dict, logger
    )
    realization      = extract_realizations(catalog=catalog, model=model,
                                            exp=exp,     source=source)
    realization_dict = {model: realization}

    for variable in variables:
        for region in regions:
            logger.info("Timeseries — variable: %s, region: %s", variable, region)

            dataset = _retrieve_dataset(
                variable=variable,
                catalog=catalog, model=model, exp=exp, source=source,
                realization_dict=realization_dict,
                region=region,
                startdate=startdate_data,
                enddate=enddate_data,
                logger=logger,
            )
            if dataset is None:
                logger.warning(
                    "Skipping timeseries for variable '%s', region '%s'.",
                    variable, region,
                )
                continue

            ts = EnsembleTimeseries(
                var=variable,
                monthly_data=dataset,
                catalog_list=catalog,
                model_list=model,
                exp_list=exp,
                source_list=source,
                outputdir=output_opts["outputdir"],
                loglevel=loglevel,
            )
            ts.run()

            has_data = any(
                getattr(ts, attr, None) is not None
                for attr in (
                    "monthly_data", "monthly_data_mean", "monthly_data_std",
                    "annual_data",  "annual_data_mean",  "annual_data_std",
                )
            )
            if not has_data:
                logger.warning(
                    "No timeseries output for variable '%s'. Skipping plot.",
                    variable,
                )
                continue

            ts_plot = PlotEnsembleTimeseries(
                catalog_list=catalog,
                model_list=model,
                exp_list=exp,
                source_list=source,
                outputdir=output_opts["outputdir"],
                loglevel=loglevel,
            )

            # Derive time bounds; prefer monthly, fall back to annual
            _time_src      = ts.monthly_data if ts.monthly_data is not None else ts.annual_data
            startdate_plot = _time_src.time.isel(time=0).values
            enddate_plot   = _time_src.time.isel(time=-1).values

            ts_plot.plot(
                var=variable,
                monthly_data=ts.monthly_data,
                monthly_data_mean=ts.monthly_data_mean,
                monthly_data_std=ts.monthly_data_std,
                save_format=output_opts["save_format"],
                plot_ensemble_members=plot_ensemble_members,
                title=title,
                startdate=startdate_plot,
                enddate=enddate_plot,
            )

            logger.info(
                "Timeseries diagnostic finished for variable '%s'.", variable
            )


def run_latlon(config_dict, args, loglevel, logger):
    """Execute the EnsembleLatLon diagnostic and plot loop.

    Reads all parameters directly from *config_dict*, which is the already-
    loaded and CLI-merged latlon configuration dictionary.

    Args:
        config_dict (dict): merged latlon configuration dictionary.
        args (argparse.Namespace): parsed CLI arguments (used for output dir
            override only; dataset fields are already merged into config_dict).
        loglevel (str): log level string passed to class constructors.
        logger: configured logger instance.
    """
    diag_config     = config_dict["diagnostics"]["ensemble"]
    output_opts     = _output_options(config_dict)
    all_plot_params = diag_config.get("plot_params", {})
    default_plot    = all_plot_params.get("default", {})

    variables = diag_config.get("variable") or []

    catalog, model, exp, source, _regrid = _resolve_dataset(
        args, config_dict, logger
    )
    realization      = extract_realizations(catalog=catalog, model=model,
                                            exp=exp,     source=source)
    realization_dict = {model: realization}

    for variable in variables:
        logger.info("LatLon — variable: %s", variable)

        dataset = _retrieve_dataset(
            variable=variable,
            catalog=catalog, model=model, exp=exp, source=source,
            realization_dict=realization_dict,
            logger=logger,
        )
        if dataset is None:
            logger.warning("Skipping latlon for variable '%s'.", variable)
            continue

        ens_latlon = EnsembleLatLon(
            var=variable,
            dataset=dataset,
            catalog_list=catalog,
            model_list=model,
            exp_list=exp,
            source_list=source,
            outputdir=output_opts["outputdir"],
            loglevel=loglevel,
        )
        ens_latlon.run()

        # Merge default and per-variable plot parameters
        plot_params = {**default_plot, **all_plot_params.get(variable, {})}
        param_dict  = diag_config.get("params", {}).get(variable, {}) or {}

        ens_latlon_plot = PlotEnsembleLatLon(
            catalog_list=catalog,
            model_list=model,
            exp_list=exp,
            source_list=source,
            outputdir=output_opts["outputdir"],
            loglevel=loglevel,
        )

        ens_latlon_plot.plot(
            var=variable,
            dataset_mean=ens_latlon.dataset_mean,
            dataset_std=ens_latlon.dataset_std,
            save_format=output_opts["save_format"],
            dpi=output_opts["dpi"],
            proj=plot_params.get("projection", "robinson"),
            proj_params=plot_params.get("projection_params", {}),
            vmin_mean=plot_params.get("vmin"),
            vmax_mean=plot_params.get("vmax"),
            vmin_std=plot_params.get("vmin_std"),
            vmax_std=plot_params.get("vmax_std"),
            units=param_dict.get("units"),
            long_name=param_dict.get("long_name"),
            transform_first=False,
            cyclic_lon=True,
            contour=True,
            coastlines=True,
            cbar_label=None,
        )

        logger.info("LatLon diagnostic finished for variable '%s'.", variable)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    loglevel = get_arg(args, "loglevel", "WARNING")
    logger   = log_configure(loglevel, "CLI Ensemble")
    logger.info("Starting unified Ensemble diagnostic.")

    cluster  = get_arg(args, "cluster",  None)
    nworkers = get_arg(args, "nworkers", None)
    client, cluster, private_cluster = open_cluster(
        nworkers=nworkers, cluster=cluster, loglevel=loglevel
    )

    # ------------------------------------------------------------------
    # Load each config independently, then apply CLI overrides to both
    # ------------------------------------------------------------------
    ts_config = load_diagnostic_config(
        diagnostic="ensemble",
        config=args.config_timeseries,
        default_config=_DEFAULT_CONFIG_TIMESERIES,
        loglevel=loglevel,
    )
    ts_config = merge_config_args(config=ts_config, args=args, loglevel=loglevel)

    ll_config = load_diagnostic_config(
        diagnostic="ensemble",
        config=args.config_latlon,
        default_config=_DEFAULT_CONFIG_LATLON,
        loglevel=loglevel,
    )
    ll_config = merge_config_args(config=ll_config, args=args, loglevel=loglevel)

    # ------------------------------------------------------------------
    # Dispatch — honour the `run` flag in each config independently
    # ------------------------------------------------------------------
    ran_something = False

    if ts_config.get("diagnostics", {}).get("ensemble", {}).get("run", False):
        logger.info("EnsembleTimeseries is enabled.")
        run_timeseries(config_dict=ts_config, args=args,
                       loglevel=loglevel, logger=logger)
        ran_something = True
    else:
        logger.info("EnsembleTimeseries is disabled (run: false).")

    if ll_config.get("diagnostics", {}).get("ensemble", {}).get("run", False):
        logger.info("EnsembleLatLon is enabled.")
        run_latlon(config_dict=ll_config, args=args,
                   loglevel=loglevel, logger=logger)
        ran_something = True
    else:
        logger.info("EnsembleLatLon is disabled (run: false).")

    if not ran_something:
        logger.warning(
            "No diagnostics ran. Set 'run: true' under diagnostics.ensemble "
            "in at least one of the config files."
        )

    close_cluster(
        client=client, cluster=cluster,
        private_cluster=private_cluster, loglevel=loglevel,
    )
