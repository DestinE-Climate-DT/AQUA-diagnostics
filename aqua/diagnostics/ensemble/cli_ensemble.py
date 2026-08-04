#!/usr/bin/env python3
"""
Command-line interface for ensemble diagnostics.

Runs EnsembleTimeseries and/or EnsembleLatLon diagnostics for a single model.
Each diagnostic reads its own original YAML configuration file:

Which diagnostics run is controlled by the ``run`` flag inside each config's
``diagnostics.ensemble`` block, exactly as before.  Both configs are loaded
every time; set ``run: false`` in either one to skip it.

CLI overrides (``--catalog``, ``--model``, ``--exp``, ``--source``,
``--outputdir``, ``--loglevel``) are applied to both configs.
"""

import argparse
import sys
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleLatLon,
    EnsembleTimeseries,
    PlotEnsembleLatLon,
    PlotEnsembleTimeseries,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    close_cluster,
    DiagnosticCLI,
    load_var_config,
    merge_config_args,
    open_cluster,
    template_parse_arguments,
)

from aqua.diagnostics.ensemble import (
    generate_realizations_path,
    extract_realizations_list,
)    

# Default config filenames (resolved by load_diagnostic_config from the
# package's config/collections/legacy/atmosphere2d/ directory)
DEFAULT_CONFIG = "config-atmosphere2d-berkeley-ensemble.yaml"

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
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    return parser.parse_args(args)

def _output_options(config_dict):
    """Extract output options from a configuration dictionary.

    Args:
        config_dict (dict): full configuration dictionary.
    Returns:
        dict: keys outputdir, rebuild, save_netcdf, save_format, dpi.
    """
    out = config_dict.get("output", {})
    return {
        "outputdir": out.get("outputdir", "./"),
        "rebuild": out.get("rebuild", True),
        "save_netcdf": out.get("save_netcdf", True),
        "save_format": out.get("save_format", SAVE_FORMAT),
        "dpi": out.get("dpi", 300),
    }

def _retrieve_dataset(
    filenames, variable, catalog, model, exp, source, realization_dict, region=None, startdate=None, enddate=None, loglevel=None
):
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
        loglevel: configured log instance. Pass via config_dict loglevel

    Returns:
        xarray.Dataset | None: merged dataset, or None if retrieval failed.
    """
    dataset = reader_retrieve_and_merge(
        filenames=filenames,
        variable=variable,
        catalog_list=catalog,
        model_list=model,
        exp_list=exp,
        source_list=source,
        region=region,
        startdate=startdate,
        enddate=enddate,
        realization=realization_dict,
        loglevel=loglevel,
    )
    if dataset is None and loglevel is not None:
        loglevel.warning("Ensemble data retrieval returned None for variable '%s'.", variable)
    return dataset

def run_timeseries(config_dict, args, loglevel):
    """Execute the EnsembleTimeseries diagnostic and plot loop.

    Reads all parameters directly from *config_dict*, which is the already-
    loaded and CLI-merged timeseries configuration dictionary.

    Args:
        config_dict (dict): merged timeseries configuration dictionary.
        args (argparse.Namespace): parsed CLI arguments (used for output dir
            override only; dataset fields are already merged into config_dict).
        loglevel: configured log instance. Pass via config_dict loglevel
    """
    diag_config = config_dict["diagnostics"]["timeseries"]
    output_opts = _output_options(config_dict)

    params = diag_config.get("params", {}).get("default", {})

    monthly = params.get("monthly")
    annual = params.get("annual")

    startdate_data = params.get("startdate")
    enddate_data = params.get("enddate")

    variables = diag_config.get("variables") or []

    datasets = config_dict.get("datasets")
    first = datasets[0]
    catalog = get_arg(args, "catalog", first["catalog"])
    model = get_arg(args, "model", first["model"])
    exp = get_arg(args, "exp", first["exp"])
    source = get_arg(args, "source", first["source"])
    regrid = get_arg(args, "regrid", first.get("regrid"))
    if startdate_data is None: startdate_data = get_arg(args, "startdate", first.get("startdate") or None)
    if enddate_data is None: enddate_data = get_arg(args, "enddate", first.get("enddate") or None) 

    if "references" in config_dict:
        ref = config_dict.get("references")
        first_ref = ref[0]
        catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
        model_ref = get_arg(args, "model", first_ref["model"])
        exp_ref = get_arg(args, "exp", first_ref["exp"])
        source_ref = get_arg(args, "source", first_ref["source"])
        startdate_ref = get_arg(args, "startdate", first_ref["startdate"])
        enddate_ref = get_arg(args, "enddate", first_ref["enddate"])

    # This is not defined in the unified config file
    plot_ensemble_members=True

    for variable in variables:

        var_params = diag_config.get("params", {}).get(variable, {})
        regions = var_params.get("regions") or []
        for region in regions:
            #logger.info("Timeseries — variable: %s, region: %s", variable, region)
            
            # Dictionary to contain all the inputs for the Ensemble Timeseries class
            timeseries_dict = {}
           
            # Dictionary to contain reference the outputs for the Ensemble Timeseries class
            # needed for the plot class 
            timeseries_ref_plot_dict = {}
 
            if monthly:
                extra_dict = {"variable": variable, "freq": "monthly", "region":region}
                mon_realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                mon_filenames = generate_realizations_path(catalog=catalog, model=model, exp=exp, realization_list=mon_realization_list, diagnostic_name="timeseries", diagnostic_product="timeseries", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)
                
                #loglevel.info(f"######## {mon_filenames} #############")
                dataset_mon = _retrieve_dataset(
                    filenames=mon_filenames,
                    variable=variable,
                    catalog=catalog,
                    model=model,
                    exp=exp,
                    source=source,
                    realization_dict=mon_realization_list,
                    region=region,
                    startdate=startdate_data,
                    enddate=enddate_data,
                    loglevel=loglevel,
                )
                if dataset_mon is None:
                    #loglevel.warning(
                    #    "Skipping monthly timeseries for variable '%s', region '%s'.",
                    #    variable,
                    #    region,
                    #)
                    continue
                
                if dataset_mon:
                    timeseries_dict["monthly_data"] = dataset_mon
 
            if annual:
                extra_dict = {"variable": variable, "freq":"annual", "region":region}
                ann_realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                ann_filenames = generate_realizations_path(catalog=catalog, model=model, exp=exp, realization_list=ann_realization_list, diagnostic_name="timeseries", diagnostic_product="timeseries", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)


                dataset_ann = _retrieve_dataset(
                    filenames=ann_filenames,
                    variable=variable,
                    catalog=catalog,
                    model=model,
                    exp=exp,
                    source=source,
                    realization_dict=ann_realization_list,
                    region=region,
                    startdate=startdate_data,
                    enddate=enddate_data,
                    loglevel=loglevel,
                )
                if dataset_ann is None:
                    #loglevel.warning(
                    #    "Skipping monthly timeseries for variable '%s', region '%s'.",
                    #    variable,
                    #    region,
                    #)
                    continue

                if dataset_ann:
                    timeseries_dict["annual_data"] = dataset_ann

            ts = EnsembleTimeseries(
                **timeseries_dict,
                var=variable,
                #monthly_data=dataset_mon,
                #annual_data=dataset_ann,
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
                    "monthly_data",
                    "monthly_data_mean",
                    "monthly_data_std",
                    "annual_data",
                    "annual_data_mean",
                    "annual_data_std",
                )
            )
            if not has_data:
                loglevel.warning(
                    "No timeseries output for variable '%s'. Skipping plot.",
                    variable,
                )
                continue

            # Monthly reference
            extra_dict = {"variable": variable, "freq": "monthly", "region":region}
            ref_realization_list = extract_realizations_list(catalog=catalog_ref, model=model_ref, exp=exp_ref, source=source_ref)
            mon_ref_filenames = generate_realizations_path(catalog=catalog_ref, model=model_ref, exp=exp_ref, realization_list=ref_realization_list, diagnostic_name="timeseries", diagnostic_product="timeseries", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)
            if mon_ref_filenames:
                dataset_mon_ref = _retrieve_dataset(
                    filenames=mon_ref_filenames,
                    variable=variable,
                    catalog=catalog_ref,
                    model=model_ref,
                    exp=exp_ref,
                    source=source_ref,
                    region=region,
                    realization_dict=ref_realization_list,
                    startdate=startdate_ref,
                    enddate=enddate_ref,
                    loglevel=loglevel,
                )
                if dataset_mon_ref is None:
                    #loglevel.warning(
                    #    "Skipping monthly timeseries for variable '%s', region '%s'.",
                    #    variable,
                    #    region,
                    #)
                    continue

                if dataset_mon_ref:
                    if isinstance(dataset_mon_ref, xr.Dataset):
                        dataset_mon_ref = dataset_mon_ref[variable]

            # Annual reference
            extra_dict = {"variable": variable, "freq": "annual", "region":region}
            ann_ref_filenames = generate_realizations_path(catalog=catalog_ref, model=model_ref, exp=exp_ref, realization_list=ref_realization_list, diagnostic_name="timeseries", diagnostic_product="timeseries", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)

            if ann_ref_filenames:
                dataset_ann_ref = _retrieve_dataset(
                    filenames=ann_ref_filenames,
                    variable=variable,
                    catalog=catalog_ref,
                    model=model_ref,
                    exp=exp_ref,
                    source=source_ref,
                    region=region,
                    realization_dict=ref_realization_list,
                    startdate=startdate_ref,
                    enddate=enddate_ref,
                    loglevel=loglevel,
                )
                if dataset_ann_ref is None:
                    #loglevel.warning(
                    #    "Skipping monthly timeseries for variable '%s', region '%s'.",
                    #    variable,
                    #    region,
                    #)
                    continue

                if dataset_ann_ref:
                    if isinstance(dataset_ann_ref, xr.Dataset):
                        dataset_ann_ref = dataset_ann_ref[variable]

            ts_plot = PlotEnsembleTimeseries(
                catalog_list=catalog,
                model_list=model,
                exp_list=exp,
                source_list=source,
                ref_catalog=catalog_ref,
                ref_model=model_ref,
                ref_exp=exp_ref,
                outputdir=output_opts["outputdir"],
                loglevel=loglevel,
            )

            # Derive time bounds; prefer monthly, fall back to annual
            _time_src = ts.monthly_data if ts.monthly_data is not None else ts.annual_data
            startdate_plot = _time_src.time.isel(time=0).values
            enddate_plot = _time_src.time.isel(time=-1).values

            ts_plot.plot(
                var=variable,
                monthly_data=ts.monthly_data,
                monthly_data_mean=ts.monthly_data_mean,
                monthly_data_std=ts.monthly_data_std,
                annual_data=ts.annual_data,
                annual_data_mean=ts.annual_data_mean,
                annual_data_std=ts.annual_data_std,
                ref_monthly_data=dataset_mon_ref if dataset_mon_ref is not None else None,
                ref_annual_data=dataset_ann_ref if dataset_ann_ref is not None else None,
                save_format=output_opts["save_format"],
                plot_ensemble_members=plot_ensemble_members,
                startdate=startdate_plot,
                enddate=enddate_plot,
            )

            #loglevel.info("Timeseries diagnostic finished for variable '%s'.", variable)


def run_latlon(config_dict, args, loglevel):
    """Execute the EnsembleLatLon diagnostic and plot loop.

    Reads all parameters directly from *config_dict*, which is the already-
    loaded and CLI-merged latlon configuration dictionary.

    Args:
        config_dict (dict): merged latlon configuration dictionary.
        args (argparse.Namespace): parsed CLI arguments (used for output dir
            override only; dataset fields are already merged into config_dict).
        loglevel (str): log level string passed to class constructors.
    """

    diag_config = config_dict["diagnostics"]["globalbiases"]
    output_opts = _output_options(config_dict)

    params = diag_config.get("params", {}).get("default", {})

    variables = diag_config.get("variables") or []

    all_plot_params = diag_config.get("plot_params", {}) 

    default_plot = all_plot_params.get("default", {})

    #catalog, model, exp, source, _regrid = _resolve_dataset(args, config_dict, loglevel)
    datasets = config_dict.get("datasets")
    first = datasets[0]
    catalog = get_arg(args, "catalog", first["catalog"])
    model = get_arg(args, "model", first["model"])
    exp = get_arg(args, "exp", first["exp"])
    source = get_arg(args, "source", first["source"])
    regrid = get_arg(args, "regrid", first.get("regrid"))

    if "references" in config_dict:
        ref = config_dict.get("references")
        first_ref = ref[0]
        catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
        model_ref = get_arg(args, "model", first_ref["model"])
        exp_ref = get_arg(args, "exp", first_ref["exp"])
        source_ref = get_arg(args, "source", first_ref["source"])

    # Reference
    ref_realization_list = extract_realizations_list(catalog=catalog_ref, model=model_ref, exp=exp_ref, source=source_ref)
    ref_filenames = generate_realizations_path(catalog=catalog_ref, model=model_ref, exp=exp_ref, realization_list=ref_realization_list, diagnostic_name="globalbiases", diagnostic_product="annual_climatology", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)
    if ref_filenames:
        dataset_ref = _retrieve_dataset(
            filenames=ref_filenames,
            variable=variable,
            catalog=catalog_ref,
            model=model_ref,
            exp=exp_ref,
            source=source_ref,
            region=region,
            realization_dict=ref_realization_list,
            startdate=startdate_ref,
            enddate=enddate_ref,
            loglevel=loglevel,
        )
        #if dataset_ref is None:
        #    #loglevel.warning(
        #    #    "Skipping monthly timeseries for variable '%s', region '%s'.",
        #    #    variable,
        #    #    region,
        #    #)
        #    continue

        if dataset_ref:
            if isinstance(dataset_ref, xr.Dataset):
                dataset_ref = dataset_ref[variable]
            annual_climatology_ref_plot_dict["ref_data"] = dataset_ref



    for variable in variables:
        #logger.info("LatLon — variable: %s", variable)

        var_params = diag_config.get("params", {}).get(variable, {})
        regions = var_params.get("regions") or []

        if regions:
            for region in regions:

                realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                extra_dict = {"variable": variable, "region":region}

                filenames = generate_realizations_path(catalog=catalog, model=model, exp=exp, realization_list=realization_list, diagnostic_name="globalbiases", diagnostic_product="annual_climatology", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)

                dataset = _retrieve_dataset(
                    filenames=filenames,
                    variable=variable,
                    catalog=catalog,
                    model=model,
                    exp=exp,
                    source=source,
                    realization_dict=realization_list,
                    loglevel=loglevel,
                )
                if dataset is None:
                    loglevel.warning("Skipping latlon for variable '%s'.", variable)
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
                var_plot = all_plot_params.get(variable, {})
                plot_params = {**default_plot, **var_plot}
                param_dict = diag_config.get("params", {}).get(variable, {}) or {}

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

                loglevel.info("LatLon diagnostic finished for variable '%s'.", variable)

        else:

            realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
            extra_dict = {"variable": variable}

            filenames = generate_realizations_path(catalog=catalog, model=model, exp=exp, realization_list=realization_list, diagnostic_name="globalbiases", diagnostic_product="annual_climatology", variable=variable, outputdir=output_opts["outputdir"], extra_keys=extra_dict, file_format=".nc", loglevel=loglevel)

            dataset = _retrieve_dataset(
                filenames=filenames,
                variable=variable,
                catalog=catalog,
                model=model,
                exp=exp,
                source=source,
                realization_dict=realization_list,
                loglevel=loglevel,
            )
            if dataset is None:
                loglevel.warning("Skipping latlon for variable '%s'.", variable)
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
            var_plot = all_plot_params.get(variable, {})
            plot_params = {**default_plot, **var_plot}
            param_dict = diag_config.get("params", {}).get(variable, {}) or {}

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

            #loglevel.info("LatLon diagnostic finished for variable '%s'.", variable)

def main(argv=None):
    """
    Main function for running multiple Ensemble classes

    Args:
        argv(list, optional): command-line arguments. Defaults to sys.argv[1:].
    """

    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(args, diagnostic_name='atmosphere2d', default_config='config-atmosphere2d-berkeley-ensemble.yaml')

    cli.prepare()
    cli.open_dask_cluster()
    
    # Ensemble Timeseries diagnostic
    run_timeseries(config_dict=cli.config_dict, args=args, loglevel=cli.loglevel)

    # Ensemble 2D maps diagnostic
    #run_latlon(config_dict=cli.config_dict, args=args, loglevel=cli.loglevel)

    cli.close_dask_cluster()

if __name__ == "__main__":
    main()
