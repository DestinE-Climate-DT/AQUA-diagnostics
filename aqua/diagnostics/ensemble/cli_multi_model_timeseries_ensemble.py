#!/usr/bin/env python3
"""
Command-line interface for ensemble global time series diagnostic.

This CLI allows to plot ensemle of global timeseries of a variable
defined in a yaml configuration file for multiple models.
"""

import argparse
import sys

import xarray as xr

from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleTimeseries,
    PlotEnsembleTimeseries,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    DiagnosticCLI,
    template_parse_arguments,
)
from aqua.diagnostics.ensemble import (
    extract_realizations_list,
    generate_realizations_path,
)

DEFAULT_CONFIG = "config_multi_model_timeseries_ensemble.yaml"


def parse_arguments(args):
    """Parse command-line arguments for the EnsembleTimeseries for single mulit-model diagnostic CLI.

    Args:
        args (list): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Runs EnsembleTimeseries diagnostic, "),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    return parser.parse_args(args)


def main(argv=None):
    """
    Main function for running EnsembleTimeseries class

    Args:
        argv(list, optional): command-line arguments. Defaults to sys.argv[1:].
    """

    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(args, diagnostic_name="EnsembleTimeseries", default_config=DEFAULT_CONFIG)

    # Preparing Dask cluster
    cli.prepare()
    cli.open_dask_cluster()

    # Output parameters
    outputdir = cli.config_dict.get("output", {}).get("outputdir", "./")
    cli.config_dict.get("output", {}).get("rebuild", True)
    cli.config_dict.get("output", {}).get("save_netcdf", True)
    save_format = cli.config_dict.get("output", {}).get("save_format", SAVE_FORMAT)
    cli.config_dict.get("output", {}).get("dpi", 300)

    diag_config = cli.config_dict["diagnostics"]["ensemble"]

    params = diag_config.get("params", {}).get("default", {})

    # Note in old catalogs freq is 'mon'/'ann'
    monthly = params.get("mon", params.get("monthly"))
    annual = params.get("annual", params.get("monthly"))
    monthly_freq = "mon"
    annual_freq = "ann"
    plot_ensemble_members = params.get("plot_ensemble_members", True)

    startdate = params.get("startdate")
    enddate = params.get("enddate")
    variables = diag_config.get("variables") or []

    # Single reference
    if "references" in cli.config_dict:
        ref = cli.config_dict.get("references")
        first_ref = ref[0]
        catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
        model_ref = get_arg(args, "model", first_ref["model"])
        exp_ref = get_arg(args, "exp", first_ref["exp"])
        source_ref = get_arg(args, "source", first_ref["source"])
        fixer_ref = get_arg(args, "fix", first_ref.get("fix"))

        cli.logger.debug(f"Reference catalog: {catalog_ref}, model: {model_ref}, exp: {exp_ref} and source: {source_ref}")

    # EnsembleTimeseries diagnostic
    if cli.config_dict["diagnostics"]["ensemble"]["run"]:
        # Variables in Timeseries config
        for variable in variables:
            var_params = diag_config.get("params", {}).get(variable, {})
            regions = var_params.get("regions") or []
            for region in regions:
                cli.logger.info("Ensemble Timeseries for variable: %s, region: %s", variable, region)

                # Dictionary to contain all the inputs for the Ensemble Timeseries class

                # Dictionary to contain reference the outputs for the Ensemble Timeseries class
                # needed for the plot class
                var_params.get("title", None)

                # Model data
                # TODO: hourly and daily data
                models = cli.config_dict["datasets"]

                catalog_list = []
                model_list = []
                exp_list = []
                source_list = []
                realization_list = []
                if models is not None:
                    models[0]["catalog"] = get_arg(args, "catalog", models[0]["catalog"])
                    models[0]["model"] = get_arg(args, "model", models[0]["model"])
                    models[0]["exp"] = get_arg(args, "exp", models[0]["exp"])
                    models[0]["source"] = get_arg(args, "source", models[0]["source"])
                    models[0]["realization"] = get_arg(args, "realization", models[0]["realization"])
                    # models[0]["fix"] = get_arg(args, "fix", models[0]["fix"])

                    for model in models:
                        catalog_list.append(model["catalog"])
                        model_list.append(model["model"])
                        exp_list.append(model["exp"])
                        source_list.append(model["source"])
                        if model["realization"] is None:
                            realization_list.append(model["realization"])
                        else:
                            realization = extract_realizations_list(
                                catalog=model["catalog"], model=model["model"], exp=exp["exp"], source=source["source"]
                            )
                            realization_list.append(realization)

                # Reterive monthly data
                if monthly:
                    monthly_dataset = reader_retrieve_and_merge(
                        variable=variable,
                        catalog_list=catalog_list,
                        model_list=model_list,
                        exp_list=exp_list,
                        source_list=source_list,
                        realization=realization_list,
                        freq=monthly_freq,
                        startdate=startdate,
                        enddate=enddate,
                    )

                    if monthly_dataset is None:
                        cli.logger.warning("Monthly ensemble data is not provided.")

                # Reterieve annual data
                if annual:
                    annual_dataset = reader_retrieve_and_merge(
                        variable=variable,
                        catalog_list=catalog_list,
                        model_list=model_list,
                        exp_list=exp_list,
                        source_list=source_list,
                        realization=realization_list,
                        freq=annual_freq,
                        startdate=startdate,
                        enddate=enddate,
                    )
                    if annual_dataset is None:
                        cli.logger.warning("Annual ensemble data is not provided.")

                # Instantiate the ensemble timeseries class
                ts = EnsembleTimeseries(
                    var=variable,
                    monthly_data=monthly_dataset,
                    annual_data=annual_dataset,
                    catalog_list=catalog_list,
                    model_list=model_list,
                    exp_list=exp_list,
                    source_list=source_list,
                    outputdir=outputdir,
                    loglevel=cli.loglevel,
                )
                # Compute statistics and save the results as netcdf
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
                    cli.logger.warning(
                        "No timeseries output for variable '%s'. Skipping plot.",
                        variable,
                    )
                    continue

                # Monthly reference timeseries
                extra_dict = {"variable": variable, "freq": "monthly", "region": region}
                ref_realization_list = extract_realizations_list(
                    catalog=catalog_ref, model=model_ref, exp=exp_ref, source=source_ref
                )
                mon_ref_filenames = generate_realizations_path(
                    catalog=catalog_ref,
                    model=model_ref,
                    exp=exp_ref,
                    realization_list=ref_realization_list,
                    diagnostic_name="timeseries",
                    diagnostic_product="timeseries",
                    variable=variable,
                    file_dir=outputdir,
                    extra_keys=extra_dict,
                    file_format=".nc",
                    loglevel=cli.loglevel,
                )

                # Loading reference monthly timeseries
                if mon_ref_filenames:
                    dataset_mon_ref = reader_retrieve_and_merge(
                        filenames=mon_ref_filenames,
                        variable=variable,
                        # catalog=catalog_ref,
                        # model=model_ref,
                        # exp=exp_ref,
                        # source=source_ref,
                        region=region,
                        realization=ref_realization_list,
                        startdate=startdate,
                        enddate=enddate,
                        fix=fixer_ref,
                        loglevel=cli.loglevel,
                    )
                    if dataset_mon_ref is None:
                        cli.logger.warning(
                            "Skipping monthly reference timeseries for variable '%s', region '%s'.",
                            variable,
                            region,
                        )
                        continue

                    if dataset_mon_ref:
                        if isinstance(dataset_mon_ref, xr.Dataset):
                            dataset_mon_ref = dataset_mon_ref[variable]

                # Annual reference timeseries
                extra_dict = {"variable": variable, "freq": "annual", "region": region}
                ann_ref_filenames = generate_realizations_path(
                    catalog=catalog_ref,
                    model=model_ref,
                    exp=exp_ref,
                    realization_list=ref_realization_list,
                    diagnostic_name="timeseries",
                    diagnostic_product="timeseries",
                    variable=variable,
                    file_dir=outputdir,
                    extra_keys=extra_dict,
                    file_format=".nc",
                    loglevel=cli.loglevel,
                )

                # Loading reference annual timeseries
                if ann_ref_filenames:
                    dataset_ann_ref = reader_retrieve_and_merge(
                        filenames=ann_ref_filenames,
                        variable=variable,
                        # catalog=catalog_ref,
                        # model=model_ref,
                        # exp=exp_ref,
                        # source=source_ref,
                        region=region,
                        realization=ref_realization_list,
                        startdate=startdate,
                        enddate=enddate,
                        fix=fixer_ref,
                        loglevel=cli.loglevel,
                    )
                    if dataset_ann_ref is None:
                        cli.logger.warning(
                            "Skipping annual reference timeseries for variable '%s', region '%s'.",
                            variable,
                            region,
                        )
                        continue

                    if dataset_ann_ref:
                        if isinstance(dataset_ann_ref, xr.Dataset):
                            dataset_ann_ref = dataset_ann_ref[variable]

                # Instiatating ensemble timeseries plotting class
                ts_plot = PlotEnsembleTimeseries(
                    catalog_list=catalog_list,
                    model_list=model_list,
                    exp_list=exp_list,
                    source_list=source_list,
                    ref_catalog=catalog_ref,
                    ref_model=model_ref,
                    ref_exp=exp_ref,
                    outputdir=outputdir,
                    loglevel=cli.loglevel,
                )

                # Derive time bounds; prefer monthly, fall back to annual
                _time_src = ts.monthly_data if ts.monthly_data is not None else ts.annual_data
                if startdate is None:
                    startdate = _time_src.time.isel(time=0).values
                if enddate is None:
                    enddate = _time_src.time.isel(time=-1).values

                # Ensemble Timeseries plotting function
                ts_plot.plot(
                    var=variable,
                    monthly_data=ts.monthly_data.squeeze(),
                    monthly_data_mean=ts.monthly_data_mean.squeeze(),
                    monthly_data_std=ts.monthly_data_std.squeeze(),
                    annual_data=ts.annual_data.squeeze(),
                    annual_data_mean=ts.annual_data_mean.squeeze(),
                    annual_data_std=ts.annual_data_std.squeeze(),
                    ref_monthly_data=dataset_mon_ref.squeeze(),
                    ref_annual_data=dataset_ann_ref.squeeze(),
                    save_format=save_format,
                    plot_ensemble_members=plot_ensemble_members,
                    startdate=startdate,
                    enddate=enddate,
                )

                cli.logger.info("Timeseries diagnostic finished for variable '%s'.", variable)

    cli.logger.info("Completed Ensemble time series diagnostic!")


if __name__ == "__main__":
    main()
