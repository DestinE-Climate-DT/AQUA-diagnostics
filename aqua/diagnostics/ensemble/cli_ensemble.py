#!/usr/bin/env python3
"""
Command-line interface for ensemble diagnostics.

Runs EnsembleTimeseries and/or EnsembleMaps diagnostics for a single model.
Each diagnostic reads its own original YAML configuration file:

Which diagnostics run is controlled by the ``run`` flag inside each config's
``diagnostics.ensemble`` block, exactly as before.  Both configs are loaded
every time; set ``run: false`` in either one to skip it.

If no reference is configured, or the reference files cannot be found or
loaded, the reference is skipped and only the ensemble products are plotted.

CLI overrides (``--catalog``, ``--model``, ``--exp``, ``--source``,
``--outputdir``, ``--loglevel``) are applied to both configs.
"""

import argparse
import os
import sys

import xarray as xr

from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleMaps,
    EnsembleTimeseries,
    PlotEnsembleMaps,
    PlotEnsembleTimeseries,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    DiagnosticCLI,
    TitleBuilder,
    template_parse_arguments,
)
from aqua.diagnostics.ensemble import extract_realizations_list, generate_realizations_path_single_model

# Default config filenames (resolved by load_diagnostic_config from the
# package's config/collections/legacy/atmosphere2d/ directory)
DEFAULT_CONFIG = "config-atmosphere2d-berkeley-ensemble.yaml"
DEFAULT_DIAGNOSTIC_NAME = "atmosphere2d"


def parse_arguments(args):
    """Parse command-line arguments for the unified ensemble diagnostic CLI.

    Args:
        args (list): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Unified Ensemble diagnostic CLI. Runs EnsembleTimeseries and EnsembleMaps back-to-back, "),
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    return parser.parse_args(args)


def _squeeze(data):
    """Squeeze a DataArray/Dataset, passing None through unchanged."""
    return data.squeeze() if data is not None else None


def load_reference(cli, filenames, variable, region, fix, label, select_variable=True):
    """
    Load a reference dataset, returning None if it is not available.

    Args:
        cli (DiagnosticCLI): CLI object, used for logging and loglevel.
        filenames (list[str]): Candidate reference file paths.
        variable (str): Variable to load.
        region (str): Region to select (may be None).
        fix (bool): Fixer flag for the reference.
        label (str): Short description used in log messages.
        select_variable (bool): If True, return the variable's DataArray instead of the Dataset.

    Returns:
        xarray.DataArray | xarray.Dataset | None: The reference data, or None if unavailable.
    """
    existing = [f for f in (filenames or []) if os.path.exists(f)]
    if not existing:
        cli.logger.warning(
            "Reference %s not available for variable '%s', region '%s' (no file found among %s). Skipping reference.",
            label,
            variable,
            region,
            filenames,
        )
        return None

    try:
        data = reader_retrieve_and_merge(
            filenames=existing,
            variable=variable,
            region=region,
            realizations="1",  # TODO
            fix=fix,
            loglevel=cli.loglevel,
        )
    except Exception as e:
        cli.logger.warning(
            "Failed to load reference %s for variable '%s', region '%s': %s. Skipping reference.",
            label,
            variable,
            region,
            e,
        )
        return None

    if data is None:
        cli.logger.warning(
            "Reference %s could not be loaded for variable '%s', region '%s'. Skipping reference.",
            label,
            variable,
            region,
        )
        return None

    if select_variable and isinstance(data, xr.Dataset) and variable in data:
        data = data[variable]
    if "ensemble" in data.dims:
        data = data.squeeze("ensemble", drop=True)

    return data


def main(argv=None):
    """
    Main function for running multiple Ensemble classes

    Args:
        argv(list, optional): command-line arguments. Defaults to sys.argv[1:].
    """

    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(args, diagnostic_name=DEFAULT_DIAGNOSTIC_NAME, default_config=DEFAULT_CONFIG)

    # Preparing Dask cluster
    cli.prepare()
    cli.open_dask_cluster()

    # Dataset for single model
    datasets = cli.config_dict.get("datasets")
    first = datasets[0]
    catalog = get_arg(args, "catalog", first["catalog"])
    model = get_arg(args, "model", first["model"])
    exp = get_arg(args, "exp", first["exp"])
    source = get_arg(args, "source", first["source"])
    get_arg(args, "regrid", first.get("regrid"))
    fixer = get_arg(args, "fix", first.get("fix"))

    cli.logger.debug(f"Ensemble single model catalog: {catalog}, model: {model}, exp: {exp}, and source {source}")

    # Single reference (optional)
    catalog_ref = model_ref = exp_ref = source_ref = fixer_ref = None
    references = cli.config_dict.get("references") or []
    if references:
        first_ref = references[0]
        catalog_ref = first_ref.get("catalog")
        model_ref = first_ref.get("model")
        exp_ref = first_ref.get("exp")
        source_ref = first_ref.get("source")
        fixer_ref = first_ref.get("fix")

    has_ref = all(x is not None for x in (catalog_ref, model_ref, exp_ref, source_ref))
    if has_ref:
        cli.logger.debug(f"Reference catalog: {catalog_ref}, model: {model_ref}, exp: {exp_ref} and source: {source_ref}")
    else:
        cli.logger.warning("No complete reference configured. Reference data will be skipped.")

    # Output parameters
    # outputdir = cli.outputdir #config_dict.get("output", {}).get("outputdir", "./")
    # config_dict.get("output", {}).get("rebuild", True)
    # cli.config_dict.get("output", {}).get("save_netcdf", True)
    save_format = cli.config_dict.get("output", {}).get("save_format", SAVE_FORMAT)
    dpi = cli.config_dict.get("output", {}).get("dpi", 300)

    # Time series ensemble
    ts_diag_config = cli.config_dict["diagnostics"]["timeseries"]

    if "timeseries" in cli.config_dict["diagnostics"]:
        if cli.config_dict["diagnostics"]["timeseries"]["run"]:
            cli.logger.info("Running CLI for EnsembleTimeseries diagnostic for a single model")

            params = ts_diag_config.get("params", {}).get("default", {})
            monthly = params.get("monthly")
            annual = params.get("annual")
            plot_ensemble_members = params.get("plot_ensemble_members", True)

            # TODO: discuss the length of the time interval for timeseries plots
            # startdate = params.get("startdate")
            # enddate = params.get("enddate")
            # cli.logger.info(f"Timeseries startdate {startdate} enddate {enddate}")
            # if startdate is None:
            #    startdate = get_arg(args, "startdate", first.get("startdate") or None)
            # if enddate is None:
            #    enddate = get_arg(args, "enddate", first.get("enddate") or None)

            variables = ts_diag_config.get("variables") or []

            # Variables in Timeseries config
            for variable in variables:
                var_params = ts_diag_config.get("params", {}).get(variable, {})
                regions = var_params.get("regions") or []
                for region in regions:
                    cli.logger.info("Ensemble Timeseries for variable: %s, region: %s", variable, region)

                    # Dictionary to contain all the inputs for the Ensemble Timeseries class
                    timeseries_dict = {}
                    mon_realization_list = None
                    ann_realization_list = None

                    # Monthly dataset
                    if monthly:
                        extra_dict = {"variable": variable, "freq": "monthly", "region": region}
                        mon_realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                        cli.logger.info(f"realization list: {mon_realization_list}")
                        mon_filenames = generate_realizations_path_single_model(
                            catalog=catalog,
                            model=model,
                            exp=exp,
                            realization_list=mon_realization_list,
                            diagnostic_name=ts_diag_config.get("diagnostic_name", "timeseries"),
                            diagnostic_product="timeseries",
                            variable=variable,
                            file_dir=cli.outputdir,
                            extra_keys=extra_dict,
                            file_format=".nc",
                            loglevel=cli.loglevel,
                        )

                        cli.logger.debug(f"Files for monthly ensemble timeseries {mon_filenames}")

                        # Loading and merging monthly Timeseries datasets
                        dataset_mon = reader_retrieve_and_merge(
                            filenames=mon_filenames,
                            variable=variable,
                            realizations=mon_realization_list,
                            region=region,
                            fix=fixer,
                            loglevel=cli.loglevel,
                        )
                        if dataset_mon is None:
                            cli.logger.warning(
                                "Skipping monthly timeseries for variable '%s', region '%s'.",
                                variable,
                                region,
                            )
                            continue

                        if dataset_mon:
                            timeseries_dict["monthly_data"] = dataset_mon

                    # Annual dataset
                    if annual:
                        extra_dict = {"variable": variable, "freq": "annual", "region": region}
                        ann_realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                        cli.logger.info(f"realization list: {ann_realization_list}")
                        ann_filenames = generate_realizations_path_single_model(
                            catalog=catalog,
                            model=model,
                            exp=exp,
                            realization_list=ann_realization_list,
                            diagnostic_name=ts_diag_config.get("diagnostic_name", "timeseries"),
                            diagnostic_product="timeseries",
                            variable=variable,
                            file_dir=cli.outputdir,
                            extra_keys=extra_dict,
                            file_format=".nc",
                            loglevel=cli.loglevel,
                        )

                        cli.logger.debug(f"Files for annual ensemble timeseries {ann_filenames}")

                        # Loading and merging annual Timeseries datasets
                        dataset_ann = reader_retrieve_and_merge(
                            filenames=ann_filenames,
                            variable=variable,
                            realizations=ann_realization_list,
                            region=region,
                            fix=fixer,
                            loglevel=cli.loglevel,
                        )
                        if dataset_ann is None:
                            cli.logger.warning(
                                "Skipping annual timeseries for variable '%s', region '%s'.",
                                variable,
                                region,
                            )
                            continue

                        if dataset_ann:
                            timeseries_dict["annual_data"] = dataset_ann

                    # Instantiate the ensemble timeseries class
                    ts = EnsembleTimeseries(
                        **timeseries_dict,
                        var=variable,
                        catalog_list=catalog,
                        model_list=model,
                        exp_list=exp,
                        source_list=source,
                        outputdir=cli.outputdir,
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

                    # Reference timeseries (optional)
                    dataset_mon_ref = None
                    dataset_ann_ref = None

                    if has_ref:
                        ts_realization_list = mon_realization_list or ann_realization_list

                        # Monthly reference timeseries
                        if monthly:
                            extra_dict = {"variable": variable, "freq": "monthly", "region": region}
                            mon_ref_filenames = generate_realizations_path_single_model(
                                catalog=catalog,
                                model=model,
                                exp=exp,
                                catalog_ref=catalog_ref,
                                model_ref=model_ref,
                                exp_ref=exp_ref,
                                realization_list=ts_realization_list,
                                realization_ref="r1",  # TODO
                                diagnostic_name=ts_diag_config.get("diagnostic_name", "timeseries"),
                                diagnostic_product="timeseries",
                                variable=variable,
                                file_dir=cli.outputdir,
                                extra_keys=extra_dict,
                                file_format=".nc",
                                loglevel=cli.loglevel,
                            )
                            cli.logger.info(f"Reference monthly timeseries file: {mon_ref_filenames}")
                            dataset_mon_ref = load_reference(
                                cli, mon_ref_filenames, variable, region, fixer_ref, label="monthly timeseries"
                            )

                        # Annual reference timeseries
                        if annual:
                            extra_dict = {"variable": variable, "freq": "annual", "region": region}
                            ann_ref_filenames = generate_realizations_path_single_model(
                                catalog=catalog,
                                model=model,
                                exp=exp,
                                catalog_ref=catalog_ref,
                                model_ref=model_ref,
                                exp_ref=exp_ref,
                                realization_list=ts_realization_list,
                                realization_ref="r1",  # TODO
                                diagnostic_name=ts_diag_config.get("diagnostic_name", "timeseries"),
                                diagnostic_product="timeseries",
                                variable=variable,
                                file_dir=cli.outputdir,
                                extra_keys=extra_dict,
                                file_format=".nc",
                                loglevel=cli.loglevel,
                            )
                            cli.logger.info(f"Reference annual timeseries file: {ann_ref_filenames}")
                            dataset_ann_ref = load_reference(
                                cli, ann_ref_filenames, variable, region, fixer_ref, label="annual timeseries"
                            )

                    # Instantiating ensemble timeseries plotting class
                    ts_plot = PlotEnsembleTimeseries(
                        catalog_list=catalog,
                        model_list=model,
                        exp_list=exp,
                        source_list=source,
                        ref_catalog=catalog_ref if dataset_mon_ref is not None or dataset_ann_ref is not None else None,
                        ref_model=model_ref if dataset_mon_ref is not None or dataset_ann_ref is not None else None,
                        ref_exp=exp_ref if dataset_mon_ref is not None or dataset_ann_ref is not None else None,
                        outputdir=cli.outputdir,
                        loglevel=cli.loglevel,
                    )

                    # Ensemble Timeseries plotting function
                    ts_plot.plot(
                        var=variable,
                        monthly_data=_squeeze(ts.monthly_data),
                        monthly_data_mean=_squeeze(ts.monthly_data_mean),
                        monthly_data_std=_squeeze(ts.monthly_data_std),
                        annual_data=_squeeze(ts.annual_data),
                        annual_data_mean=_squeeze(ts.annual_data_mean),
                        annual_data_std=_squeeze(ts.annual_data_std),
                        ref_monthly_data=_squeeze(dataset_mon_ref),
                        ref_annual_data=_squeeze(dataset_ann_ref),
                        save_format=save_format,
                        plot_ensemble_members=plot_ensemble_members,
                        # startdate=startdate,
                        # enddate=enddate,
                    )

                    cli.logger.info("Timeseries diagnostic finished for variable '%s'.", variable)

            cli.logger.info("Ensemble timeseries diagnostic completed!")

    # Bias Ensemble Maps
    gb_diag_config = cli.config_dict["diagnostics"]["biases"]

    if "biases" in cli.config_dict["diagnostics"]:
        if cli.config_dict["diagnostics"]["biases"]["run"]:
            cli.logger.info("Running CLI for EnsembleMap diagnostic for a single model")

            params = gb_diag_config.get("params", {}).get("default", {})
            variables = gb_diag_config.get("variables") or []
            all_plot_params = gb_diag_config.get("plot_params", {})
            default_plot = all_plot_params.get("default", {})

            for variable in variables:
                cli.logger.info("2D Maps (lat-lon) — variable: %s", variable)
                var_params = gb_diag_config.get("params", {}).get(variable, {})

                region = None

                realization_list = extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source)
                cli.logger.info(f"Realization list: {realization_list}")
                extra_dict = {"variable": variable}
                filenames = generate_realizations_path_single_model(
                    catalog=catalog,
                    model=model,
                    exp=exp,
                    realization_list=realization_list,
                    diagnostic_name=gb_diag_config.get("diagnostic_name", "biases"),
                    diagnostic_product="annual_climatology",
                    variable=variable,
                    file_dir=cli.outputdir,
                    extra_keys=extra_dict,
                    file_format=".nc",
                    loglevel=cli.loglevel,
                )

                dataset = reader_retrieve_and_merge(
                    filenames=filenames,
                    variable=variable,
                    realizations=realization_list,
                    fix=fixer,
                    loglevel=cli.loglevel,
                )
                if dataset is None:
                    cli.logger.warning(
                        "Unable to load and merge the dataset in Ensemble maps for for variable '%s'. Skipping it!", variable
                    )
                    continue

                ens_latlon = EnsembleMaps(
                    var=variable,
                    dataset=dataset,
                    catalog_list=catalog,
                    model_list=model,
                    exp_list=exp,
                    source_list=source,
                    outputdir=cli.outputdir,
                    loglevel=cli.loglevel,
                )
                ens_latlon.run()

                # Plot parameters (needed with or without a reference)
                var_plot = all_plot_params.get(variable, {})
                plot_params = {**default_plot, **var_plot}
                param_dict = gb_diag_config.get("plot_params", {}).get(variable, {}) or {}

                # Need to define them in the config files
                vmin_ensemble = param_dict.get("vmin_ensemble") or None
                vmax_ensemble = param_dict.get("vmax_ensemble") or None
                vmin_std_ensemble = param_dict.get("vmin_std_ensemble") or None
                vmax_std_ensemble = param_dict.get("vmax_std_ensemble") or None

                # will be taken from the original analysis config file
                vmin_bias = param_dict.get("vmin") or None
                vmax_bias = param_dict.get("vmax") or None
                vmin_std_bias = param_dict.get("vmin_std") or None
                vmax_std_bias = param_dict.get("vmax_std") or None

                cmap = param_dict.get("cmap") or None
                cbar_label = param_dict.get("cbar_label") or None

                # Reference (optional)
                dataset_ref = None
                # TODO:
                # Reference dataset STD bias is not plotted because we do not have reference STD data
                dataset_std_ref = None

                if has_ref:
                    ref_filenames = generate_realizations_path_single_model(
                        catalog=catalog,
                        model=model,
                        exp=exp,
                        catalog_ref=catalog_ref,
                        model_ref=model_ref,
                        exp_ref=exp_ref,
                        realization_ref="r1",  # TODO
                        realization_list=realization_list,
                        diagnostic_name=gb_diag_config.get("diagnostic_name", "biases"),
                        diagnostic_product="annual_climatology",
                        variable=variable,
                        file_dir=cli.outputdir,
                        extra_keys=extra_dict,
                        file_format=".nc",
                        loglevel=cli.loglevel,
                    )
                    cli.logger.info(f"reference file name: {ref_filenames}")
                    dataset_ref = load_reference(
                        cli, ref_filenames, variable, region, fixer_ref, label="map", select_variable=False
                    )
                    cli.logger.debug(f"Reference dataset: {dataset_ref}")

                ens_latlon_plot = PlotEnsembleMaps(
                    catalog_list=catalog,
                    model_list=model,
                    exp_list=exp,
                    source_list=source,
                    ref_catalog=catalog_ref if dataset_ref is not None else None,
                    ref_model=model_ref if dataset_ref is not None else None,
                    ref_exp=exp_ref if dataset_ref is not None else None,
                    region=region,
                    outputdir=cli.outputdir,
                    loglevel=cli.loglevel,
                )

                # Ensemble mean plot
                if ens_latlon.dataset_mean is not None:
                    title = TitleBuilder(diagnostic="Ensemble mean diagnostic", variable=variable, model=model).generate()
                    ens_latlon_plot.plot(
                        var=variable,
                        dataset=ens_latlon.dataset_mean.squeeze(),
                        save_format=save_format,
                        dpi=dpi,
                        proj=plot_params.get("projection", "robinson"),
                        proj_params=plot_params.get("projection_params", {}),
                        vmin=vmin_ensemble,
                        vmax=vmax_ensemble,
                        units=param_dict.get("units"),
                        long_name=param_dict.get("long_name"),
                        transform_first=False,
                        cyclic_lon=True,
                        contour=True,
                        coastlines=True,
                        cbar_label=cbar_label,
                        data_name="ensemble_mean",
                        cmap=cmap,
                        title=title,
                    )

                # Ensemble STD plot
                if ens_latlon.dataset_std is not None:
                    title = TitleBuilder(diagnostic="Ensemble STD diagnostic", variable=variable, model=model).generate()

                    ens_latlon_plot.plot(
                        var=variable,
                        dataset=ens_latlon.dataset_std.squeeze(),
                        save_format=save_format,
                        dpi=dpi,
                        proj=plot_params.get("projection", "robinson"),
                        proj_params=plot_params.get("projection_params", {}),
                        vmin=vmin_std_ensemble,
                        vmax=vmax_std_ensemble,
                        units=param_dict.get("units"),
                        long_name=param_dict.get("long_name"),
                        transform_first=False,
                        cyclic_lon=True,
                        contour=True,
                        coastlines=True,
                        cbar_label=cbar_label,
                        data_name="ensemble_std",
                        cmap=cmap,
                        title=title,
                    )

                # Ensemble mean bias plot (only with a reference)
                if (ens_latlon.dataset_mean is not None) and (dataset_ref is not None):
                    title = TitleBuilder(diagnostic="Mean ensemble bias diagnostic", variable=variable, model=model).generate()

                    ens_latlon_plot.plot_ensemble_diff_bias(
                        var=variable,
                        dataset=ens_latlon.dataset_mean.squeeze(),
                        ref_dataset=dataset_ref.squeeze(),
                        save_format=save_format,
                        dpi=dpi,
                        proj=plot_params.get("projection", "robinson"),
                        proj_params=plot_params.get("projection_params", {}),
                        vmin=vmin_bias,
                        vmax=vmax_bias,
                        units=param_dict.get("units"),
                        long_name=param_dict.get("long_name"),
                        transform_first=False,
                        cyclic_lon=True,
                        contour=True,
                        coastlines=True,
                        cbar_label=cbar_label,
                        data_name="ensemble_bias_mean",
                        cmap=cmap,
                        title=title,
                    )
                elif dataset_ref is None:
                    cli.logger.info("No reference for variable '%s': skipping ensemble bias plots.", variable)

                # Ensemble STD bias plot
                if (ens_latlon.dataset_std is not None) and (dataset_std_ref is not None):
                    title = TitleBuilder(diagnostic="STD ensemble bias diagnostic", variable=variable, model=model).generate()

                    ens_latlon_plot.plot_ensemble_diff_bias(
                        var=variable,
                        dataset=ens_latlon.dataset_std.squeeze(),
                        ref_dataset=dataset_std_ref.squeeze(),
                        save_format=save_format,
                        dpi=dpi,
                        proj=plot_params.get("projection", "robinson"),
                        proj_params=plot_params.get("projection_params", {}),
                        vmin=vmin_std_bias,
                        vmax=vmax_std_bias,
                        units=param_dict.get("units"),
                        long_name=param_dict.get("long_name"),
                        transform_first=False,
                        cyclic_lon=True,
                        contour=True,
                        coastlines=True,
                        cbar_label=cbar_label,
                        data_name="ensemble_bias_std",
                        cmap=cmap,
                        title=title,
                    )

                cli.logger.info("Ensemble maps diagnostic finished for variable '%s'.", variable)

            cli.logger.info("Ensemble maps diagnostic completed!")

    cli.close_dask_cluster()


if __name__ == "__main__":
    main()
