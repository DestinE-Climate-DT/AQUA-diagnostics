#!/usr/bin/env python3
"""
Command-line interface for ensemble 2D Maps Lat-Lon diagnostic.

This CLI allows to plot a map of aqua analysis atmglobalmean
defined in a yaml configuration file for multiple models.
"""

import argparse
import sys

import xarray as xr

from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleMaps,
    PlotEnsembleMaps,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    DiagnosticCLI,
    TitleBuilder,
    template_parse_arguments,
)
from aqua.diagnostics.ensemble import (
    extract_realizations_list,
    generate_realizations_path,
)

DEFAULT_CONFIG = "config_multi_model_maps_ensemble.yaml"


def parse_arguments(args):
    """Parse command-line arguments for the
    EnsembleMaps for single mulit-model diagnostic CLI.

    Args:
        args (list): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Runs Multi-model ensembleMaps diagnostic, "),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    return parser.parse_args(args)


def main(argv=None):
    """
    Main function for running EnsembleMaps class

    Args:
        argv(list, optional): command-line arguments. Defaults to sys.argv[1:].
    """
    # TODO:
    # reader_kwargs is assigned as an empty dictionay.
    # Need to test its implementation in the 'reader_retrieve_and_merge' function

    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(args, diagnostic_name="EnsembleMaps", default_config=DEFAULT_CONFIG)

    # Preparing Dask cluster
    cli.prepare()
    cli.open_dask_cluster()

    # Output parameters
    outputdir = cli.config_dict.get("output", {}).get("outputdir", "./")
    cli.config_dict.get("output", {}).get("rebuild", True)
    cli.config_dict.get("output", {}).get("save_netcdf", True)
    save_format = cli.config_dict.get("output", {}).get("save_format", SAVE_FORMAT)
    dpi = cli.config_dict.get("output", {}).get("dpi", 300)

    # Ensemble map diagnostic
    diag_config = cli.config_dict["diagnostics"]["ensemble"]

    diag_config.get("params", {}).get("default", {})
    variables = diag_config.get("variables") or []
    all_plot_params = diag_config.get("plot_params", {})
    default_plot = all_plot_params.get("default", {})

    region = None

    # Single reference
    if "references" in cli.config_dict:
        ref = cli.config_dict.get("references")
        first_ref = ref[0]
        catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
        model_ref = get_arg(args, "model", first_ref["model"])
        exp_ref = get_arg(args, "exp", first_ref["exp"])
        source_ref = get_arg(args, "source", first_ref["source"])
        get_arg(args, "fix", first_ref.get("fix"))

        cli.logger.debug(
            f"Reference catalog: {catalog_ref}, model: {{model_ref}}, exp: {{exp_ref}} and source: {{source_ref}}"
        )

    # EnsembleMaps diagnostic
    if cli.config_dict["diagnostics"]["ensemble"]["run"]:
        # Variables in Timeseries config
        for variable in variables:
            diag_config.get("params", {}).get(variable, {})

            cli.logger.info("Ensemble Map diagnostics for variable: %s", variable)

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
                models[0]["fix"] = get_arg(args, "fix", models[0]["fix"])

                for model in models:
                    catalog_list.append(model["catalog"])
                    model_list.append(model["model"])
                    exp_list.append(model["exp"])
                    source_list.append(model["source"])
                    if model["realization"] is None:
                        realization_list.append(model["realization"])
                    else:
                        realization = extract_realizations_list(
                            catalog=model["catalog"],
                            model=model["model"],
                            exp=model["exp"],
                            source=model["source"],
                        )
                        realization_list.append(realization)

                # Reterive data
                dataset = reader_retrieve_and_merge(
                    variable=variable,
                    catalog_list=catalog_list,
                    model_list=model_list,
                    exp_list=exp_list,
                    source_list=source_list,
                    realization=realization_list,
                    # fix=fix,
                    # areas=areas,
                    # regrid=regrid,
                    loglevel=cli.loglevel,
                )

                if dataset is None:
                    cli.logger.warning("Ensemble map data is not provided.")

                ens_latlon = EnsembleMaps(
                    var=variable,
                    dataset=dataset,
                    catalog_list=catalog_list,
                    model_list=model_list,
                    exp_list=exp_list,
                    source_list=source_list,
                    outputdir=outputdir,
                    loglevel=cli.loglevel,
                )
                ens_latlon.run()

                # Reference
                dataset_ref = None
                # TODO:
                # Reference dataset STD bias is not plotted because we do not have reference STD data
                dataset_std_ref = None

                extra_dict = {"variable": variable, "region": region}
                ref_realization = extract_realizations_list(
                    catalog=catalog_ref, model=model_ref, exp=exp_ref, source=source_ref
                )
                ref_filenames = generate_realizations_path(
                    catalog=catalog_ref,
                    model=model_ref,
                    exp=exp_ref,
                    realization_list=ref_realization,
                    diagnostic_name="globalbiases",
                    diagnostic_product="annual_climatology",
                    variable=variable,
                    file_dir=outputdir,
                    extra_keys=extra_dict,
                    file_format=".nc",
                    loglevel=cli.loglevel,
                )
                if ref_filenames:
                    dataset_ref = reader_retrieve_and_merge(
                        filenames=ref_filenames,
                        variable=variable,
                        # catalog=catalog_ref,
                        # model=model_ref,
                        # exp=exp_ref,
                        # source=source_ref,
                        region=region,
                        realization=ref_realization,
                        # fix=fixer_ref,
                        loglevel=cli.loglevel,
                    )

                    if dataset_ref is None:
                        cli.logger.warning(
                            "Unable to load reference map for variable '%s', region '%s'.",
                            variable,
                            region,
                        )
                        continue

                    if dataset_ref:
                        dataset_ref = dataset_ref.squeeze("ensemble")
                        if isinstance(dataset_ref, xr.Dataset):
                            dataset_ref = dataset_ref[variable]
                    # Merge default and per-variable plot parameters
                    var_plot = all_plot_params.get(variable, {})
                    plot_params = {**default_plot, **var_plot}
                    param_dict = diag_config.get("plot_params", {}).get(variable, {}) or {}

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

                ens_latlon_plot = PlotEnsembleMaps(
                    catalog_list=catalog_list,
                    model_list=model_list,
                    exp_list=exp_list,
                    source_list=source_list,
                    ref_catalog=catalog_ref,
                    ref_model=model_ref,
                    ref_exp=exp_ref,
                    region=region,
                    outputdir=outputdir,
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

                # Ensemble mean bias plot
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


if __name__ == "__main__":
    main()
