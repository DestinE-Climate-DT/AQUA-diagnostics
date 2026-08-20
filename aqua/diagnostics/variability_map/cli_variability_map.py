"""
Command-line interface for VariabilityMap diagnostic.

This CLI allows to plot maps of VariabilityMaps (STD in time dimension)
defined in a yaml configuration file for single model and reference dataset.
"""

import sys
import argparse
from aqua.core.util import get_arg

from aqua.diagnostics import (
    VariabilityMap, 
    PlotVariabilityMap,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    DiagnosticCLI,
    TitleBuilder,
    template_parse_arguments,
)

# Default config filenames (resolved by load_diagnostic_config from the
# package's config/collections/legacy/ocean2d/ directory)
DEFAULT_CONFIG = "config-ocean2d-aviso.yaml"
DEFAULT_DIAGNOSTIC_NAME = "ocean2d"

def parse_arguments(args):
    """Parse command-line arguments for VariabilityMap diagnostic.

    Args:
        args (list): list of command-line arguments to parse.
    """
    parser = argparse.ArgumentParser(description="VariabilityMap CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)

def main(argv=None):
    """Run the VariabilityMap diagnostic CLI.

    Args:
        argv (list, optional): command-line arguments. Defaults to sys.argv[1:].
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    cli = DiagnosticCLI(
        args,
        diagnostic_name=DEFAULT_DIAGNOSTIC_NAME,
        default_config=DEFAULT_CONFIG,
    )
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
    realization = get_arg(args, "realization", None)
    reader_kwargs = get_arg(args, "reader_kwargs", {})

    if realization:
        cli.logger.info(f"Realization option is set to {realization}")
        reader_kwargs = {"realization": realization}

    if dataset["zoom"]: reader_kwargs.update({"zoom": dataset["zoom"]})

    cli.logger.debug(f"VariabilityMap diagnostic catalog: {catalog}, model: {model}, exp: {exp}, and source {source}")

    # Single reference
    if "references" in cli.config_dict:
        ref = cli.config_dict.get("references")
        first_ref = ref[0]
        catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
        model_ref = get_arg(args, "model", first_ref["model"])
        exp_ref = get_arg(args, "exp", first_ref["exp"])
        source_ref = get_arg(args, "source", first_ref["source"])
        fixer_ref = get_arg(args, "fix", first_ref.get("fix"))

        cli.logger.debug(f"VariabilityMap diagnostic reference catalog: {catalog_ref}, model: {model_ref}, exp: {exp_ref} and source: {source_ref}")

    # Output parameters
    outputdir = cli.config_dict.get("output", {}).get("outputdir", "./")
    cli.config_dict.get("output", {}).get("rebuild", True)
    cli.config_dict.get("output", {}).get("save_netcdf", True)
    save_format = cli.config_dict.get("output", {}).get("save_format", SAVE_FORMAT)
    dpi = cli.config_dict.get("output", {}).get("dpi", 300)

    # Variability Map 
    diag_config = cli.config_dict["diagnostics"]["VariabilityMap"]
    if "VariabilityMap" in config_dict["diagnostics"]:
        if config_dict["diagnostics"]["VariabilityMap"]["run"]:
            cli.logger.info("Running VariabilityMap diagnostic.")

            params = diag_config.get("params", {}).get("default", {})
            all_plot_params = diag_config.get("plot_params", {})
            default_plot = all_plot_params.get("default", {})
            #proj = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("projection", "robinson")
            #proj_params = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("projection_params", {})

            startdate = params.get("startdate")
            enddate = params.get("enddate")
            if startdate is None:
                startdate = get_arg(args, "startdate", first.get("startdate") or None)
            if enddate is None:
                enddate = get_arg(args, "enddate", first.get("enddate") or None)
            
            # Variables in the config file
            variables = diag_config.get("variables") or []
            for variable in variables:
                var_params = diag_config.get("params", {}).get(variable, {})




            logger.debug(f"Using projection: {proj} for variable: {variable}")
            vmin = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("vmin", None)
            vmax = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("vmax", None)
            # Regridder options for plots
            tgt_grid_name = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("tgt_grid_name", None)
            regrid_method = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["default"].get("regrid_method", None)

            # Sub region selection
            region_name = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["sub_region"].get("name", None)
            region_proj = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["sub_region"].get(
                "projection", "plate_carree"
            )
            region_proj_params = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["sub_region"].get(
                "projection_params", {}
            )

            lon_limits = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["sub_region"].get("lon_limits", None)
            lat_limits = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["sub_region"].get("lat_limits", None)

            mask_northern_boundary = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["mask_options"].get(
                "mask_northern_boundary", None
            )
            mask_southern_boundary = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["mask_options"].get(
                "mask_southern_boundary", None
            )
            northern_boundary_latitude = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["mask_options"].get(
                "northern_boundary_latitude", None
            )
            southern_boundary_latitude = config_dict["diagnostics"]["ssh_variability"]["plot_params"]["mask_options"].get(
                "southern_boundary_latitude", None
            )


            # Initialize SSH Variability for model dataset
            if (
                (dataset_dict["catalog"] is not None)
                or (dataset_dict["model"] is not None)
                or (dataset_dict["exp"] is not None)
                or (dataset_dict["source"] is not None)
            ):
                ssh_dataset = ssh_variability_compute(
                    **dataset_dict,
                    var=variable,
                    startdate=startdate_data,
                    enddate=enddate_data,
                    reader_kwargs=reader_kwargs,
                )
                # Perform computation here for model dataset
                ssh_dataset.run()

            # Initialize SSH Variability for reference dataset
            if (
                (dataset_dict_ref["catalog"] is not None)
                or (dataset_dict_ref["model"] is not None)
                or (dataset_dict_ref["exp"] is not None)
                or (dataset_dict_ref["source"] is not None)
            ):
                ssh_ref = ssh_variability_compute(
                    **dataset_dict_ref,
                    var=variable,
                    startdate=startdate_ref,
                    enddate=enddate_ref,
                    # reader_kwargs=reader_kwargs,
                )
                # Perform computation here for reference dataset
                ssh_ref.run()

            # Initialize plotting class
            plot_class = ssh_variability_plot(outputdir=outputdir, loglevel=loglevel)

            # Dictionary for dataset plot
            if ssh_dataset.data_std is not None:
                plot_arguments_dataset = {
                    "var": variable,
                    "catalog": dataset["catalog"],
                    "model": dataset["model"],
                    "exp": dataset["exp"],
                    "save_format": save_format,
                    "startdate": startdate_data,
                    "enddate": enddate_data,
                    "proj": proj,
                    "proj_params": proj_params,
                    "vmin": vmin,
                    "vmax": vmax,
                    "tgt_grid_name": tgt_grid_name,
                    "regrid_method": regrid_method,
                }
                plot_class.plot(dataset_std=ssh_dataset.data_std, **plot_arguments_dataset)

            # Dictionary for sub-region dataset plot
            if ssh_dataset.data_std is not None and region_name is not None:
                plot_arguments_dataset = {
                    "var": variable,
                    "catalog": dataset["catalog"],
                    "model": dataset["model"],
                    "exp": dataset["exp"],
                    "startdate": startdate_data,
                    "enddate": enddate_data,
                    "proj": region_proj,
                    "proj_params": region_proj_params,
                    "vmin": vmin,
                    "vmax": vmax,
                    "region": region_name,
                    "lon_limits": lon_limits,
                    "lat_limits": lat_limits,
                    "mask_northern_boundary": mask_northern_boundary,
                    "mask_southern_boundary": mask_southern_boundary,
                    "northern_boundary_latitude": northern_boundary_latitude,
                    "southern_boundary_latitude": southern_boundary_latitude,
                    "tgt_grid_name": tgt_grid_name,
                    "regrid_method": regrid_method,
                }
                plot_class.plot(dataset_std=ssh_dataset.data_std, **plot_arguments_dataset)

            # Dictionary for reference plot
            if ssh_ref.data_std is not None:
                plot_arguments_ref = {
                    "var": variable,
                    "catalog": dataset_ref["catalog"],
                    "model": dataset_ref["model"],
                    "exp": dataset_ref["exp"],
                    "startdate": startdate_ref,
                    "enddate": enddate_ref,
                    "proj": proj,
                    "proj_params": proj_params,
                    "vmin": vmin,
                    "vmax": vmax,
                    "tgt_grid_name": tgt_grid_name,
                    "regrid_method": regrid_method,
                }
                plot_class.plot(dataset_std=ssh_ref.data_std, **plot_arguments_ref)

            # Dictionary for sub-region reference plot
            if ssh_ref.data_std is not None:
                plot_arguments_ref = {
                    "var": variable,
                    "catalog": dataset_ref["catalog"],
                    "model": dataset_ref["model"],
                    "exp": dataset_ref["exp"],
                    "startdate": startdate_ref,
                    "enddate": enddate_ref,
                    "region": region_name,
                    "lon_limits": lon_limits,
                    "lat_limits": lat_limits,
                    "proj": region_proj,
                    "proj_params": region_proj_params,
                    "vmin": vmin,
                    "vmax": vmax,
                    "tgt_grid_name": tgt_grid_name,
                    "regrid_method": regrid_method,
                }
                plot_class.plot(dataset_std=ssh_ref.data_std, **plot_arguments_ref)

            # Dictionary for difference of ssh_variability plot
            if ssh_dataset.data_std is not None and ssh_ref.data_std is not None:
                plot_arguments_diff = {
                    "var": variable,
                    "catalog": dataset["catalog"],
                    "model": dataset["model"],
                    "exp": dataset["exp"],
                    "catalog_ref": dataset_ref["catalog"],
                    "model_ref": dataset_ref["model"],
                    "exp_ref": dataset_ref["exp"],
                    "save_format": save_format,
                    "startdate": startdate_data,
                    "enddate": enddate_data,
                    "startdate_ref": startdate_ref,
                    "enddate_ref": enddate_ref,
                    "tgt_grid_name": tgt_grid_name,
                    "regrid_method": regrid_method,
                }
                plot_class.plot_diff(dataset_std=ssh_dataset.data_std, dataset_std_ref=ssh_ref.data_std, **plot_arguments_diff)

            logger.info(f"Finished SSH Variability diagnostic for {variable}.")

    # Close the Dask client and cluster
    close_cluster(client=client, cluster=cluster, private_cluster=private_cluster, loglevel=loglevel)
