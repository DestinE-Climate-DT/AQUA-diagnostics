"""
Command-line interface for VariabilityMap diagnostic.

This CLI allows to plot maps of VariabilityMaps (STD in time dimension)
defined in a yaml configuration file for single model and reference dataset.

TODO: 
- implement freq variable
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

# Default config filename in directory aqua/diagnostics/config/collections/legacy/ocean2d/config-ocean2d-aviso.yaml)
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
    catalog = get_arg(args, "catalog", first["catalog"] or None)
    model = get_arg(args, "model", first["model"] or None)
    exp = get_arg(args, "exp", first["exp"] or None)
    source = get_arg(args, "source", first["source"] or None)
    get_arg(args, "regrid", first.get("regrid") or False)
    fixer = get_arg(args, "fix", first.get("fix") or True)
    realization = get_arg(args, "realization", first.get("realization") or None)
    reader_kwargs = get_arg(args, "reader_kwargs",first.get("reader_kwargs") or {})
    startdate_data = get_arg(args, "startdate", first.get("startdate") or None)
    enddate_data = get_arg(args, "enddate", first.get("enddate") or None)

    if realization:
        cli.logger.info(f"Realization option is set to {realization}")
        reader_kwargs.update({"realization": realization})

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
        startdate_ref = get_arg(args, "startdate", first.get("startdate") or None)
        enddate_ref = get_arg(args, "enddate", first.get("enddate") or None)


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
            proj = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("projection", "robinson")
            proj_params = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("projection_params", {})

            # Variables in the config file
            variables = diag_config.get("variables") or []
            for variable in variables:
                var_params = diag_config.get("params", {}).get(variable, {})

                cli.logger.debug(f"Using projection: {proj} for variable: {variable}")
                vmin = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("vmin", None)
                vmax = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("vmax", None)
                # Regridder options for plots
                tgt_grid_name = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("tgt_grid_name", None)
                regrid_method = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["default"].get("regrid_method", None)

                # Sub region selection
                region_name = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["sub_region"].get("name", None)
                region_proj = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["sub_region"].get(
                    "projection", "plate_carree"
                )
                region_proj_params = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["sub_region"].get(
                    "projection_params", {}
                )

                lon_limits = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["sub_region"].get("lon_limits", None)
                lat_limits = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["sub_region"].get("lat_limits", None)

                mask_northern_boundary = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["mask_options"].get(
                    "mask_northern_boundary", None
                )
                mask_southern_boundary = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["mask_options"].get(
                    "mask_southern_boundary", None
                )
                northern_boundary_latitude = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["mask_options"].get(
                    "northern_boundary_latitude", None
                )
                southern_boundary_latitude = config_dict["diagnostics"]["VariabilityMap"]["plot_params"]["mask_options"].get(
                    "southern_boundary_latitude", None
                )


                # Initialize VariabilityMap for model dataset
                if (model is not None) and (exp is not None) and (source is not None):
                    std_dataset = VariabilityMap(
                        catalog=catalog,
                        model=model,
                        exp=exp,
                        source=source,
                        var=variable,
                        startdate=startdate_data,
                        enddate=enddate_data,
                        reader_kwargs=reader_kwargs,
                    )
                    # Perform computation here for model dataset
                    std_dataset.run()

                # Initialize VariabilityMap for reference dataset
                if (model_ref is not None) and (exp_ref is not None) and (source_ref is not None):
                    std_dataset_ref = VariabilityMap(
                        catalog=catalog_ref,
                        model=model_ref,
                        exp=exp,
                        source=source_ref,
                        var=variable_ref,
                        startdate=startdate_ref,
                        enddate=enddate_ref,
                        #reader_kwargs=reader_kwargs, #TODO: define a separate dict here
                    )
                    # Perform computation here for reference dataset
                    std_dataset_ref.run()


                # Initialize plotting class
                plot_class = PlotVariabilityMap(outputdir=outputdir, loglevel=cli.loglevel)

                # Dictionary for dataset plot
                if std_dataset.data_std is not None:
                    plot_arguments_dataset = {
                        "var": variable,
                        "catalog": catalog,
                        "model": model,
                        "exp": exp,
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
                    plot_class.plot(dataset_std=std_dataset.data_std, **plot_arguments_dataset)

                # Dictionary for sub-region dataset plot
                if std_dataset.data_std is not None and region_name is not None:
                    plot_arguments_dataset = {
                        "var": variable,
                        "catalog": catalog,
                        "model": model,
                        "exp": exp,
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
                    plot_class.plot(dataset_std=std_dataset.data_std, **plot_arguments_dataset)

                # Dictionary for reference plot
                if std_dataset_ref.data_std is not None:
                    plot_arguments_ref = {
                        "var": variable,
                        "catalog": catalog_ref,
                        "model": model_ref,
                        "exp": exp_ref,
                        "startdate": startdate_ref,
                        "enddate": enddate_ref,
                        "proj": proj,
                        "proj_params": proj_params,
                        "vmin": vmin,
                        "vmax": vmax,
                        "tgt_grid_name": tgt_grid_name,
                        "regrid_method": regrid_method,
                    }
                    plot_class.plot(dataset_std=std_dataset_ref.data_std, **plot_arguments_ref)

                # Dictionary for sub-region reference plot
                if std_dataset_ref.data_std is not None:
                    plot_arguments_ref = {
                        "var": variable,
                        "catalog": catalog_ref,
                        "model": model_ref,
                        "exp": exp_ref,
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
                    plot_class.plot(dataset_std=std_dataset_ref.data_std, **plot_arguments_ref)

                # Dictionary for difference of Variability maps plot
                if std_dataset.data_std is not None and std_dataset_ref.data_std is not None:
                    plot_arguments_diff = {
                        "var": variable,
                        "catalog": catalog,
                        "model": model,
                        "exp": exp,
                        "catalog_ref": catalog_ref,
                        "model_ref": model_ref,
                        "exp_ref": exp_ref,
                        "save_format": save_format,
                        "startdate": startdate_data,
                        "enddate": enddate_data,
                        "startdate_ref": startdate_ref,
                        "enddate_ref": enddate_ref,
                        "tgt_grid_name": tgt_grid_name,
                        "regrid_method": regrid_method,
                    }
                    plot_class.plot_diff(dataset_std=std_dataset.data_std, dataset_std_ref=std_dataset_ref.data_std, **plot_arguments_diff)

                cli.logger.info(f"VariabilityMap diagnostic for {variable} completed.")
    cli.close_dask_cluster()

