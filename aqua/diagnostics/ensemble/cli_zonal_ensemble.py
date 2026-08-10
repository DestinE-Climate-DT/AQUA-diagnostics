#!/usr/bin/env python3
"""
Command-line interface for ensemble zonalmean diagnostic.

This CLI allows to plot a map of aqua analysis zonalmean
defined in a yaml configuration file for multiple models.
"""

import argparse
import sys
import xarray as xr

from aqua.core.logger import log_configure
from aqua.core.util import get_arg
from aqua.core.logger import log_configure
from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleZonal,
    PlotEnsembleZonal,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    close_cluster,
    DiagnosticCLI,
    open_cluster,
    template_parse_arguments,
    TitleBuilder,
)
from aqua.diagnostics.ensemble import (
    extract_realizations_list,
)    

def parse_arguments(args):
    """Parse command-line arguments for the EnsembleZonal diagnostic CLI.

    Args:
        args (list): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Runs EnsembleZonal diagnostic, "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = template_parse_arguments(parser)

    return parser.parse_args(args)

def main(argv=None):
    """
    Main function for running EnsembleZonal class

    Args:
        argv(list, optional): command-line arguments. Defaults to sys.argv[1:].
    """

    args = parse_arguments(argv if argv is not None else sys.argv[1:])

    # Initialize and prepare CLI
    cli = DiagnosticCLI(args, diagnostic_name='EnsembleZonal', default_config='config_zonalmean_ensemble.yaml')
    
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
    regrid = get_arg(args, "regrid", first.get("regrid"))

    cli.logger.debug(f"Ensemble catalog: {catalog}, model: {model}, exp: {exp}, and source {source}")
 
    ## Single reference 
    #if "references" in cli.config_dict:
    #    ref = cli.config_dict.get("references")
    #    first_ref = ref[0]
    #    catalog_ref = get_arg(args, "catalog", first_ref["catalog"])
    #    model_ref = get_arg(args, "model", first_ref["model"])
    #    exp_ref = get_arg(args, "exp", first_ref["exp"])
    #    source_ref = get_arg(args, "source", first_ref["source"])
    #    
    #    cli.logger.debug(f"Reference catalog: {catalog}")
    #    cli.logger.debug(f"Reference model: {model}")
    #    cli.logger.debug(f"Reference exp: {exp}")
    #    cli.logger.debug(f"Reference source: {source}")

    # Output parameters
    outputdir = cli.config_dict.get("output", {}).get("outputdir", "./")
    rebuild = cli.config_dict.get("output", {}).get("rebuild", True)
    save_netcdf = cli.config_dict.get("output", {}).get("save_netcdf", True)
    save_format = cli.config_dict.get("output", {}).get("save_format", SAVE_FORMAT)
    dpi = cli.config_dict.get("output", {}).get("dpi", 300)
    description_mean = cli.config_dict.get("output", {}).get("description_mean", None)
    description_std = cli.config_dict.get("output", {}).get("description_std", None)
    # Ensemble Zonal
    diag_config = cli.config_dict["diagnostics"]["ensemble"]
    params = diag_config.get("params", {}).get("default", {})

    for variable in cli.config_dict["diagnostics"]["ensemble"].get("variable", None):
        for region in cli.config_dict["diagnostics"]["ensemble"].get("region", None):
            cli.logger.info(f"Variable under consideration: {variable}")
            title_mean = cli.config_dict["diagnostics"]["ensemble"]["plot_params"]["default"].get("title_mean", None)
            title_std = cli.config_dict["diagnostics"]["ensemble"]["plot_params"]["default"].get("title_std", None)
            cbar_label = cli.config_dict["diagnostics"]["ensemble"]["plot_params"]["default"].get("cbar_label", None)
            figure_size = cli.config_dict["diagnostics"]["ensemble"]["plot_params"]["default"].get("figure_size", None)
 
            # Model data
            models = cli.config_dict["datasets"]

            catalog_list = []
            model_list = []
            exp_list = []
            source_list = []
            realization_dict = {}
            if models is not None:
                models[0]["catalog"] = get_arg(args, "catalog", models[0]["catalog"])
                models[0]["model"] = get_arg(args, "model", models[0]["model"])
                models[0]["exp"] = get_arg(args, "exp", models[0]["exp"])
                models[0]["source"] = get_arg(args, "source", models[0]["source"])
                models[0]["realization"] = get_arg(args, "realization", models[0]["realization"])
                if models[0]["realization"] is None: 
                    models[0]["realization"] = extract_realizations_list(catalog=models[0]["catalog"], model=models[0]["model"], exp=models[0]["exp"], source=models[0]["source"])
 

                for model in models:
                    catalog_list.append(model["catalog"])
                    model_list.append(model["model"])
                    exp_list.append(model["exp"])
                    source_list.append(model["source"])
                    if model["realization"] is None: 
                        model["realization"] = extract_realizations_list(catalog=model["catalog"], model=model["model"], exp=model["exp"], source=model["source"])
                    realization_dict.update({model["model"]: model["realization"]})

            # Loading and merging data
            ens_dataset = reader_retrieve_and_merge(
                region=region,
                variable=variable,
                catalog_list=catalog_list,
                model_list=model_list,
                exp_list=exp_list,
                source_list=source_list,
                regrid=False,
                areas=False,
                fix=True,
                realization=realization_dict,
                ens_dim="ensemble",
                loglevel=cli.loglevel,
            )

            # Initialize EnsembleZonal class
            ens_zm = EnsembleZonal(
                var=variable,
                dataset=ens_dataset,
                catalog_list=catalog_list,
                model_list=model_list,
                exp_list=exp_list,
                source_list=source_list,
                outputdir=outputdir,
                loglevel=cli.loglevel,
            )
            ens_zm.run()

            # Initialize PlotEnsembleZonal class
            plot_class_arguments = {
                "catalog_list": catalog_list,
                "model_list": model_list,
                "exp_list": exp_list,
                "source_list": source_list,
                "outputdir": outputdir,
            }

            ens_zm_plot = PlotEnsembleZonal(**plot_class_arguments,loglevel=cli.loglevel)

            # PlotEnsembleLatLon plot options
            plot_arguments = {
                "save_format": save_format,
                "var": variable,
                "cbar_label": None,
            }
            if not description_mean: description_mean = f"Ensemble mean zonal diagnostic"
            if not description_std: description_std = f"Ensemble STD zonal diagnostic"
            ens_zm_plot.plot(**plot_arguments, dataset=ens_zm.dataset_mean, description=description_mean, data_name="ensemble_zonal_mean")
            ens_zm_plot.plot(**plot_arguments, dataset=ens_zm.dataset_std, description=description_std, data_name="ensemble_zonal_std")
 
            cli.logger.info(f"Finished Ensemble_Zonal diagnostic for {variable}.")

if __name__ == "__main__":
    main()
