"""
Command-line interface for VariabilityMap diagnostic.

This CLI allows to plot maps of VariabilityMap (STD in time dimension)
defined in a yaml configuration file for single model and reference dataset.
"""

import argparse
import sys

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.core.util import time_to_string
from aqua.diagnostics.base import DiagnosticCLI, template_parse_arguments
from aqua.diagnostics.variability_map import PlotVariabilityMap, VariabilityMap

# Default config, in aqua/diagnostics/config/collections/legacy/ocean2d/
DEFAULT_CONFIG = "config-ocean2d-aviso.yaml"
DEFAULT_DIAGNOSTIC_NAME = "ocean2d"
TOOLNAME = "variabilitymap"


def parse_arguments(args):
    """
    Parse command-line arguments for VariabilityMap diagnostic.

    Args:
        args (list[str]): list of command-line arguments to parse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="VariabilityMap CLI")
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    """
    Run the VariabilityMap diagnostic CLI.

    Args:
        argv (list[str], optional): command-line arguments. Defaults to sys.argv[1:].
    """
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name=DEFAULT_DIAGNOSTIC_NAME, default_config=DEFAULT_CONFIG).prepare()
    cli.open_dask_cluster()
    try:
        tool_dict = cli.config_dict["diagnostics"].get(TOOLNAME) or {}
        if not tool_dict.get("run", False):
            cli.logger.info("VariabilityMap is disabled, nothing to do")
            return
        diagnostic_name = tool_dict.get("diagnostic_name", "variability_map")
        dataset = cli.config_dict["datasets"][0]
        # Only the first dataset and the first reference are used
        entries = [(cli.dataset_args(dataset), dataset)]
        references = cli.config_dict.get("references") or []
        if references:
            entries.append((cli.reference_args(references[0]), references[0]))

        plot_params = tool_dict.get("plot_params", {})
        default_plot = plot_params.get("default", {})
        sub_region = plot_params.get("sub_region") or {}
        common_args = {
            "tgt_grid_name": default_plot.get("tgt_grid_name"),
            "regrid_method": default_plot.get("regrid_method"),
            "save_format": cli.save_format,
            "dpi": cli.dpi,
            "rebuild": cli.rebuild,
        }
        map_args = {
            **common_args,
            "proj": default_plot.get("projection", "robinson"),
            "proj_params": default_plot.get("projection_params", {}),
            "contour": default_plot.get("contour", False),
            "vmin": default_plot.get("vmin"),
            "vmax": default_plot.get("vmax"),
            "plot_options": {"cmap": default_plot["cmap"]} if default_plot.get("cmap") else {},
        }
        region_args = {
            **map_args,
            "proj": sub_region.get("projection", "plate_carree"),
            "proj_params": sub_region.get("projection_params", {}),
            "contour": sub_region.get("contour", False),
            "region": sub_region.get("name"),
            "lon_limits": sub_region.get("lon_limits"),
            "lat_limits": sub_region.get("lat_limits"),
        }

        plot_class = PlotVariabilityMap(diagnostic_name=diagnostic_name, outputdir=cli.outputdir, loglevel=cli.loglevel)

        for var in tool_dict.get("variables") or []:
            cli.logger.info("VariabilityMap diagnostic for %s", var)
            try:
                std_maps = [_compute(cli, dataset_args, entry, var, diagnostic_name) for dataset_args, entry in entries]
            except (NoDataError, NotEnoughDataError, KeyError, ValueError) as e:
                cli.logger.warning("Skipping %s: %s", var, e)
                continue

            for std_map in std_maps:
                info = _plot_info(std_map, var)
                plot_class.plot(dataset_std=std_map.data_std, **info, **map_args)
                if region_args["region"] is not None:
                    plot_class.plot(dataset_std=std_map.data_std, **info, **region_args)

            if len(std_maps) == 2:
                model_map, ref_map = std_maps
                ref_info = {f"{key}_ref": value for key, value in _plot_info(ref_map, var).items() if key != "var"}
                plot_class.plot_diff(
                    dataset_std=model_map.data_std,
                    dataset_std_ref=ref_map.data_std,
                    **_plot_info(model_map, var),
                    **ref_info,
                    **common_args,
                    proj=map_args["proj"],
                    proj_params=map_args["proj_params"],
                    contour=True,
                )
            cli.logger.info("VariabilityMap diagnostic for %s completed.", var)
    finally:
        cli.close_dask_cluster()


def _compute(cli, dataset_args, entry, var, diagnostic_name):
    """Compute (and save) the STD map of one dataset or reference."""
    std_map = VariabilityMap(
        **dataset_args,
        var=var,
        reader_kwargs=entry.get("reader_kwargs") or {},
        diagnostic_name=diagnostic_name,
        loglevel=cli.loglevel,
    )
    std_map.run(outputdir=cli.outputdir, rebuild=cli.rebuild, save_netcdf=cli.save_netcdf)
    return std_map


def _plot_info(std_map, var):
    """Dataset identifiers and the period actually retrieved, for titles and filenames."""
    return {
        "var": var,
        "catalog": std_map.catalog,
        "model": std_map.model,
        "exp": std_map.exp,
        "startdate": time_to_string(std_map.startdate),
        "enddate": time_to_string(std_map.enddate),
    }


if __name__ == "__main__":
    main()
