#!/usr/bin/env python3
"""
Command-line interface for ensemble diagnostics.

Runs the ensemble diagnostics for a single model, from the output of each realization.
Each block of the ``diagnostics`` section enables one product:

- ``timeseries``: EnsembleTimeseries
- ``biases``: EnsembleMaps of the annual climatology and, with ``seasons: true``
  in the params, of the seasonal climatology
- ``seasonalcycles``: EnsembleProfiles of the seasonal cycles
- ``lat_lon_profiles``: EnsembleProfiles of the longterm and seasonal latitude profiles
- ``zonal``: EnsembleZonal of the zonal mean of the 3D annual climatologies
- ``seaice_2d_bias``: EnsembleMaps of the monthly sea ice climatologies of each hemisphere
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
import calendar
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.core.util import get_arg
from aqua.diagnostics import (
    EnsembleMaps,
    EnsembleProfiles,
    EnsembleTimeseries,
    EnsembleZonal,
    PlotEnsembleMaps,
    PlotEnsembleProfiles,
    PlotEnsembleTimeseries,
    PlotEnsembleZonal,
    reader_retrieve_and_merge,
)
from aqua.diagnostics.base import (
    SAVE_FORMAT,
    DiagnosticCLI,
    TitleBuilder,
    template_parse_arguments,
)
from aqua.diagnostics.ensemble import extract_realizations_list, generate_realizations_path_single_model
from aqua.diagnostics.ensemble.plot_ensemble_maps import SEASONS

# Default config filenames (resolved by load_diagnostic_config from the
# package's config/collections/legacy/atmosphere2d/ directory)
DEFAULT_CONFIG = "config-atmosphere2d-berkeley-ensemble.yaml"
DEFAULT_DIAGNOSTIC_NAME = "atmosphere2d"
SEASONAL_PRODUCT = "EnsembleSeasonalMaps"
SEASONAL_CYCLE_PRODUCT = "EnsembleSeasonalCycles"
SEAICE_PRODUCT = "EnsembleSeaIceMaps"
MIN_MEMBERS = 2  # A smaller ensemble has no spread

# Filename keys of the periods of the latitude profiles
PROFILE_PERIODS = {
    "longterm": {"freq": "longterm"},
    **{season: {"freq": "seasonal", "season": season.lower()} for season in SEASONS},
}


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


@dataclass
class EnsembleContext:
    """Everything the products of one ensemble have in common."""

    cli: DiagnosticCLI
    catalog: str
    model: str
    exp: str
    source: str
    fixer: bool
    fixer_ref: bool
    realization_list: list
    ref_args: dict
    save_format: list
    dpi: int

    @property
    def names(self):
        """Dataset names, as expected by the ensemble classes."""
        return {"catalog_list": self.catalog, "model_list": self.model, "exp_list": self.exp, "source_list": self.source}

    def paths(self, diagnostic_name, diagnostic_product, variable, extra_keys, ref_args=None):
        """Return the file of each realization, or the candidate files of the reference if `ref_args` is given."""
        return generate_realizations_path_single_model(
            catalog=self.catalog,
            model=self.model,
            exp=self.exp,
            realization_list=self.realization_list,
            diagnostic_name=diagnostic_name,
            diagnostic_product=diagnostic_product,
            variable=variable,
            file_dir=self.cli.outputdir,
            extra_keys=extra_keys,
            file_format=".nc",
            loglevel=self.cli.loglevel,
            **(ref_args or {}),
        )

    def load(self, filenames, variable):
        """
        Merge the available members of one variable.

        Args:
            filenames (list[str]): File of each realization. Glob patterns are accepted.
            variable (str): Variable to load.

        Returns:
            xarray.Dataset | None: The merged members, None if less than MIN_MEMBERS are available.
        """
        found = {r: sorted(glob.glob(f)) for r, f in zip(self.realization_list, filenames)}
        available = {r: files[0] for r, files in found.items() if files}

        missing = [r for r in self.realization_list if r not in available]
        if missing:
            self.cli.logger.warning("No file for variable '%s' in realizations %s: %s", variable, missing, filenames)
        if len(available) < MIN_MEMBERS:
            self.cli.logger.warning("Less than %d members available for variable '%s'. Skipping it!", MIN_MEMBERS, variable)
            return None

        return reader_retrieve_and_merge(
            filenames=list(available.values()),
            variable=variable,
            realizations=list(available),
            fix=self.fixer,
            loglevel=self.cli.loglevel,
        )

    def reference(self, diagnostic_name, diagnostic_product, variable, extra_keys, label, ref_args=None):
        """Load a reference as a DataArray, returning None if there is no reference or it is not available."""
        ref_args = ref_args or self.ref_args
        if not ref_args:
            return None
        filenames = self.paths(diagnostic_name, diagnostic_product, variable, extra_keys, ref_args)
        return load_reference(self.cli, filenames, variable, None, self.fixer_ref, label=label)

    def ref_names(self, ref_args=None):
        """Reference names for the plot classes."""
        ref_args = ref_args or self.ref_args or {}
        return {
            "ref_catalog": ref_args.get("catalog_ref"),
            "ref_model": ref_args.get("model_ref"),
            "ref_exp": ref_args.get("exp_ref"),
        }


def run_seasonal_maps(ctx, diag_config, variable):
    """
    Compute and plot the ensemble statistics of the seasonal climatology of one variable.

    Args:
        ctx (EnsembleContext): The ensemble to process.
        diag_config (dict): The ``biases`` block of the configuration.
        variable (str): Variable to process.
    """
    cli = ctx.cli
    diagnostic_name = diag_config.get("diagnostic_name", "biases")
    path_args = (diagnostic_name, "seasonal_climatology", variable, {"variable": variable})

    dataset = ctx.load(ctx.paths(*path_args), variable)
    if dataset is None:
        cli.logger.warning("Unable to load the seasonal climatology for variable '%s'. Skipping it!", variable)
        return

    ens_seasonal = EnsembleMaps(
        var=variable,
        dataset=dataset,
        diagnostic_product=SEASONAL_PRODUCT,
        outputdir=cli.outputdir,
        loglevel=cli.loglevel,
        **ctx.names,
    )
    ens_seasonal.run()

    dataset_ref = ctx.reference(*path_args, label="seasonal map")

    ens_plot = PlotEnsembleMaps(
        diagnostic_product=SEASONAL_PRODUCT,
        outputdir=cli.outputdir,
        loglevel=cli.loglevel,
        **ctx.names,
        **(ctx.ref_names() if dataset_ref is not None else {}),
    )

    all_plot_params = diag_config.get("plot_params", {})
    param_dict = all_plot_params.get(variable) or {}
    plot_params = {**all_plot_params.get("default", {}), **param_dict}
    common = {
        "var": variable,
        "save_format": ctx.save_format,
        "dpi": ctx.dpi,
        "proj": plot_params.get("projection", "robinson"),
        "proj_params": plot_params.get("projection_params", {}),
        "cbar_label": param_dict.get("cbar_label"),
        "cmap": param_dict.get("cmap"),
    }

    ens_plot.plot_seasonal(
        dataset=ens_seasonal.dataset_mean,
        vmin=param_dict.get("vmin_ensemble"),
        vmax=param_dict.get("vmax_ensemble"),
        data_name="ensemble_mean",
        title=TitleBuilder(diagnostic="Seasonal ensemble mean", variable=variable, model=ctx.model).generate(),
        **common,
    )
    ens_plot.plot_seasonal(
        dataset=ens_seasonal.dataset_std,
        vmin=param_dict.get("vmin_std_ensemble"),
        vmax=param_dict.get("vmax_std_ensemble"),
        data_name="ensemble_std",
        title=TitleBuilder(diagnostic="Seasonal ensemble STD", variable=variable, model=ctx.model).generate(),
        **common,
    )
    if dataset_ref is not None:
        ens_plot.plot_seasonal(
            dataset=ens_seasonal.dataset_mean,
            ref_dataset=dataset_ref,
            vmin=param_dict.get("vmin"),
            vmax=param_dict.get("vmax"),
            data_name="ensemble_bias_mean",
            title=TitleBuilder(diagnostic="Seasonal mean ensemble bias", variable=variable, model=ctx.model).generate(),
            **common,
        )

    cli.logger.info("Seasonal ensemble maps finished for variable '%s'.", variable)


def run_seasonal_cycles(ctx, diag_config):
    """
    Compute and plot the ensemble statistics of the seasonal cycles.

    Args:
        ctx (EnsembleContext): The ensemble to process.
        diag_config (dict): The ``seasonalcycles`` block of the configuration.
    """
    cli = ctx.cli
    diagnostic_name = diag_config.get("diagnostic_name", "seasonalcycles")

    for variable in diag_config.get("variables") or []:
        regions = diag_config.get("params", {}).get(variable, {}).get("regions") or ["global"]
        for region in regions:
            cli.logger.info("Ensemble seasonal cycle for variable: %s, region: %s", variable, region)
            path_args = (
                diagnostic_name,
                "seasonalcycles",
                variable,
                {"variable": variable, "freq": "monthly", "region": region},
            )

            dataset = ctx.load(ctx.paths(*path_args), variable)
            if dataset is None:
                continue

            ens = EnsembleProfiles(
                var=variable,
                dataset=dataset,
                diagnostic_product=SEASONAL_CYCLE_PRODUCT,
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
            )
            ens.run()

            ref = ctx.reference(*path_args, label="seasonal cycle")
            plot = PlotEnsembleProfiles(
                diagnostic_product=SEASONAL_CYCLE_PRODUCT,
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
                **(ctx.ref_names() if ref is not None else {}),
            )
            plot.plot(
                var=variable,
                mean=ens.dataset_mean,
                std=ens.dataset_std,
                members=dataset[variable],
                ref=ref,
                coord="month",
                data_name="seasonal_cycle",
                title=TitleBuilder(
                    diagnostic="Ensemble seasonal cycle", variable=variable, regions=region, model=ctx.model
                ).generate(),
                save_format=ctx.save_format,
                dpi=ctx.dpi,
            )

    cli.logger.info("Ensemble seasonal cycles completed!")


def run_lat_lon_profiles(ctx, diag_config):
    """
    Compute and plot the ensemble statistics of the longterm and seasonal latitude profiles.

    Args:
        ctx (EnsembleContext): The ensemble to process.
        diag_config (dict): The ``lat_lon_profiles`` block of the configuration.
    """
    cli = ctx.cli
    diagnostic_name = diag_config.get("diagnostic_name", "lat_lon_profiles")
    product = f"{diag_config.get('mean_type', 'zonal')}_profile"

    for entry in diag_config.get("variables") or []:
        variable = entry["name"]
        for region in entry.get("regions") or [None]:
            cli.logger.info("Ensemble %s for variable: %s, region: %s", product, variable, region)

            members = {}
            refs = {}
            for period, keys in PROFILE_PERIODS.items():
                extra_keys = {**keys, "variable": variable}
                if region is not None:
                    extra_keys["region"] = region
                path_args = (diagnostic_name, product, variable, extra_keys)
                members[period] = ctx.load(ctx.paths(*path_args), variable)
                refs[period] = ctx.reference(*path_args, label=f"{period} profile")

            members = {period: data for period, data in members.items() if data is not None}
            if set(members) != set(PROFILE_PERIODS):
                cli.logger.warning("Incomplete profiles for variable '%s', region '%s'. Skipping them!", variable, region)
                continue

            # The longterm and seasonal statistics are handled together along the period dimension
            dataset = xr.concat(
                [data.drop_vars("season", errors="ignore") for data in members.values()],
                dim=xr.DataArray(list(members), dims="period", name="period"),
            )

            ens = EnsembleProfiles(
                var=variable,
                dataset=dataset,
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
            )
            ens.run()

            has_ref = any(ref is not None for ref in refs.values())
            plot = PlotEnsembleProfiles(
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
                **(ctx.ref_names() if has_ref else {}),
            )
            common = {
                "var": variable,
                "mean": ens.dataset_mean,
                "std": ens.dataset_std,
                "members": dataset[variable],
                "save_format": ctx.save_format,
                "dpi": ctx.dpi,
            }
            longterm = {key: common[key].sel(period="longterm") for key in ("mean", "std", "members")}
            plot.plot(
                **{**common, **longterm},
                ref=refs["longterm"],
                data_name="longterm",
                title=TitleBuilder(
                    diagnostic="Ensemble zonal profile", variable=variable, regions=region, model=ctx.model
                ).generate(),
            )
            plot.plot(
                **common,
                ref=refs,
                panels={season: {"period": season} for season in SEASONS},
                data_name="seasonal",
                figure_size=(12, 8),
                title=TitleBuilder(
                    diagnostic="Seasonal ensemble zonal profile", variable=variable, regions=region, model=ctx.model
                ).generate(),
            )

    cli.logger.info("Ensemble latitude profiles completed!")


def run_zonal(ctx, diag_config):
    """
    Compute and plot the ensemble statistics of the zonal mean of the 3D annual climatologies.

    Args:
        ctx (EnsembleContext): The ensemble to process.
        diag_config (dict): The ``zonal`` block of the configuration.
    """
    cli = ctx.cli
    diagnostic_name = diag_config.get("diagnostic_name", "zonal")
    plot_params = diag_config.get("plot_params", {}).get("default", {})

    for variable in diag_config.get("variables") or []:
        cli.logger.info("Ensemble zonal mean for variable: %s", variable)
        path_args = (diagnostic_name, "annual_climatology", variable, {"variable": variable})

        dataset = ctx.load(ctx.paths(*path_args), variable)
        if dataset is None:
            continue

        ens = EnsembleZonal(
            var=variable,
            dataset=dataset[[variable]].mean("lon"),
            outputdir=cli.outputdir,
            loglevel=cli.loglevel,
            **ctx.names,
        )
        ens.run()

        plot = PlotEnsembleZonal(outputdir=cli.outputdir, loglevel=cli.loglevel, **ctx.names)
        common = {
            "var": variable,
            "save_format": ctx.save_format,
            "dpi": ctx.dpi,
            "ylim": tuple(plot_params.get("ylim", (100000, 100))),
            "ylabel": plot_params.get("ylabel", "Pressure (Pa)"),
        }
        plot.plot(
            dataset=ens.dataset_mean.compute(),
            data_name="ensemble_mean",
            title=TitleBuilder(diagnostic="Ensemble zonal mean", variable=variable, model=ctx.model).generate(),
            description=f"Ensemble zonal mean of {variable}",
            cmap=plot_params.get("cmap", "RdBu_r"),
            **common,
        )
        plot.plot(
            dataset=ens.dataset_std.compute(),
            data_name="ensemble_std",
            title=TitleBuilder(diagnostic="Ensemble zonal STD", variable=variable, model=ctx.model).generate(),
            description=f"Ensemble zonal STD of {variable}",
            cmap=plot_params.get("cmap_std", "viridis"),
            **common,
        )

        ref = ctx.reference(*path_args, label="zonal mean")
        if ref is not None:
            bias = (ens.dataset_mean - ref.mean("lon")).compute()
            limit = float(abs(bias).max())
            plot.plot(
                dataset=bias,
                data_name="ensemble_bias_mean",
                title=TitleBuilder(
                    diagnostic="Ensemble zonal mean bias",
                    variable=variable,
                    model=ctx.model,
                    comparison="relative to",
                    ref_model=ctx.ref_args["model_ref"],
                ).generate(),
                description=f"Ensemble zonal mean bias of {variable}",
                countour_levels=np.linspace(-limit, limit, 21),
                cmap="RdBu_r",
                **common,
            )

    cli.logger.info("Ensemble zonal means completed!")


def run_seaice_maps(ctx, diag_config):
    """
    Compute and plot the ensemble statistics of the monthly sea ice climatologies of each hemisphere.

    Args:
        ctx (EnsembleContext): The ensemble to process.
        diag_config (dict): The ``seaice_2d_bias`` block of the configuration.
    """
    cli = ctx.cli
    diagnostic_name = diag_config.get("diagnostic_name", "seaice")
    regions = diag_config.get("regions") or ["arctic", "antarctic"]
    months = diag_config.get("months") or [3, 9]
    month_names = [calendar.month_name[month] for month in months]

    for method in diag_config.get("methods") or []:
        method_params = diag_config.get("plot_params", {}).get(method, {})
        # The members were produced from their own source: match it instead of assuming the CLI one
        filenames = ctx.paths(diagnostic_name, "bias", method, {"method": method, "source": "*", "regions": "_".join(regions)})

        for region in regions:
            variable = f"sea_ice_{method}_{region}"
            cli.logger.info("Ensemble sea ice maps for %s", variable)

            dataset = ctx.load(filenames, variable)
            if dataset is None:
                continue

            ens = EnsembleMaps(
                var=variable,
                dataset=dataset,
                diagnostic_product=SEAICE_PRODUCT,
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
            )
            ens.run()

            ref_config = diag_config.get("references", {}).get(method, {}).get(region)
            ref = None
            ref_args = None
            if ref_config:
                ref_args = {
                    "catalog_ref": ref_config["catalog"],
                    "model_ref": ref_config["model"],
                    "exp_ref": ref_config["exp"],
                    "realization_ref": "r1",  # TODO
                }
                ref = ctx.reference(
                    diagnostic_name,
                    "bias",
                    variable,
                    {"method": method, "source": ref_config["source"], "regions": region},
                    label="sea ice map",
                    ref_args=ref_args,
                )

            plot = PlotEnsembleMaps(
                diagnostic_product=SEAICE_PRODUCT,
                region=region,
                outputdir=cli.outputdir,
                loglevel=cli.loglevel,
                **ctx.names,
                **(ctx.ref_names(ref_args) if ref is not None else {}),
            )
            region_params = diag_config.get("plot_params", {}).get(region, {})
            common = {
                "var": variable,
                "save_format": ctx.save_format,
                "dpi": ctx.dpi,
                "proj": region_params.get("projection", "orthographic"),
                "proj_params": region_params.get("projection_params", {}),
                "extent": region_params.get("extent"),
                "cyclic_lon": True,
                "dim": "month",
                "panels": months,
                "titles": month_names,
                "figsize": (10, 5),
                "cbar_label": method_params.get("cbar_label"),
            }

            plot.plot_seasonal(
                dataset=ens.dataset_mean,
                vmin=method_params.get("vmin_ensemble"),
                vmax=method_params.get("vmax_ensemble"),
                cmap=method_params.get("cmap_ensemble"),
                data_name="ensemble_mean",
                title=TitleBuilder(diagnostic="Ensemble mean", variable=variable, model=ctx.model).generate(),
                **common,
            )
            plot.plot_seasonal(
                dataset=ens.dataset_std,
                vmin=0,
                vmax=method_params.get("vmax_std_ensemble"),
                cmap=method_params.get("cmap_std_ensemble"),
                data_name="ensemble_std",
                title=TitleBuilder(diagnostic="Ensemble STD", variable=variable, model=ctx.model).generate(),
                **common,
            )
            if ref is not None:
                plot.plot_seasonal(
                    dataset=ens.dataset_mean,
                    ref_dataset=ref,
                    vmin=method_params.get("vmin"),
                    vmax=method_params.get("vmax"),
                    data_name="ensemble_bias_mean",
                    title=TitleBuilder(diagnostic="Ensemble mean bias", variable=variable, model=ctx.model).generate(),
                    **common,
                )

    cli.logger.info("Ensemble sea ice maps completed!")


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

    ctx = EnsembleContext(
        cli=cli,
        catalog=catalog,
        model=model,
        exp=exp,
        source=source,
        fixer=fixer,
        fixer_ref=fixer_ref,
        realization_list=extract_realizations_list(catalog=catalog, model=model, exp=exp, source=source),
        ref_args={"catalog_ref": catalog_ref, "model_ref": model_ref, "exp_ref": exp_ref, "realization_ref": "r1"}  # TODO
        if has_ref
        else None,
        save_format=save_format,
        dpi=dpi,
    )

    # Time series ensemble
    ts_diag_config = cli.config_dict["diagnostics"].get("timeseries", {})

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
    gb_diag_config = cli.config_dict["diagnostics"].get("biases", {})

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

                if params.get("seasons", False):
                    run_seasonal_maps(ctx, gb_diag_config, variable)

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

    runners = {
        "seasonalcycles": run_seasonal_cycles,
        "lat_lon_profiles": run_lat_lon_profiles,
        "zonal": run_zonal,
        "seaice_2d_bias": run_seaice_maps,
    }
    for key, runner in runners.items():
        diag_config = cli.config_dict["diagnostics"].get(key, {})
        if diag_config.get("run", False):
            try:
                runner(ctx, diag_config)
            except (NoDataError, KeyError, ValueError) as e:
                cli.logger.warning("Ensemble diagnostic '%s' failed: %s", key, e)

    cli.close_dask_cluster()


if __name__ == "__main__":
    main()
