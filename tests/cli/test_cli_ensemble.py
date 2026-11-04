"""Tests of the orchestration in cli_ensemble, with synthetic members instead of the Reader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.ensemble import cli_ensemble
from aqua.diagnostics.ensemble.cli_ensemble import (
    EnsembleContext,
    run_lat_lon_profiles,
    run_seaice_maps,
    run_seasonal_cycles,
    run_seasonal_maps,
    run_zonal,
)

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

LAT = np.linspace(-85, 85, 18)
LON = np.linspace(0, 350, 36)


@pytest.fixture
def ctx(tmp_path):
    outputdir = tmp_path / "catalog" / "model" / "exp" / "ensemble" / "collection"
    return EnsembleContext(
        cli=MagicMock(outputdir=str(outputdir), loglevel="WARNING"),
        catalog="catalog",
        model="model",
        exp="exp",
        source="source",
        fixer=None,
        fixer_ref=None,
        realization_list=[2, 3, 4],
        ref_args=None,
        save_format="png",
        dpi=50,
    )


def touch(paths):
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


def ensemble(variable, dims, shape, **coords):
    data = np.arange(3).reshape((3,) + (1,) * len(shape)) + np.full(shape, 280.0)
    return xr.Dataset(
        {variable: (("ensemble", *dims), data, {"units": "K"})},
        coords={"ensemble": ["r2", "r3", "r4"], **coords},
    )


class TestLoad:
    def paths(self, ctx):
        return ctx.paths("collection", "product", "2t", {"variable": "2t"})

    def test_one_file_per_realization(self, ctx):
        paths = self.paths(ctx)

        assert [path.split("/")[-4] for path in paths] == ["r2", "r3", "r4"]

    def test_all_members(self, ctx):
        paths = self.paths(ctx)
        touch(paths)

        with patch.object(cli_ensemble, "reader_retrieve_and_merge") as merge:
            ctx.load(paths, "2t")

        assert merge.call_args.kwargs["filenames"] == paths
        assert merge.call_args.kwargs["realizations"] == [2, 3, 4]
        ctx.cli.logger.warning.assert_not_called()

    def test_missing_member_is_reported_and_realizations_stay_aligned(self, ctx):
        paths = self.paths(ctx)
        touch([paths[0], paths[2]])

        with patch.object(cli_ensemble, "reader_retrieve_and_merge") as merge:
            ctx.load(paths, "2t")

        assert merge.call_args.kwargs["filenames"] == [paths[0], paths[2]]
        assert merge.call_args.kwargs["realizations"] == [2, 4]
        assert ctx.cli.logger.warning.call_args.args[2] == [3]

    def test_too_few_members(self, ctx):
        paths = self.paths(ctx)
        touch(paths[:1])

        with patch.object(cli_ensemble, "reader_retrieve_and_merge") as merge:
            assert ctx.load(paths, "2t") is None

        merge.assert_not_called()

    def test_glob_pattern(self, ctx):
        paths = self.paths(ctx)
        touch([path.replace("*", "x") for path in paths])
        pattern = [path.replace(".nc", "*.nc") for path in paths]

        with patch.object(cli_ensemble, "reader_retrieve_and_merge") as merge:
            ctx.load(pattern, "2t")

        assert merge.call_args.kwargs["filenames"] == paths


class TestRunners:
    def test_lat_lon_profiles_stack_the_periods(self, ctx):
        members = ensemble("2t", ("lat",), (LAT.size,), lat=LAT)
        config = {"diagnostic_name": "collection", "variables": [{"name": "2t", "regions": [None]}]}

        with (
            patch.object(EnsembleContext, "load", return_value=members),
            patch.object(cli_ensemble, "EnsembleProfiles") as ens,
        ):
            with patch.object(cli_ensemble, "PlotEnsembleProfiles"):
                run_lat_lon_profiles(ctx, config)

        dataset = ens.call_args.kwargs["dataset"]
        assert dataset.period.values.tolist() == ["longterm", "DJF", "MAM", "JJA", "SON"]

    def test_lat_lon_profiles_skip_incomplete_periods(self, ctx):
        config = {"variables": [{"name": "2t", "regions": [None]}]}

        with patch.object(EnsembleContext, "load", return_value=None), patch.object(cli_ensemble, "EnsembleProfiles") as ens:
            run_lat_lon_profiles(ctx, config)

        ens.assert_not_called()

    def test_zonal_averages_over_longitude(self, ctx):
        members = ensemble(
            "t", ("plev", "lat", "lon"), (3, LAT.size, LON.size), plev=[100000.0, 50000.0, 10000.0], lat=LAT, lon=LON
        )
        config = {"diagnostic_name": "collection", "variables": ["t"]}

        with patch.object(EnsembleContext, "load", return_value=members), patch.object(cli_ensemble, "EnsembleZonal") as ens:
            with patch.object(cli_ensemble, "PlotEnsembleZonal"):
                run_zonal(ctx, config)

        assert "lon" not in ens.call_args.kwargs["dataset"].dims

    def test_seaice_runs_each_hemisphere_and_matches_any_source(self, ctx):
        calls = []

        def load(self, filenames, variable):
            calls.append((filenames[0], variable))
            return None

        config = {"diagnostic_name": "collection", "methods": ["fraction"], "regions": ["arctic", "antarctic"]}
        with patch.object(EnsembleContext, "load", load):
            run_seaice_maps(ctx, config)

        assert [variable for _, variable in calls] == ["sea_ice_fraction_arctic", "sea_ice_fraction_antarctic"]
        assert all(".fraction.*.arctic_antarctic.nc" in filename for filename, _ in calls)

    def test_seaice_uses_the_reference_of_each_method_and_hemisphere(self, ctx):
        members = ensemble(
            "sea_ice_fraction_arctic",
            ("month", "lat", "lon"),
            (12, LAT.size, LON.size),
            month=np.arange(1, 13),
            lat=LAT,
            lon=LON,
        )
        reference = {"catalog": "obs", "model": "OSI-SAF", "exp": "osi-saf-ssmis", "source": "nh-monthly"}
        config = {
            "diagnostic_name": "collection",
            "methods": ["fraction"],
            "regions": ["arctic"],
            "months": [3, 9],
            "references": {"fraction": {"arctic": reference}},
        }

        with (
            patch.object(EnsembleContext, "load", return_value=members),
            patch.object(EnsembleContext, "reference", return_value=MagicMock()) as load_reference,
            patch.object(cli_ensemble, "EnsembleMaps"),
            patch.object(cli_ensemble, "PlotEnsembleMaps") as plot,
        ):
            run_seaice_maps(ctx, config)

        assert load_reference.call_args.args[3] == {"method": "fraction", "source": "nh-monthly", "regions": "arctic"}
        assert load_reference.call_args.kwargs["ref_args"]["model_ref"] == "OSI-SAF"
        assert plot.call_args.kwargs["ref_model"] == "OSI-SAF"

        calls = plot.return_value.plot_seasonal.call_args_list
        assert [call.kwargs["data_name"] for call in calls] == ["ensemble_mean", "ensemble_std", "ensemble_bias_mean"]
        assert all(call.kwargs["dim"] == "month" and call.kwargs["panels"] == [3, 9] for call in calls)
        assert calls[0].kwargs["titles"] == ["March", "September"]

    def test_seasonal_cycles_run_each_region(self, ctx):
        members = ensemble("2t", ("month",), (12,), month=np.arange(1, 13))
        config = {
            "diagnostic_name": "collection",
            "variables": ["2t"],
            "params": {"2t": {"regions": ["global", "europe"]}},
        }

        with (
            patch.object(EnsembleContext, "load", return_value=members),
            patch.object(cli_ensemble, "EnsembleProfiles") as ens,
            patch.object(cli_ensemble, "PlotEnsembleProfiles") as plot,
        ):
            run_seasonal_cycles(ctx, config)

        assert [call.kwargs["region"] for call in ens.call_args_list] == ["global", "europe"]
        assert all(call.kwargs["diagnostic_product"] == "EnsembleSeasonalCycles" for call in ens.call_args_list)
        assert plot.return_value.plot.call_args.kwargs["coord"] == "month"
        assert plot.return_value.plot.call_args.kwargs["ref"] is None

    def test_seasonal_cycles_skip_missing_data(self, ctx):
        config = {"variables": ["2t"], "params": {"2t": {"regions": ["global"]}}}

        with patch.object(EnsembleContext, "load", return_value=None), patch.object(cli_ensemble, "EnsembleProfiles") as ens:
            run_seasonal_cycles(ctx, config)

        ens.assert_not_called()

    @pytest.mark.parametrize(
        "reference, products",
        [
            (None, ["ensemble_mean", "ensemble_std"]),
            (MagicMock(), ["ensemble_mean", "ensemble_std", "ensemble_bias_mean"]),
        ],
    )
    def test_seasonal_maps_plot_the_bias_only_with_a_reference(self, ctx, reference, products):
        with (
            patch.object(EnsembleContext, "load", return_value=MagicMock()),
            patch.object(EnsembleContext, "reference", return_value=reference),
            patch.object(cli_ensemble, "EnsembleMaps") as ens,
            patch.object(cli_ensemble, "PlotEnsembleMaps") as plot,
        ):
            run_seasonal_maps(ctx, {"diagnostic_name": "collection"}, "2t")

        assert ens.call_args.kwargs["diagnostic_product"] == "EnsembleSeasonalMaps"
        calls = plot.return_value.plot_seasonal.call_args_list
        assert [call.kwargs["data_name"] for call in calls] == products

    def test_seasonal_maps_skip_missing_data(self, ctx):
        with patch.object(EnsembleContext, "load", return_value=None), patch.object(cli_ensemble, "EnsembleMaps") as ens:
            run_seasonal_maps(ctx, {}, "2t")

        ens.assert_not_called()

    def test_zonal_bias_is_the_difference_of_the_zonal_means(self, ctx):
        plev = [100000.0, 50000.0]
        members = ensemble("t", ("plev", "lat", "lon"), (2, LAT.size, LON.size), plev=plev, lat=LAT, lon=LON)
        zonal_mean = members["t"].mean(("ensemble", "lon"))
        reference = members["t"].isel(ensemble=0, drop=True) - 2.0
        ctx.ref_args = {"catalog_ref": "obs", "model_ref": "ECMWF", "exp_ref": "era5", "realization_ref": "r1"}

        with (
            patch.object(EnsembleContext, "load", return_value=members),
            patch.object(EnsembleContext, "reference", return_value=reference),
            patch.object(cli_ensemble, "EnsembleZonal") as ens,
            patch.object(cli_ensemble, "PlotEnsembleZonal") as plot,
        ):
            ens.return_value.dataset_mean = zonal_mean
            ens.return_value.dataset_std = zonal_mean
            run_zonal(ctx, {"diagnostic_name": "collection", "variables": ["t"]})

        calls = plot.return_value.plot.call_args_list
        assert [call.kwargs["data_name"] for call in calls] == ["ensemble_mean", "ensemble_std", "ensemble_bias_mean"]
        # The reference is 2 K below the first member, which is 1 K below the ensemble mean
        assert calls[2].kwargs["dataset"].values == pytest.approx(3.0)
        assert calls[2].kwargs["countour_levels"][0] == pytest.approx(-3.0)


class TestReference:
    def test_no_reference_configured(self, ctx):
        assert ctx.reference("collection", "product", "2t", {"variable": "2t"}, label="test") is None

    def test_reference_files_are_generated_from_the_reference_names(self, ctx):
        ctx.ref_args = {"catalog_ref": "obs", "model_ref": "ERA5", "exp_ref": "era5", "realization_ref": "r1"}

        with patch.object(cli_ensemble, "load_reference") as load_reference:
            ctx.reference("collection", "product", "2t", {"variable": "2t"}, label="test")

        filenames = load_reference.call_args.args[1]
        assert all(".obs.ERA5.era5.r1." in filename for filename in filenames)
