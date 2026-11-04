import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics import EnsembleProfiles, PlotEnsembleProfiles
from aqua.diagnostics.ensemble.plot_ensemble_maps import SEASONS

pytestmark = [pytest.mark.diagnostics, pytest.mark.ensemble]

NAMES = {"catalog_list": "ci", "model_list": "IFS", "exp_list": "test", "source_list": "short"}


def members(coord, values):
    """Three members of a 1D profile, differing by 1 K."""
    data = np.stack([np.linspace(280, 290, len(values)) + member for member in range(3)])
    return xr.Dataset(
        {"2t": (("ensemble", coord), data, {"units": "K"})},
        coords={"ensemble": ["r1", "r2", "r3"], coord: values},
    )


@pytest.fixture
def profile():
    return members("lat", np.linspace(-85, 85, 18))


@pytest.fixture
def seasonal_cycle():
    return members("month", np.arange(1, 13))


def run(dataset, outputdir, **kwargs):
    ens = EnsembleProfiles(var="2t", dataset=dataset, outputdir=outputdir, **NAMES, **kwargs)
    ens.run()
    return ens


class TestEnsembleProfiles:
    def test_statistics(self, profile, tmp_path):
        ens = run(profile, str(tmp_path))

        assert ens.dataset_mean.dims == ("lat",)
        assert ens.dataset_mean.values == pytest.approx(profile["2t"].isel(ensemble=1).values)
        assert ens.dataset_std.values == pytest.approx(np.std([0, 1, 2]))

    def test_region_and_product_are_in_the_filename(self, seasonal_cycle, tmp_path):
        run(seasonal_cycle, str(tmp_path), diagnostic_product="EnsembleSeasonalCycles", region="europe")

        names = [path.name for path in tmp_path.rglob("*.nc")]
        assert len(names) == 2
        assert all("ensembleseasonalcycles" in name and "europe" in name for name in names)

    def test_plot_seasonal_cycle(self, seasonal_cycle, tmp_path):
        ens = run(seasonal_cycle, str(tmp_path))
        plot = PlotEnsembleProfiles(outputdir=str(tmp_path), ref_model="OBS", **NAMES)

        fig = plot.plot(
            var="2t",
            mean=ens.dataset_mean,
            std=ens.dataset_std,
            members=seasonal_cycle["2t"],
            ref=ens.dataset_mean + 1,
            coord="month",
            data_name="seasonal_cycle",
            save_format="png",
            dpi=50,
        )

        ax = fig.axes[0]
        assert ax.get_xticklabels()[0].get_text() == "Jan"
        # Three members, the ensemble mean and the reference
        assert len(ax.lines) == 5
        assert len(list(tmp_path.rglob("*.png"))) == 1

    def test_plot_panels(self, profile, tmp_path):
        dataset = xr.concat([profile] * 4, dim=xr.DataArray(SEASONS, dims="period", name="period"))
        ens = run(dataset, str(tmp_path))
        plot = PlotEnsembleProfiles(outputdir=str(tmp_path), **NAMES)

        fig = plot.plot(
            var="2t",
            mean=ens.dataset_mean,
            std=ens.dataset_std,
            members=dataset["2t"],
            panels={season: {"period": season} for season in SEASONS},
            data_name="seasonal",
            save_format="png",
            dpi=50,
        )

        assert [ax.get_title() for ax in fig.axes] == SEASONS

    def test_regional_profile_is_limited_to_the_region(self, profile, tmp_path):
        regional = profile.where(abs(profile.lat) < 20)
        ens = run(regional, str(tmp_path))
        plot = PlotEnsembleProfiles(outputdir=str(tmp_path), **NAMES)

        fig = plot.plot(var="2t", mean=ens.dataset_mean, std=ens.dataset_std, save_format="png", dpi=50)

        left, right = fig.axes[0].get_xlim()
        assert -20 < left and right < 20

    @pytest.mark.parametrize("empty", [True, False])
    def test_plot_without_data(self, profile, tmp_path, empty):
        plot = PlotEnsembleProfiles(outputdir=str(tmp_path), **NAMES)
        mean = profile["2t"].isel(ensemble=0) * np.nan if empty else None

        assert plot.plot(var="2t", mean=mean, std=mean) is None
