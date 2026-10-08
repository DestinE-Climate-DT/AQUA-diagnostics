"""Small numerical and plotting regressions independent of catalog availability."""

from unittest.mock import Mock

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from aqua.core.reader import Reader
from aqua.diagnostics import PlotTrends as PublicPlotTrends
from aqua.diagnostics import Trends as PublicTrends
from aqua.diagnostics.ocean_trends import PlotTrends as OceanPlotTrends
from aqua.diagnostics.ocean_trends import Trends as OceanTrends
from aqua.diagnostics.ocean_trends.multiple_maps import plot_maps
from aqua.diagnostics.trends import PlotTrends, Trends

pytestmark = pytest.mark.diagnostics


@pytest.fixture
def diagnostic():
    trend = Trends(catalog="ci", model="model", exp="exp", source="monthly")
    # Numerical tests supply in-memory fields; catalog retrieval is tested separately.
    trend.reader = Mock(spec=Reader)
    return trend


@pytest.mark.parametrize("frequency", ["MS", "YS", "D", "irregular"])
@pytest.mark.parametrize("lazy", [False, True])
def test_elapsed_year_slope_and_history(diagnostic, frequency, lazy):
    times = pd.date_range("2000-01-01", periods=24, freq="MS" if frequency == "irregular" else frequency)
    if frequency == "irregular":
        times = times.delete([1, 4, 9])
    years = (times - times[0]).total_seconds().to_numpy() / (365.25 * 86400)
    data = xr.Dataset(
        {
            "temperature": (("time", "ncells"), 280 + 2 * years[:, None] + np.array([0, 1, 2])),
            "salinity": (("time", "ncells"), 35 - 0.5 * years[:, None] + np.array([0, 1, 2])),
        },
        coords={"time": times, "lat": ("ncells", [-30, 0, 30]), "lon": ("ncells", [0, 90, 180])},
        attrs={"history": "source history", "AQUA_model": "model"},
    )
    data.temperature.attrs = {"units": "K", "long_name": "Temperature"}
    data.salinity.attrs = {"units": "psu"}
    data.temperature[3, 1] = np.nan
    diagnostic.data = data.chunk({"time": 6}) if lazy else data
    original = diagnostic.data.copy(deep=True)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args[0])):
        result = diagnostic.compute_trend()
    assert not tasks, "Computing the coefficients must not execute the Dask graph"
    assert bool(result.chunks) == lazy
    np.testing.assert_allclose(result.temperature, 2, rtol=1e-10)
    np.testing.assert_allclose(result.salinity, -0.5, rtol=1e-10)
    assert result.temperature.attrs["units"] == "K/year"
    assert result.temperature.attrs["long_name"] == "Temperature"
    assert "source history" in result.attrs["history"]
    assert "_coeffs" in result.attrs["history"]
    assert "365.25" in result.attrs["history"]
    assert result.attrs["AQUA_model"] == "model"
    assert result.attrs["AQUA_trend_year_days"] == 365.25
    assert result.aqua.instance is diagnostic.reader
    assert diagnostic.trend_coef is result
    xr.testing.assert_identical(result.lat, data.lat)
    xr.testing.assert_identical(result.lon, data.lon)
    xr.testing.assert_identical(diagnostic.data, original)


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "proleptic_gregorian"])
def test_cftime_fit_uses_elapsed_days(diagnostic, calendar):
    # This validates the fit, independently of Diagnostic.retrieve's date handling.
    times = xr.date_range("2000-01-01", periods=24, freq="MS", calendar=calendar, use_cftime=True)
    years = np.array([(time - times[0]).total_seconds() / (365.25 * 86400) for time in times])
    diagnostic.data = xr.Dataset({"temperature": ("time", 280 + 2 * years)}, coords={"time": times})
    assert float(diagnostic.compute_trend().temperature) == pytest.approx(2, rel=1e-10)


@pytest.fixture
def ocean_coefficients():
    depth = [5, 100, 500]
    lat = [-30, 0, 30]
    lon = [-120, -60, 0, 60, 120]
    values = np.arange(45, dtype=float).reshape(3, 3, 5) / 100 - 0.1
    data = xr.Dataset(
        {"thetao": (("depth", "lat", "lon"), values), "so": (("depth", "lat", "lon"), -values)},
        coords={"depth": depth, "lat": lat, "lon": lon},
        attrs={
            "AQUA_catalog": "ci",
            "AQUA_model": "model",
            "AQUA_exp": "exp",
            "AQUA_realization": "r1",
            "AQUA_region": "global",
            "AQUA_startdate": "2000-03-01",
            "AQUA_enddate": "2001-06-30",
            "history": "source history",
        },
    )
    data.thetao.attrs = {"units": "K/year", "long_name": "Ocean temperature"}
    data.so.attrs = {"units": "psu/year", "long_name": "Salinity"}
    return data


def test_depth_preparation_preserves_inputs(ocean_coefficients, tmp_path):
    data = ocean_coefficients.copy(deep=True)
    data["so"] = data.so.where(data.depth != 100)
    original = data.copy(deep=True)
    requested = [0, 100, 1000, 500]
    plotter = OceanPlotTrends(data, outputdir=tmp_path)
    plotter.levels = requested
    plotter.set_data_list()
    assert requested == [0, 100, 1000, 500]
    assert plotter.levels == [0, 100, 500]
    assert len(plotter.data_list) == 6
    xr.testing.assert_allclose(plotter.data_list[0], data.thetao.isel(depth=0))
    assert bool(plotter.data_list[3].isnull().all())
    xr.testing.assert_identical(data, original)
    plotter.levels = [1000]
    with pytest.raises(ValueError, match="No valid"):
        plotter.set_data_list()


@pytest.mark.parametrize("plot_class", [PlotTrends, OceanPlotTrends])
@pytest.mark.parametrize("variables, levels", [(["thetao"], [100]), (["thetao", "so"], [0, 100, 500, 1000])])
def test_ocean_products_and_colour_limits(plot_class, variables, levels, ocean_coefficients, tmp_path, monkeypatch):
    import aqua.diagnostics.ocean_trends.plot_trends as ocean_module

    captured = []
    real_plot_maps = ocean_module.plot_maps

    def capture_maps(**kwargs):
        fig = real_plot_maps(**kwargs)
        captured.append((kwargs, fig))
        return fig

    monkeypatch.setattr(ocean_module, "plot_maps", capture_maps)
    data = ocean_coefficients[variables].chunk({"depth": 1})
    original = data.copy(deep=True)
    requested = list(levels)
    plotter = plot_class(data, outputdir=tmp_path)
    plotter.plot_multilevel(levels=levels, cbar_limits={"thetao": {"vmin": -1, "vmax": 2}}, save_format="png", dpi=40)
    args, figure = captured[0]
    assert args["col_vmin"][0] == -1
    assert args["col_vmax"][0] == 2
    assert len(figure.axes) == len(variables) * (len([level for level in levels if level <= 500]) + 1)
    assert figure.axes[0].collections[0].get_clim() == (-1, 2)
    assert levels == requested
    assert figure.axes[-1].get_position().y1 < figure.axes[-len(variables) - 1].get_position().y0
    assert data.chunks
    xr.testing.assert_identical(data, original)
    with pytest.raises(ValueError, match="longitude mean"):
        plotter.plot_zonal(save_format="png", dpi=40)
    zonal = data.mean("lon")
    plot_class(zonal, outputdir=tmp_path).plot_zonal(save_format="png", dpi=40)
    for product in ["multilevel_trend", "zonal_mean"]:
        assert (tmp_path / "png" / f"trends.{product}.ci.model.exp.r1.global.png").stat().st_size > 0
    assert not plt.get_fignums()


def test_map_projection_and_partial_limits(ocean_coefficients):
    data = ocean_coefficients.isel(depth=0)
    proj = ccrs.PlateCarree(central_longitude=30)
    fig = plot_maps(
        [data.thetao, data.so],
        nrows=1,
        ncols=2,
        proj=proj,
        col_vmin=[-1, None],
        col_vmax=[2, None],
        sym=False,
        cyclic_lon=False,
    )
    assert fig.axes[0].projection == proj
    assert fig.axes[0].collections[0].get_clim() == (-1, 2)
    assert all(np.isfinite(fig.axes[1].collections[0].get_clim()))
    plt.close(fig)


def test_map_description_and_saved_metadata(ocean_coefficients, diagnostic, tmp_path):
    data = ocean_coefficients.isel(depth=0, drop=True)
    plotter = PlotTrends(data, outputdir=tmp_path)
    assert "2000-03" in plotter.set_description("thetao")
    assert "2001-06" in plotter.set_description("thetao")
    plotter.plot_trend(var="thetao", save_format="png", dpi=40)
    assert (tmp_path / "png" / "trends.map_trend.ci.model.exp.r1.thetao.global.png").stat().st_size > 0
    diagnostic.trend_coef = data
    path = diagnostic.save_netcdf(outputdir=tmp_path)
    with xr.open_dataset(path) as restored:
        assert "source history" in restored.attrs["history"]
        assert restored.attrs["AQUA_region"] == "global"
        assert restored.attrs["AQUA_startdate"] == "2000-03-01"
        assert restored.thetao.attrs["units"] == "K/year"


def test_public_ocean_imports_remain_compatible():
    assert PublicTrends is OceanTrends
    assert PublicPlotTrends is OceanPlotTrends


@pytest.mark.parametrize("surface_only", [False, True])
def test_lazy_trend_netcdf_roundtrip(diagnostic, ocean_coefficients, surface_only, tmp_path):
    """Computation stays lazy and writing preserves numerical results and their provenance."""
    expected = ocean_coefficients.isel(depth=0, drop=True) if surface_only else ocean_coefficients
    times = xr.DataArray(pd.date_range("2000-03-01", periods=24, freq="MS"), dims="time")
    years = (times - times[0]) / np.timedelta64(1, "D") / 365.25
    diagnostic.data = (expected * years).assign_coords(time=times).chunk({"time": 6})
    diagnostic.data.thetao.attrs["units"] = "K"
    diagnostic.data.so.attrs["units"] = "psu"
    diagnostic.data.attrs["AQUA_enddate"] = "2002-02-01"
    result = diagnostic.compute_trend()
    assert result.chunks
    path = diagnostic.save_netcdf(outputdir=tmp_path)
    with xr.open_dataset(path) as saved:
        xr.testing.assert_allclose(saved, expected)
        assert "source history" in saved.attrs["history"]
        assert "_coeffs" in saved.attrs["history"]
        assert "365.25" in saved.attrs["history"]
        assert saved.attrs["product"] == "Calculated trend coefficients"
        assert saved.attrs["AQUA_startdate"] == "2000-03-01"
        assert saved.attrs["AQUA_enddate"] == "2002-02-01"
        assert saved.thetao.attrs["units"] == "K/year"
        assert saved.so.attrs["units"] == "psu/year"
    assert diagnostic.data.chunks
