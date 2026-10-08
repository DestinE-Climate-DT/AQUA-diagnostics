import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from aqua.core.exceptions import NotEnoughDataError
from aqua.diagnostics.trends import PlotTrends, Trends
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL


@pytest.fixture(scope="module")
def ocean_trend(tmp_path_factory):
    """Retrieve the 3D ocean fields once for the generic workflow tests."""
    tmp_path = tmp_path_factory.mktemp("general_trends")
    trend = Trends(catalog="ci", model="FESOM", exp="hpz3", source="monthly-3d", regrid="r100", loglevel=loglevel)
    trend.run(var=["thetao", "so"], region="go", outputdir=tmp_path)
    return trend, trend.trend_coef, tmp_path


@pytest.mark.diagnostics
def test_trends(ocean_trend):
    """Validate pointwise slopes against a direct elapsed-year covariance calculation."""
    trend, result, tmp_path = ocean_trend
    assert isinstance(result, xr.Dataset)
    assert result.attrs["AQUA_region"] == "Global Ocean"
    indexers = {"depth": 1, "lat": slice(None, None, 30), "lon": slice(None, None, 60)}
    years = ((trend.data.time - trend.data.time[0]) / np.timedelta64(1, "D")).values / 365.25
    for var in ["thetao", "so"]:
        values = trend.data[var].isel(indexers).transpose("time", "lat", "lon").values
        valid = np.isfinite(values)
        count = valid.sum(axis=0)
        mean_time = (years[:, None, None] * valid).sum(axis=0) / np.maximum(count, 1)
        mean_value = np.nansum(values, axis=0) / np.maximum(count, 1)
        centered_time = years[:, None, None] - mean_time
        numerator = np.nansum(centered_time * (values - mean_value), axis=0)
        denominator = (centered_time**2 * valid).sum(axis=0)
        expected = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=count > 1)
        assert np.isfinite(expected).any()
        np.testing.assert_allclose(result[var].isel(indexers), expected, rtol=1e-7, atol=1e-10)
    path = trend.save_netcdf(result, outputdir=tmp_path)
    with xr.open_dataset(path) as saved:
        assert "_coeffs" in saved.attrs["history"]
        assert "365.25" in saved.attrs["history"]
        assert saved.attrs["product"] == "Calculated trend coefficients"
    try:
        PlotTrends(result, outputdir=tmp_path).plot_multilevel(levels=[10, 100], save_format="png", dpi=50)
        PlotTrends(result.mean("lon"), outputdir=tmp_path).plot_zonal(save_format="png", dpi=50)
        for product in ["multilevel_trend", "zonal_mean"]:
            assert (tmp_path / "png" / f"trends.{product}.ci.FESOM.hpz3.r1.global_ocean.png").stat().st_size > 0
    finally:
        plt.close("all")


@pytest.mark.diagnostics
def test_trends_region_dim_mean(ocean_trend):
    """A trend averaged over a dimension must be restricted to the region, not fall back to the global domain."""
    trend, _, _ = ocean_trend

    zonal_global = trend.compute_trend(dim_mean="lon")
    zonal_region = trend.compute_trend(region="io", dim_mean="lon")

    assert zonal_region.attrs["AQUA_region"] == "Indian Ocean"
    assert zonal_region.attrs["AQUA_dim_mean"] == "lon"
    assert "lon" not in zonal_region.dims
    assert float(zonal_global["thetao"].mean()) != pytest.approx(float(zonal_region["thetao"].mean()), rel=approx_rel)


@pytest.mark.diagnostics
def test_surface_trend_outputs(tmp_path):
    trend = Trends(catalog="ci", model="ERA5", exp="era5-hpz3", source="monthly", regrid="r100", loglevel=loglevel)
    result = trend.run(var="2t", outputdir=tmp_path)
    assert set(result["2t"].dims) == {"lat", "lon"}
    assert result["2t"].attrs["units"].endswith("/year")
    PlotTrends(result, outputdir=tmp_path).plot_trend(save_format="png", dpi=50)
    assert len(list((tmp_path / "png").glob("*.png"))) == 1
    assert len(list((tmp_path / "netcdf").glob("*.nc"))) == 1


@pytest.mark.diagnostics
def test_surface_trend_requires_a_year(tmp_path):
    """The NEMO fixture has six months and must fail the annual coverage check."""
    trend = Trends(catalog="ci", model="NEMO", exp="test-eORCA1", source="long-2d", loglevel=loglevel)
    with pytest.raises(NotEnoughDataError, match="at least 12 months required, only 6 found"):
        trend.run(var="tos", outputdir=tmp_path, reader_kwargs={"areas": False})
    assert trend.trend_coef is None
    assert not list(tmp_path.rglob("*.nc"))


@pytest.mark.diagnostics
def test_native_coordinates_and_reusable_regions(icon_test_r2b0_short_reader, icon_test_r2b0_short_data, tmp_path):
    """Native ICON fields carry geographic coordinates over their cell dimension."""
    trend = Trends(catalog="ci", model="ICON", exp="test-r2b0", source="short", loglevel=loglevel)
    trend.reader = icon_test_r2b0_short_reader
    # This tests fitting already retrieved fields; retrieve/run enforce annual coverage.
    trend.data = icon_test_r2b0_short_data[["t"]]
    # drop=True needs eager coordinate masks; keep the temperature field lazy.
    trend.data = trend.data.assign_coords({coord: trend.data[coord].compute() for coord in ["lat", "lon"]})
    original = trend.data.copy(deep=True)
    global_trend = trend.compute_trend()
    for coord in ["lat", "lon"]:
        assert coord not in trend.data.dims
        xr.testing.assert_identical(global_trend[coord], trend.data[coord])
    first = trend.compute_trend(lon_limits=[160, -160], lat_limits=[-30, 30])
    assert first.lon.size > 0
    wrapped_lon = (first.lon + 180) % 360 - 180
    assert bool((abs(wrapped_lon) >= 160).all())
    second = trend.compute_trend(lon_limits=[20, 100], lat_limits=[-30, 30])
    assert first.attrs["AQUA_region"] != second.attrs["AQUA_region"]
    assert trend.save_netcdf(first, outputdir=tmp_path) != trend.save_netcdf(second, outputdir=tmp_path)
    regional = trend.select_region(global_trend, region="io")["data"]
    direct = trend.compute_trend(region="io")
    xr.testing.assert_allclose(regional, direct)
    assert global_trend.attrs["history"] in regional.attrs["history"]
    assert regional.attrs["AQUA_region"] == "Indian Ocean"
    assert global_trend.attrs["AQUA_region"] == "global"
    xr.testing.assert_identical(trend.data, original)
