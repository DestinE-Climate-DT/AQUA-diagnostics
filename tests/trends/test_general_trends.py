import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from aqua.core.reader import Reader
from aqua.diagnostics.trends import PlotTrends, Trends
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL


@pytest.mark.diagnostics
def test_trends():
    """Test the trends class."""
    # Create an instance of the trends class
    trend = Trends(catalog="ci", model="FESOM", exp="hpz3", source="monthly-3d", regrid="r100", loglevel=loglevel)

    trend.run(var=["thetao", "so"], region="go")

    assert isinstance(trend.trend_coef, xr.Dataset), "trend_coef should be an xarray Dataset"
    assert trend.trend_coef.attrs["AQUA_region"] == "Global Ocean"
    assert trend.trend_coef["thetao"].isel(depth=1).mean("lat").mean("lon").values == pytest.approx(
        -0.06603967, rel=approx_rel
    )
    assert trend.trend_coef["so"].isel(depth=1).mean("lat").mean("lon").values == pytest.approx(0.02622599, rel=approx_rel)


@pytest.mark.diagnostics
def test_trends_region_dim_mean():
    """A trend averaged over a dimension must be restricted to the region, not fall back to the global domain."""
    trend = Trends(catalog="ci", model="FESOM", exp="hpz3", source="monthly-3d", regrid="r100", loglevel=loglevel)
    trend.retrieve(var="thetao")

    zonal_global = trend.compute_trend(dim_mean="lon")
    zonal_region = trend.compute_trend(region="io", dim_mean="lon")

    assert zonal_region.attrs["AQUA_region"] == "Indian Ocean"
    assert zonal_region.attrs["AQUA_dim_mean"] == "lon"
    assert "lon" not in zonal_region.dims
    assert float(zonal_global["thetao"].mean()) != pytest.approx(float(zonal_region["thetao"].mean()), rel=approx_rel)


@pytest.mark.diagnostics
def test_lazy_trend_metadata_and_save(icon_test_r2b0_short_reader, icon_test_r2b0_short_data, tmp_path, monkeypatch):
    """Fit without execution, then preserve the known slope and provenance through writing."""
    source = icon_test_r2b0_short_data
    trend = Trends(catalog="ci", model="ICON", exp="test-r2b0", source="short", loglevel=loglevel)
    trend.reader = icon_test_r2b0_short_reader
    years = (source.time - source.time[0]) / np.timedelta64(1, "D") / 365.25
    # Use the existing Reader grid and timestamps with a known slope of two units/year.
    trend.data = source.assign(t=xr.zeros_like(source.t) + 280 + 2 * years)
    trend.data.t.attrs = source.t.attrs.copy()
    assert trend.data.chunks
    monkeypatch.setattr(Reader, "instance", None)

    # Check the fit trend is lazy, without executing the Dask graph.
    with Callback(pretask=lambda *args: pytest.fail("Trend fitting executed the Dask graph")):
        result = trend.compute_trend()

    assert result.chunks
    assert result.aqua.instance is trend.reader
    assert trend.trend_coef is None  # Only run() stores the result on the diagnostic.
    trend.save_netcdf(data=result, outputdir=tmp_path)
    (path,) = (tmp_path / "netcdf").glob("*.nc")
    with xr.open_dataset(path) as saved:
        np.testing.assert_allclose(saved.t, 2, rtol=1e-5)
        assert saved.t.attrs["units"] == f"{source.t.attrs['units']}/year"
        assert source.attrs["history"] in saved.attrs["history"]
        assert "polynomial of order 1" in saved.attrs["history"]
        assert "365.25 elapsed days" in saved.attrs["history"]
        assert saved.attrs["product"] == "Calculated trend coefficients"
        assert saved.t.attrs["AQUA_model"] == source.t.attrs["AQUA_model"]
    assert trend.data.chunks


@pytest.mark.diagnostics
@pytest.mark.parametrize("enddate", ["2001-06-30", None])
def test_trend_description_period(icon_test_r2b0_short_data, tmp_path, enddate):
    """Descriptions include the analysis months only when both dates are supplied."""
    data = icon_test_r2b0_short_data.isel(time=0, drop=True).assign_attrs(AQUA_startdate="2000-03-01", AQUA_enddate=enddate)
    description = PlotTrends(data, outputdir=tmp_path).set_description("t")
    assert (" between " in description) == (enddate is not None)
    if enddate is not None:
        assert "between 2000-03 and 2001-06" in description
