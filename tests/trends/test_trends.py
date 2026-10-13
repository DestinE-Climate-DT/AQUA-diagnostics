import pytest
import xarray as xr

from aqua.diagnostics.trends import Trends
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
