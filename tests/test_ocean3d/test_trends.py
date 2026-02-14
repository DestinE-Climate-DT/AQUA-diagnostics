import pytest
from aqua.diagnostics.ocean_trends import Trends
from conftest import APPROX_REL, LOGLEVEL
from aqua.diagnostics.ocean_trends.plot_trends import PlotTrends
import os

loglevel = LOGLEVEL
approx_rel = APPROX_REL*10

@pytest.mark.diagnostics
def test_trends():
    """Test the trends class."""
    # Create an instance of the trends class
    trend = Trends(catalog='ci', model='FESOM',
                          exp='hpz3', source='monthly-3d',
                          regrid='r100', loglevel=loglevel)
    
    trend.run(
        # dim_mean="lat",
              var=['thetao', 'so'],
              region='go'
              )
    print(trend.trend_coef)
    assert trend is not None, "trend instance should not be None"
    assert trend.trend_coef["thetao"].isel(level=1).mean('lat').mean('lon').values == pytest.approx(-0.06603967,rel=approx_rel)
    assert trend.trend_coef["so"].isel(level=1).mean('lat').mean('lon').values == pytest.approx(0.02622599,rel=approx_rel)

    pt = PlotTrends(data = trend.trend_coef, loglevel=loglevel)
    pt.plot_multilevel()
    assert os.path.exists("png/trends.multilevel_trend.ci.FESOM.hpz3.r1.global_ocean.png"), "Expected output file not found: png/trends.multilevel_trend.ci.FESOM.hpz3.r1.global_ocean.png"

    pt = PlotTrends(data = trend.trend_coef.mean('lon'), loglevel=loglevel)
    pt.plot_zonal()
    assert os.path.exists("png/trends.zonal_mean.ci.FESOM.hpz3.r1.global_ocean.png"), "Expected output file not found: png/trends.zonal_mean.ci.FESOM.hpz3.r1.global_ocean.png"
