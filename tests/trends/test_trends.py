import pytest

from aqua.diagnostics.trends import Trends
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL


@pytest.mark.diagnostics
def test_trends():
    """Test the trends class."""
    # Create an instance of the trends class
    trend = Trends(catalog="ci", model="FESOM", exp="hpz3", source="monthly-3d", regrid="r100", loglevel=loglevel)

    trend.run(
        var=["thetao", "so"],
        region="go",
    )
    assert trend is not None, "trend instance should not be None"
    assert trend.trend_coef["Global Ocean"]["thetao"].isel(depth=1).mean("lat").mean("lon").values == pytest.approx(
        -0.06603967, rel=approx_rel
    )
    assert trend.trend_coef["Global Ocean"]["so"].isel(depth=1).mean("lat").mean("lon").values == pytest.approx(
        0.02622599, rel=approx_rel
    )
