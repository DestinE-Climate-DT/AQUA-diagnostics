import os
import pytest
from aqua.diagnostics.ocean_stratification.stratification import Stratification
from conftest import APPROX_REL, LOGLEVEL
from aqua.diagnostics.ocean_stratification import PlotStratification, PlotMLD

loglevel = LOGLEVEL
approx_rel = APPROX_REL*10

# pytestmark groups tests that run sequentially on the same worker to avoid conflicts
pytestmark = [
    pytest.mark.diagnostics,
    pytest.mark.xdist_group(name="dask_operations")
]

def test_stratification():
    """Test the stratification class."""
    # Create an instance of the stratification class
    strat = Stratification(catalog='ci', model='FESOM',
                          exp='hpz3', source='monthly-3d',
                          regrid='r100', loglevel=loglevel)

    strat.run(
        dim_mean=["lat","lon"],
        var=['thetao', 'so'],
        climatology='January',
        region='ls',
        mld=True,
        )
    assert strat is not None, "strat instance should not be None"
    assert strat.data["mld"].values == pytest.approx(2.5000076,rel=approx_rel)
    assert strat.data["rho"].isel(level=1).values == pytest.approx(-3.34617212e+08,rel=approx_rel)

    ps = PlotStratification(data=strat.data[['thetao', 'so', 'rho']],
                            obs=strat.data[['thetao', 'so', 'rho']]*1.2,
                            loglevel=loglevel,
                            )
    ps.plot_stratification()

    assert os.path.exists("png/ocean_stratification.stratification.ci.FESOM.hpz3.r1.labrador_sea.png"), "Expected output file not found: png/ocean_stratification.stratification.ci.FESOM.hpz3.r1.labrador_sea.png"

    strat.run(
        var=['thetao', 'so'],
        climatology='January',
        region='ls',
        mld=True,
        )
    
    ps = PlotMLD(data=strat.data[['mld']],
              obs=strat.data[['mld']]*2, loglevel=loglevel
              )
    # ps.plot_stratification()
    ps.plot_mld()
    assert os.path.exists("png/ocean_stratification.mld.ci.FESOM.hpz3.r1.labrador_sea.png"), "Expected output file not found: png/ocean_stratification.mld.ci.FESOM.hpz3.r1.labrador_sea.png"
