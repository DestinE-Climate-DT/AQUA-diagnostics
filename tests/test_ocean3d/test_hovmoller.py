import pytest
from aqua.diagnostics.ocean_drift import Hovmoller
from aqua.diagnostics.ocean_drift.plot_hovmoller import PlotHovmoller
from conftest import LOGLEVEL
import os

loglevel = LOGLEVEL


@pytest.mark.diagnostics
def test_hovmoller(tmp_path):
    """Test the Hovmoller class."""
    # Create an instance of the Hovmoller class
    hovmoller = Hovmoller(catalog='ci', model='FESOM',
                          exp='hpz3', source='monthly-3d',
                          startdate='1990-01-01', enddate='1990-03-31',
                          regrid='r200', 
                          loglevel=loglevel)

    hovmoller.run(anomaly_ref="t0",
                    outputdir=tmp_path,
                  region='sss')
    assert hovmoller is not None, "Hovmoller instance should not be None"

    assert hovmoller.processed_data_list[0].thetao.isel(level=1, time=1).values == pytest.approx(22.2086629652034, abs=1e-4)
    assert hovmoller.processed_data_list[0].so.isel(level=1, time=1).values == pytest.approx(36.57638045014168, abs=1e-4)

    assert hovmoller.processed_data_list[1].thetao.isel(level=1, time=1).values == pytest.approx(-0.6924832430820729, abs=1e-4)
    assert hovmoller.processed_data_list[1].so.isel(level=1, time=1).values == pytest.approx(0.02545398252818387, abs=1e-4)

    assert hovmoller.processed_data_list[2].thetao.isel(level=1, time=1).values == pytest.approx(-2.07305172, abs=1e-4)
    assert hovmoller.processed_data_list[2].so.isel(level=1, time=1).values == pytest.approx(2.35781597, abs=1e-4)
    

    hov_plot = PlotHovmoller(data=hovmoller.processed_data_list, 
                            loglevel=loglevel,
                            outputdir=tmp_path
                            )
    hov_plot.plot_hovmoller(save_png=True, save_pdf=True)

    assert os.path.exists(f"{tmp_path}/png/oceandrift.hovmoller.ci.FESOM.hpz3.r1.sargasso_sea.png"), f"Expected output file not found: {tmp_path}/png/oceandrift.hovmoller.ci.FESOM.hpz3.r1.sargasso_sea.png"
