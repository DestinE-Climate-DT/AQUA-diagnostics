"""Tests for the OceanHeatBudget tool.

The `ci` catalog has no dataset with both the atmospheric fluxes and the ocean heat content, so the computation
is tested on a synthetic dataset with a known answer; the `ci` catalog is only used for the failure case.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.ocean_heat_budget import OceanHeatBudget, PlotOceanHeatBudget
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

DATASET = {"catalog": "ci", "model": "ERA5", "exp": "era5-hpz3", "source": "monthly"}

# Net flux = -50 - 40 + 150 - 10 - 1e-5 * 3.34e5 over the ocean; the heat content grows at 40 W m**-2
NET_FLUX = 50 - 1e-5 * 3.34e5
TENDENCY = 40.0
MONTHS = 24


@pytest.fixture
def diag():
    """An OceanHeatBudget whose data and reader are replaced by a synthetic ocean with one land column."""
    time = pd.date_range("2000-01-01", periods=MONTHS, freq="MS")
    coords = {"time": time, "lat": [-10.0, 10.0], "lon": [0.0, 10.0, 20.0]}
    shape = (MONTHS, 2, 3)

    def field(value):
        return xr.DataArray(np.full(shape, value), coords=coords, dims=("time", "lat", "lon"))

    land = xr.DataArray([False, False, True], coords={"lon": coords["lon"]})  # the last column is land
    seconds = xr.DataArray((time - time[0]) / np.timedelta64(1, "s"), coords={"time": time})
    data = xr.Dataset(
        {
            "slhtf": field(-50.0).where(~land, -500.0),  # a land value that a mask by zero would not remove from the mean
            "snlwrf": field(-40.0),
            "snswrf": field(150.0),
            "ishf": field(-10.0),
            "tsrwe": field(1e-5),
            "tos": field(280.0).where(~land),
            "hc700m": (1e10 + TENDENCY * seconds + 0 * field(0.0)).where(~land),
        },
        attrs={"AQUA_startdate": "2000-01-01", "AQUA_enddate": "2001-12-01"},
    )
    # Like retrieve(): the period is a dataset attribute, the Reader provenance lives on the variables
    data["hc700m"].attrs.update(
        long_name="Heat content in the upper 700 m", AQUA_catalog="ci", AQUA_model="ERA5", AQUA_exp="era5-hpz3"
    )
    diag = OceanHeatBudget(**DATASET, loglevel=LOGLEVEL)
    diag.data = data
    diag.reader = SimpleNamespace(fldmean=lambda da: da.mean(["lat", "lon"], skipna=True))
    return diag


def test_budget_terms(diag):
    """Fluxes are averaged over the ocean only, the tendency is the forward difference of the heat content."""
    diag.compute_result()
    result = diag.result
    np.testing.assert_allclose(result["net_surface_flux"], NET_FLUX, rtol=APPROX_REL)
    np.testing.assert_allclose(result["hc_tendency"], TENDENCY, rtol=APPROX_REL)
    assert result["hc_tendency"].attrs["units"] == "W m**-2"
    assert "700 m" in result["hc_tendency"].attrs["long_name"]
    assert result.attrs["AQUA_model"] == "ERA5"


def test_last_step_has_no_tendency(diag):
    """The forward difference of N steps gives N-1 values, and both terms are on the same times."""
    diag.compute_result()
    assert diag.result.sizes["time"] == MONTHS - 1
    assert not np.isnan(diag.result["hc_tendency"]).any()


def test_missing_variable_raises():
    """A source without the budget variables fails with NoDataError, it is the CLI that decides to skip."""
    diag = OceanHeatBudget(**DATASET, loglevel=LOGLEVEL)
    with pytest.raises((NoDataError, KeyError, ValueError)):
        diag.run(save_netcdf=False)


def test_save_load_round_trip(diag, tmp_path):
    """A fresh instance that never retrieved anything gets back the same result (plot-only runs)."""
    diag.compute_result()
    diag.save_netcdf(outputdir=str(tmp_path))
    fresh = OceanHeatBudget(**DATASET, loglevel=LOGLEVEL)
    fresh.load(outputdir=str(tmp_path))
    assert fresh.result is not None
    np.testing.assert_allclose(fresh.result["net_surface_flux"], diag.result["net_surface_flux"], rtol=APPROX_REL)
    assert fresh.result.attrs["AQUA_model"] == "ERA5"


def test_load_checks_the_period(diag, tmp_path):
    """With dates, only a file computed over that period is loaded: this is what plot-only runs ask for."""
    diag.compute_result()
    diag.save_netcdf(outputdir=str(tmp_path))

    same = OceanHeatBudget(**DATASET, startdate="2000-01-01", enddate="2001-12-01", loglevel=LOGLEVEL)
    same.load(outputdir=str(tmp_path))
    assert same.result is not None

    other = OceanHeatBudget(**DATASET, startdate="1991-01-01", enddate="1992-12-31", loglevel=LOGLEVEL)
    other.load(outputdir=str(tmp_path))
    assert other.result is None


def test_load_missing_file(tmp_path):
    """A missing file leaves the result untouched."""
    diag = OceanHeatBudget(**DATASET, loglevel=LOGLEVEL)
    diag.load(outputdir=str(tmp_path))
    assert diag.result is None


def test_plot(diag, tmp_path):
    """The plot class works from the result alone and writes the budget and the difference figures."""
    diag.compute_result()
    plot = PlotOceanHeatBudget(data=diag.result, outputdir=str(tmp_path), loglevel=LOGLEVEL)
    assert "ERA5" in plot.set_title()
    assert plot.set_description().endswith(".")
    np.testing.assert_allclose(plot.difference, NET_FLUX - TENDENCY, rtol=APPROX_REL)
    plot.plot(save_format="png", dpi=DPI)
    pngs = sorted(tmp_path.rglob("*.png"))
    assert len(pngs) == 2
    for png in pngs:
        assert_nonempty(png)
