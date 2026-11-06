"""Tests for the Budget tool.

The `ci` catalog has no dataset with both the atmospheric fluxes and the ocean heat content, so the computation
is tested on a synthetic dataset with a known answer; the `ci` catalog is only used for the failure case.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.budget import Budget, PlotBudget
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

DATASET = {"catalog": "ci", "model": "ERA5", "exp": "era5-hpz3", "source": "monthly"}
MONTHS = 24

# Ocean: net flux = -50 - 40 + 150 - 10 - 1e-5 * 3.34e5 over the ocean; the heat content grows at 40 W m**-2
OCEAN = {
    "first": "slhtf+snlwrf+snswrf+ishf-3.34e5*tsrwe",
    "second": "hc700m",
    "tendency": True,
    "mask": "tos",
    "units": "W m**-2",
}
NET_FLUX = 50 - 1e-5 * 3.34e5
TENDENCY = 40.0
# Atmosphere: TOA net = 240 - 239, no tendency
ATMOSPHERE = {"first": "tnlwrf+tnswrf", "second": "slhtf+snlwrf+snswrf+ishf", "long_names": ["TOA", "Surface"]}


@pytest.fixture
def diag():
    """A Budget whose data and reader are replaced by a synthetic globe with one land column."""
    time = pd.date_range("2000-01-01", periods=MONTHS, freq="MS")
    coords = {"time": time, "lat": [-10.0, 10.0], "lon": [0.0, 10.0, 20.0]}

    def field(value, units="W m**-2"):
        return xr.DataArray(np.full((MONTHS, 2, 3), value), coords=coords, dims=("time", "lat", "lon"), attrs={"units": units})

    land = xr.DataArray([False, False, True], coords={"lon": coords["lon"]})  # the last column is land
    seconds = xr.DataArray((time - time[0]) / np.timedelta64(1, "s"), coords={"time": time})
    data = xr.Dataset(
        {
            "slhtf": field(-50.0).where(~land, -500.0),  # a land value that a mask by zero would not remove from the mean
            "snlwrf": field(-40.0),
            "snswrf": field(150.0),
            "ishf": field(-10.0),
            "tsrwe": field(1e-5, "kg m**-2 s**-1"),
            "tos": field(280.0, "K").where(~land),
            "hc700m": (1e10 + TENDENCY * seconds + field(0.0)).where(~land).assign_attrs(units="J m**-2"),
            "tnlwrf": field(-239.0),
            "tnswrf": field(240.0),
        },
        attrs={"AQUA_startdate": "2000-01-01", "AQUA_enddate": "2001-12-01"},
    )
    # Like retrieve(): the period is a dataset attribute, the Reader provenance lives on the variables
    for var in data.data_vars:
        data[var].attrs.update(AQUA_catalog="ci", AQUA_model="ERA5", AQUA_exp="era5-hpz3")
    diag = Budget(**DATASET, loglevel=LOGLEVEL)
    diag.data = data
    diag.reader = SimpleNamespace(fldmean=lambda da: da.mean(["lat", "lon"], skipna=True))
    return diag


def test_ocean_budget(diag):
    """Masked means of a formula and of the tendency of a variable, converted to the same units."""
    diag.compute_result(**OCEAN)
    result = diag.result
    np.testing.assert_allclose(result["first"], NET_FLUX, rtol=APPROX_REL)
    np.testing.assert_allclose(result["second"], TENDENCY, rtol=APPROX_REL)
    assert result["second"].attrs["units"] == "W m**-2"
    assert result["second"].attrs["long_name"] == "Time derivative of hc700m"
    assert result.attrs["AQUA_model"] == "ERA5"
    # The forward difference of N steps gives N-1 values, and both quantities are on the same times
    assert result.sizes["time"] == MONTHS - 1


def test_atmosphere_budget(diag):
    """Without tendency both quantities are formulae on every time step, with the configured names."""
    diag.compute_result(**ATMOSPHERE)
    result = diag.result
    assert result.sizes["time"] == MONTHS
    np.testing.assert_allclose(result["first"], 1.0, rtol=APPROX_REL)
    # No mask: the land value of slhtf enters the mean
    np.testing.assert_allclose(result["second"], (2 * -50 - 500) / 3 + 100, rtol=APPROX_REL)
    assert result["first"].attrs["long_name"] == "TOA"


def test_missing_variable_raises():
    """A source without the budget variables fails, it is the CLI that decides to skip."""
    diag = Budget(**DATASET, loglevel=LOGLEVEL)
    with pytest.raises((NoDataError, KeyError, ValueError)):
        diag.run(name="ocean", save_netcdf=False, **OCEAN)


def test_save_load_round_trip(diag, tmp_path):
    """A fresh instance that never retrieved anything gets back the same result (plot-only runs)."""
    diag.compute_result(**OCEAN)
    diag.save_netcdf(name="ocean", outputdir=str(tmp_path))
    fresh = Budget(**DATASET, loglevel=LOGLEVEL)
    fresh.load(name="ocean", outputdir=str(tmp_path))
    assert fresh.result is not None
    np.testing.assert_allclose(fresh.result["first"], diag.result["first"], rtol=APPROX_REL)
    assert fresh.result.attrs["AQUA_model"] == "ERA5"
    # Another budget name is another file
    other = Budget(**DATASET, loglevel=LOGLEVEL)
    other.load(name="atmosphere", outputdir=str(tmp_path))
    assert other.result is None


def test_load_checks_the_period(diag, tmp_path):
    """With dates, only a file computed over that period is loaded: this is what plot-only runs ask for."""
    diag.compute_result(**OCEAN)
    diag.save_netcdf(name="ocean", outputdir=str(tmp_path))

    same = Budget(**DATASET, startdate="2000-01-01", enddate="2001-12-01", loglevel=LOGLEVEL)
    same.load(name="ocean", outputdir=str(tmp_path))
    assert same.result is not None

    other = Budget(**DATASET, startdate="1991-01-01", enddate="1992-12-31", loglevel=LOGLEVEL)
    other.load(name="ocean", outputdir=str(tmp_path))
    assert other.result is None


def test_plot(diag, tmp_path):
    """The plot class works from the result alone and writes the two figures, showing the expected value."""
    diag.compute_result(**OCEAN)
    plot = PlotBudget(data=diag.result, name="ocean", expected=0.5, outputdir=str(tmp_path), loglevel=LOGLEVEL)
    assert "ERA5" in plot.set_title()
    assert "0.5" in plot.set_description(difference=True)
    np.testing.assert_allclose(plot.difference, NET_FLUX - TENDENCY, rtol=APPROX_REL)
    figures = plot.plot(save_format="png", dpi=DPI)
    _, ax = figures[1]
    assert 0.5 in [line.get_ydata()[0] for line in ax.get_lines()]
    pngs = sorted(tmp_path.rglob("*.png"))
    assert len(pngs) == 2
    for png in pngs:
        assert_nonempty(png)
        assert ".ocean." in png.name
