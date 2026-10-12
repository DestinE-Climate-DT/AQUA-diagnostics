from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.ocean_drift import Hovmoller, PlotHovmoller
from aqua.diagnostics.ocean_drift.plot_hovmoller import DEFAULT_CBAR_LIMITS
from aqua.diagnostics.ocean_drift.util import set_year_ticks
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10
dpi = DPI

# --- Constants ---
EXPECTED_FULL = {"thetao": 295.3586629652034, "so": 36.57638045014168}
EXPECTED_DRIFT_TYPES = ["full", "anom_t0", "std_anom_t0"]
PLOT_STEM = "oceandrift.{product}.ci.FESOM.hpz3.r1.sargasso_sea"
EXPECTED_NTIME = 15  # crosses a year boundary, so the yearly split/concat is tested

pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="ocean_drift")]

HOVMOLLER_CONFIG = {
    "init": {
        "catalog": "ci",
        "model": "FESOM",
        "exp": "hpz3",
        "source": "monthly-3d",
        "startdate": "1990-01-01",
        "enddate": "1991-03-31",
        "regrid": "r200",
        "loglevel": loglevel,
    },
    "run": {
        "anomaly_ref": "t0",
        # The unknown region must be skipped without stopping the valid ones
        "regions": ["sss", "ao", "not_a_region"],
    },
    "plot": {
        "save_format": ["png", "pdf", "svg"],
        "products": ["hovmoller", "timeseries"],
    },
}

# --- Fixtures ---


@pytest.fixture(scope="session")
def hovmoller_config():
    return HOVMOLLER_CONFIG


@pytest.fixture(scope="module")
def hovmoller_result(tmp_path_factory, hovmoller_config):
    """Run the Hovmoller pipeline once for this module."""
    tmp_path = tmp_path_factory.mktemp("hovmoller")
    hov = Hovmoller(**hovmoller_config["init"])
    hov.run(**hovmoller_config["run"], outputdir=tmp_path)
    return hov, tmp_path


@pytest.fixture(scope="module")
def hovmoller_plot(hovmoller_result, hovmoller_config):
    """Run both plot types once."""
    hov, tmp_path = hovmoller_result
    save_format = hovmoller_config["plot"]["save_format"]
    hov_plot = PlotHovmoller(data=hov.processed_data["sss"], loglevel=loglevel, outputdir=tmp_path)
    hov_plot.plot_hovmoller(save_format=save_format, dpi=dpi)
    hov_plot.plot_timeseries(save_format=save_format, dpi=dpi)
    return tmp_path


# --- Tests ---


def _by_drift_type(hov):
    """Index the processed datasets by drift type, so tests do not rely on list order."""
    return {ds.attrs["AQUA_ocean_drift_type"]: ds for ds in hov.processed_data["sss"]}


def test_processed_data_types(hovmoller_result):
    hov, _ = hovmoller_result
    types = [ds.attrs["AQUA_ocean_drift_type"] for ds in hov.processed_data["sss"]]
    assert types == EXPECTED_DRIFT_TYPES


def test_multiple_regions_in_one_run(hovmoller_result):
    """One run() stores every valid region, each with the three drift products over the whole period."""
    hov, _ = hovmoller_result
    assert set(hov.processed_data) == {"sss", "ao"}
    for region in ("sss", "ao"):
        types = [ds.attrs["AQUA_ocean_drift_type"] for ds in hov.processed_data[region]]
        assert types == EXPECTED_DRIFT_TYPES
        assert all(ds.sizes["time"] == EXPECTED_NTIME for ds in hov.processed_data[region])


@pytest.mark.parametrize("var, expected", sorted(EXPECTED_FULL.items()))
def test_full_values(hovmoller_result, var, expected):
    """Anchor the untransformed field, the one value the derived checks build on."""
    hov, _ = hovmoller_result
    full = _by_drift_type(hov)["full"]
    actual = full[var].isel({hov.vert_coord: 1, "time": 1}).values
    assert actual == pytest.approx(expected, rel=approx_rel)


def test_anomaly_is_referenced_to_first_timestep(hovmoller_result):
    """anom_t0 is the field minus its own first timestep, hence exactly zero there."""
    hov, _ = hovmoller_result
    data = _by_drift_type(hov)
    full, anom = data["full"], data["anom_t0"]

    xr.testing.assert_allclose(anom, full - full.isel(time=0))
    assert float(abs(anom.isel(time=0).to_dataarray()).max()) == 0.0


def test_standardised_anomaly_is_scaled_by_its_own_std(hovmoller_result):
    """std_anom_t0 is anom_t0 divided by its temporal standard deviation, hence dimensionless."""
    hov, _ = hovmoller_result
    data = _by_drift_type(hov)
    anom, std_anom = data["anom_t0"], data["std_anom_t0"]

    xr.testing.assert_allclose(std_anom, anom / anom.std("time"))
    assert all(std_anom[var].attrs["units"] == "1" for var in std_anom.data_vars)


def test_panel_settings(hovmoller_result, tmp_path):
    """Units, colormaps and color limits of the panels (rows: full, anom_t0, std_anom_t0; columns: thetao, so)."""
    hov, _ = hovmoller_result
    hov_plot = PlotHovmoller(data=hov.processed_data["sss"], loglevel=loglevel, outputdir=tmp_path)
    hov_plot.set_units_labels()
    hov_plot.set_cmaps()
    hov_plot.set_cbar_limits({"thetao": {"anom_t0": (-0.5, 2.0)}})

    assert hov_plot.units_labels == ["K", "g kg$^{-1}$", "K", "g kg$^{-1}$", None, None]
    assert hov_plot.cmap == ["plasma", "YlGnBu"] + ["RdBu_r", "BrBG"] * 2
    expected = [DEFAULT_CBAR_LIMITS[var][drift] for drift in EXPECTED_DRIFT_TYPES for var in ("thetao", "so")]
    expected[2] = (-2.0, 2.0)  # the asymmetric anomaly limits are centered on zero
    assert list(zip(hov_plot.vmin, hov_plot.vmax)) == expected
    # Uneven levels for the anomalies in physical units only
    assert [levels is not None for levels in hov_plot.cbar_levels] == [False, False, True, True, False, False]


def test_timeseries_levels(hovmoller_result, tmp_path):
    """Selecting the timeseries levels, even a single one, leaves the full profiles to the Hovmoller plot."""
    hov, _ = hovmoller_result
    data = hov.processed_data["sss"]
    hov_plot = PlotHovmoller(data=data, loglevel=loglevel, outputdir=tmp_path)
    hov_plot.levels = [1500]
    hov_plot.set_levels()
    hov_plot.set_data_for_levels()
    hov_plot.set_line_plot_colours()

    assert hov_plot.timeseries_data[0].sizes[hov.vert_coord] == 1
    assert len(hov_plot.line_plot_colours) == 1
    assert hov_plot.data is data


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("1990-01", "1991-03", ["1990", "1991"]),
        ("1950-01", "2049-12", ["1960", "1980", "2000", "2020", "2040"]),
        ("1990-03", "1990-11", ["1990"]),  # no year start within the span
    ],
)
def test_year_ticks(start, end, expected):
    """The time axis shows round years only, at most six of them."""
    time = np.arange(start, end, dtype="datetime64[M]").astype("datetime64[ns]")
    _, ax = plt.subplots()
    ax.plot(time, np.zeros(time.size))
    ax.set_xlim(time[0], time[-1])
    set_year_ticks(ax)

    low, high = ax.get_xlim()
    ticks = [tick for tick in ax.get_xticks() if low <= tick <= high]
    assert ax.xaxis.get_major_formatter().format_ticks(ticks) == expected


@pytest.mark.parametrize("drift_type", EXPECTED_DRIFT_TYPES)
def test_netcdf_output(hovmoller_result, drift_type):
    _, tmp_path = hovmoller_result
    nc = Path(tmp_path) / "netcdf" / f"{PLOT_STEM.format(product='hovmoller')}.{drift_type}.nc"
    assert_nonempty(nc)


@pytest.mark.parametrize("product", HOVMOLLER_CONFIG["plot"]["products"])
@pytest.mark.parametrize("ext", HOVMOLLER_CONFIG["plot"]["save_format"])
def test_plot_output(hovmoller_plot, product, ext):
    path = Path(hovmoller_plot) / ext / f"{PLOT_STEM.format(product=product)}.{ext}"
    assert_nonempty(path)
