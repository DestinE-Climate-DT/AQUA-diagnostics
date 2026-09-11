from pathlib import Path

import pytest
import xarray as xr

from aqua.diagnostics.ocean_drift import Hovmoller, PlotHovmoller
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10

# --- Constants ---
EXPECTED_FULL = {"thetao": 22.2086629652034, "so": 36.57638045014168}
EXPECTED_DRIFT_TYPES = ["full", "anom_t0", "std_anom_t0"]
PLOT_STEM = "oceandrift.{product}.ci.FESOM.hpz3.r1.sargasso_sea"

pytestmark = [pytest.mark.diagnostics]

HOVMOLLER_CONFIG = {
    "init": {
        "catalog": "ci",
        "model": "FESOM",
        "exp": "hpz3",
        "source": "monthly-3d",
        "startdate": "1990-01-01",
        "enddate": "1990-03-31",
        "regrid": "r200",
        "loglevel": loglevel,
    },
    "run": {
        "anomaly_ref": "t0",
        "region": "sss",
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
    """Run both plot types once. Hovmoller must run before timeseries."""
    hov, tmp_path = hovmoller_result
    save_format = hovmoller_config["plot"]["save_format"]
    hov_plot = PlotHovmoller(data=hov.processed_data_list, loglevel=loglevel, outputdir=tmp_path)
    hov_plot.plot_hovmoller(save_format=save_format)
    hov_plot.plot_timeseries(save_format=save_format)
    return tmp_path


def _assert_nonempty(path):
    assert path.is_file(), f"File not found: {path}"
    assert path.stat().st_size > 0


# --- Tests ---


def _by_drift_type(hov):
    """Index the processed datasets by drift type, so tests do not rely on list order."""
    return {ds.attrs["AQUA_ocean_drift_type"]: ds for ds in hov.processed_data_list}


def test_processed_data_types(hovmoller_result):
    hov, _ = hovmoller_result
    types = [ds.attrs["AQUA_ocean_drift_type"] for ds in hov.processed_data_list]
    assert types == EXPECTED_DRIFT_TYPES


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
    """std_anom_t0 is anom_t0 divided by its temporal standard deviation."""
    hov, _ = hovmoller_result
    data = _by_drift_type(hov)
    anom, std_anom = data["anom_t0"], data["std_anom_t0"]

    xr.testing.assert_allclose(std_anom, anom / anom.std("time"))


@pytest.mark.parametrize("drift_type", EXPECTED_DRIFT_TYPES)
def test_netcdf_output(hovmoller_result, drift_type):
    _, tmp_path = hovmoller_result
    nc = Path(tmp_path) / "netcdf" / f"{PLOT_STEM.format(product='hovmoller')}.{drift_type}.nc"
    _assert_nonempty(nc)


@pytest.mark.parametrize("product", HOVMOLLER_CONFIG["plot"]["products"])
@pytest.mark.parametrize("ext", HOVMOLLER_CONFIG["plot"]["save_format"])
def test_plot_output(hovmoller_plot, product, ext):
    path = Path(hovmoller_plot) / ext / f"{PLOT_STEM.format(product=product)}.{ext}"
    _assert_nonempty(path)
