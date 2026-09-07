from pathlib import Path

import pytest

from aqua.diagnostics.ocean_drift import Hovmoller, PlotHovmoller
from tests.shared_constants import LOGLEVEL

loglevel = LOGLEVEL

# --- Constants ---
EXPECTED_THETAO = [22.2086629652034, -0.6924832430820729, -2.07305172]
EXPECTED_SO = [36.57638045014168, 0.02545398252818387, 2.35781597]
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


def test_processed_data_types(hovmoller_result):
    hov, _ = hovmoller_result
    types = [ds.attrs["AQUA_ocean_drift_type"] for ds in hov.processed_data_list]
    assert types == EXPECTED_DRIFT_TYPES


@pytest.mark.parametrize("dataset_idx, expected", enumerate(EXPECTED_THETAO))
def test_thetao_values(hovmoller_result, dataset_idx, expected):
    hov, _ = hovmoller_result
    actual = hov.processed_data_list[dataset_idx].thetao.isel({hov.vert_coord: 1, "time": 1}).values
    assert actual == pytest.approx(expected, abs=1e-4), f"thetao mismatch at dataset {dataset_idx}"


@pytest.mark.parametrize("dataset_idx, expected", enumerate(EXPECTED_SO))
def test_so_values(hovmoller_result, dataset_idx, expected):
    hov, _ = hovmoller_result
    actual = hov.processed_data_list[dataset_idx].so.isel({hov.vert_coord: 1, "time": 1}).values
    assert actual == pytest.approx(expected, abs=1e-4), f"so mismatch at dataset {dataset_idx}"


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