from pathlib import Path

import pytest

from aqua.diagnostics.ocean_trends import PlotTrends, Trends
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10

# --- Constants ---
EXPECTED_THETAO_TREND = -0.06603967
EXPECTED_SO_TREND = 0.02622599
PLOT_STEM = "trends.{product}.ci.FESOM.hpz3.r1.global_ocean"

pytestmark = [pytest.mark.diagnostics]

TRENDS_CONFIG = {
    "init": {
        "catalog": "ci",
        "model": "FESOM",
        "exp": "hpz3",
        "source": "monthly-3d",
        "regrid": "r100",
        "loglevel": loglevel,
    },
    "run": {
        "var": ["thetao", "so"],
        "region": "go",
    },
    "plot": {
        "save_format": ["png"],
        "products": ["multilevel_trend", "zonal_mean"],
    },
}


# --- Fixtures ---


@pytest.fixture(scope="session")
def trends_config():
    return TRENDS_CONFIG


@pytest.fixture(scope="module")
def trends_result(tmp_path_factory, trends_config):
    """Run the Trends pipeline once for this module."""
    tmp_path = tmp_path_factory.mktemp("trends")
    trend = Trends(**trends_config["init"])
    trend.run(**trends_config["run"], outputdir=tmp_path)
    return trend, tmp_path


@pytest.fixture(scope="module")
def trends_plots(trends_result, trends_config):
    """Run both plot types once. Multilevel must use full maps; zonal uses lon-mean."""
    trend, tmp_path = trends_result
    save_format = trends_config["plot"]["save_format"]
    PlotTrends(data=trend.trend_coef, outputdir=tmp_path, loglevel=loglevel).plot_multilevel(save_format=save_format)
    PlotTrends(data=trend.trend_coef.mean("lon"), outputdir=tmp_path, loglevel=loglevel).plot_zonal(save_format=save_format)
    return tmp_path


def _assert_nonempty(path):
    assert path.is_file(), f"File not found: {path}"
    assert path.stat().st_size > 0


# --- Tests ---


@pytest.mark.parametrize(
    "var, expected",
    [
        ("thetao", EXPECTED_THETAO_TREND),
        ("so", EXPECTED_SO_TREND),
    ],
)
def test_trend_coef(trends_result, var, expected):
    trend, _ = trends_result
    actual = trend.trend_coef[var].isel({trend.vert_coord: 1}).mean("lat").mean("lon").values
    assert actual == pytest.approx(expected, rel=approx_rel)


def test_netcdf_output(trends_result):
    _, tmp_path = trends_result
    nc = Path(tmp_path) / "netcdf" / f"{PLOT_STEM.format(product='trend')}.nc"
    _assert_nonempty(nc)


@pytest.mark.parametrize("product", TRENDS_CONFIG["plot"]["products"])
@pytest.mark.parametrize("ext", TRENDS_CONFIG["plot"]["save_format"])
def test_plot_output(trends_plots, product, ext):
    path = Path(trends_plots) / ext / f"{PLOT_STEM.format(product=product)}.{ext}"
    _assert_nonempty(path)
