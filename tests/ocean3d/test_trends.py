from pathlib import Path

import pytest
import xarray as xr

from aqua.diagnostics.ocean_trends import PlotTrends, Trends
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10
dpi = DPI

# --- Constants ---
EXPECTED_THETAO_TREND = -0.06603967
EXPECTED_SO_TREND = 0.02622599
PLOT_STEM = "trends.{product}.ci.FESOM.hpz3.r1.global_ocean"

pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="ocean_trends")]

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
        "levels": [10, 100, 500, 1000],
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
    # Copy the levels: PlotTrends.set_data_list pops all-NaN levels from the list it receives.
    PlotTrends(data=trend.trend_coef, outputdir=tmp_path, loglevel=loglevel).plot_multilevel(
        levels=list(trends_config["plot"]["levels"]), save_format=save_format, dpi=dpi
    )
    PlotTrends(data=trend.trend_coef.mean("lon"), outputdir=tmp_path, loglevel=loglevel).plot_zonal(
        save_format=save_format, dpi=dpi
    )
    return tmp_path


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
    assert_nonempty(nc)


@pytest.mark.parametrize("product", TRENDS_CONFIG["plot"]["products"])
@pytest.mark.parametrize("ext", TRENDS_CONFIG["plot"]["save_format"])
def test_plot_output(trends_plots, product, ext):
    path = Path(trends_plots) / ext / f"{PLOT_STEM.format(product=product)}.{ext}"
    assert_nonempty(path)


def test_load_roundtrip(trends_result, trends_config):
    """A run that only plots finds on disk exactly the trend a previous run computed.

    The consumer never retrieves: catalog, realization and region name are rebuilt without data access.
    """
    producer, tmp_path = trends_result
    consumer = Trends(model="FESOM", exp="hpz3", source="monthly-3d", loglevel=loglevel)
    consumer.load(outputdir=tmp_path, region=trends_config["run"]["region"])

    assert consumer.region == producer.region
    xr.testing.assert_allclose(consumer.trend_coef, producer.trend_coef)

    # The attributes PlotTrends reads survived the round trip
    for attr in ["AQUA_catalog", "AQUA_model", "AQUA_exp"]:
        assert consumer.trend_coef["thetao"].attrs[attr] == producer.trend_coef["thetao"].attrs[attr]
    assert consumer.trend_coef.attrs["AQUA_region"] == producer.region

    # The loaded trend is cut into regions without any Reader, as the CLI does
    data, region = consumer.select_region(data=consumer.trend_coef, region="io")
    assert region == "Indian Ocean"
    assert data.attrs["AQUA_region"] == "Indian Ocean"


def test_load_nothing_on_disk(tmp_path):
    """With no file to read, the result is left as it is instead of being wiped."""
    consumer = Trends(model="FESOM", exp="hpz3", source="monthly-3d", loglevel=loglevel)
    consumer.load(outputdir=tmp_path)
    assert consumer.trend_coef is None

    # A load that finds nothing must not destroy what a run has just computed
    computed = xr.Dataset({"thetao": ("lat", [1.0, 2.0])})
    consumer.trend_coef = computed
    consumer.load(outputdir=tmp_path)
    assert consumer.trend_coef is computed
