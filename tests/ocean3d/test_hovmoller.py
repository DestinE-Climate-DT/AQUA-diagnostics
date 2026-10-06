import shutil
from pathlib import Path

import pytest
import xarray as xr

from aqua.diagnostics.ocean_drift import Hovmoller, PlotHovmoller
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10
dpi = DPI

# --- Constants ---
EXPECTED_FULL = {"thetao": 22.2086629652034, "so": 36.57638045014168}
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
    """Run both plot types once. Hovmoller must run before timeseries."""
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
    """std_anom_t0 is anom_t0 divided by its temporal standard deviation."""
    hov, _ = hovmoller_result
    data = _by_drift_type(hov)
    anom, std_anom = data["anom_t0"], data["std_anom_t0"]

    xr.testing.assert_allclose(std_anom, anom / anom.std("time"))


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


def _consumer():
    """A Hovmoller that never retrieved and does not even know its catalog"""
    return Hovmoller(model="FESOM", exp="hpz3", source="monthly-3d", loglevel=loglevel)


def test_load_roundtrip(hovmoller_result, hovmoller_config):
    """A run that only plots finds on disk exactly the products a previous run computed.

    The consumer never retrieves: catalog, realization and region names are rebuilt without data access.
    """
    producer, tmp_path = hovmoller_result
    consumer = _consumer()
    consumer.load(outputdir=tmp_path, **{key: hovmoller_config["run"][key] for key in ("regions", "anomaly_ref")})

    # The unknown region is skipped, as run does
    assert set(consumer.processed_data) == set(producer.processed_data)
    for region, produced in producer.processed_data.items():
        loaded = consumer.processed_data[region]
        # Same products in the same order, which sets the rows of the plots
        assert [ds.attrs["AQUA_ocean_drift_type"] for ds in loaded] == EXPECTED_DRIFT_TYPES
        for loaded_ds, produced_ds in zip(loaded, produced):
            xr.testing.assert_allclose(loaded_ds, produced_ds)
            # The attributes PlotHovmoller reads survived the round trip
            assert loaded_ds.attrs["AQUA_region"] == produced_ds.attrs["AQUA_region"]
            for attr in ["AQUA_catalog", "AQUA_model", "AQUA_exp"]:
                assert loaded_ds["thetao"].attrs[attr] == produced_ds["thetao"].attrs[attr]


def test_load_nothing_on_disk(tmp_path):
    """With no files to read, the results are left as they are instead of being wiped."""
    consumer = _consumer()
    consumer.load(outputdir=tmp_path, regions=["sss"], anomaly_ref="t0")
    assert consumer.processed_data == {}

    # A load that finds nothing must not destroy what a run has just computed
    computed = [xr.Dataset()]
    consumer.processed_data["sss"] = computed
    consumer.load(outputdir=tmp_path, regions=["sss"], anomaly_ref="t0")
    assert consumer.processed_data["sss"] is computed


def test_load_incomplete_products(hovmoller_result, tmp_path):
    """A missing product gives no data for its region, rather than a plot silently missing a row."""
    _, run_dir = hovmoller_result
    (tmp_path / "netcdf").mkdir()
    for drift_type in ["full", "anom_t0"]:  # std_anom_t0 is missing
        name = f"{PLOT_STEM.format(product='hovmoller')}.{drift_type}.nc"
        shutil.copy(Path(run_dir) / "netcdf" / name, tmp_path / "netcdf" / name)

    consumer = _consumer()
    consumer.load(outputdir=tmp_path, regions=["sss"], anomaly_ref="t0")

    assert "sss" not in consumer.processed_data
