from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.ocean_stratification import PlotMLD, PlotStratification
from aqua.diagnostics.ocean_stratification.stratification import Stratification
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10

# --- Constants ---
# Expected values valid with aqua-core >=1.0.0a6, which renames the FESOM/NEMO
# vertical coordinate 'level' -> 'depth' (CoordIdentifier NEMO-layers rule).
# They refer to the DJF climatology, i.e. the season that config-ocean3d-en4-stratification
# pairs with the Labrador Sea.
EXPECTED_MLD = 24.76439717
EXPECTED_RHO = 26.82583261
PLOT_STEM = "ocean_stratification.{product}.ci.FESOM.hpz3.r1.labrador_sea"
NC_STEM = "stratification.{product}.ci.FESOM.hpz3.r1.labrador_sea"

pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="dask_operations")]


# --- Fixtures ---


@pytest.fixture(scope="session")
def strat_config():
    return {
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
            "climatology": "DJF",
            "region": "ls",
            "mld": True,
        },
        "dim_mean": ["lat", "lon"],
    }


@pytest.fixture(scope="module")
def strat_dimean_result(tmp_path_factory, strat_config):
    """Run with dim_mean — collapsed scalar, used for value assertions and PlotStratification."""
    tmp_path = tmp_path_factory.mktemp("strat_dimean")
    strat = Stratification(**strat_config["init"])
    strat.run(
        **strat_config["run"],
        dim_mean=strat_config["dim_mean"],
        outputdir=tmp_path,
    )
    return strat, tmp_path


@pytest.fixture(scope="module")
def strat_map_result(tmp_path_factory, strat_config):
    """Run without dim_mean — 2D map data, used for PlotMLD."""
    tmp_path = tmp_path_factory.mktemp("strat_map")
    strat = Stratification(**strat_config["init"])
    strat.run(**strat_config["run"], outputdir=tmp_path)
    return strat, tmp_path


@pytest.fixture(scope="module")
def stratification_plot(strat_dimean_result):
    """Run PlotStratification once, saving PNG."""
    strat, tmp_path = strat_dimean_result
    data = strat.data[["thetao", "so", "rho"]]
    obs = data * 1.2
    obs.attrs["model"] = strat.model
    obs.attrs["exp"] = strat.exp
    PlotStratification(
        data=data,
        obs=obs,
        loglevel=loglevel,
        outputdir=tmp_path,
    ).plot_stratification(save_format=["png"])
    return tmp_path


@pytest.fixture(scope="module")
def mld_plot(strat_map_result):
    """Run PlotMLD once, saving PNG."""
    strat, tmp_path = strat_map_result
    PlotMLD(
        data=strat.data[["mld"]],
        obs=strat.data[["mld"]] * 2,
        outputdir=tmp_path,
        loglevel=loglevel,
    ).plot_mld(save_format=["png"])
    return tmp_path


def _assert_nonempty(path):
    assert path.is_file(), f"File not found: {path}"
    assert path.stat().st_size > 0


# --- Tests ---


def test_mld_value(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat.data["mld"].values == pytest.approx(EXPECTED_MLD, rel=approx_rel)


def test_rho_value(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat.data["rho"].isel({strat.vert_coord: 1}).values == pytest.approx(EXPECTED_RHO, rel=approx_rel)


@pytest.mark.parametrize("result_fixture", ["strat_dimean_result", "strat_map_result"])
def test_netcdf_output(request, result_fixture):
    _, tmp_path = request.getfixturevalue(result_fixture)
    nc = Path(tmp_path) / "netcdf" / f"{NC_STEM.format(product='mld')}.nc"
    _assert_nonempty(nc)


@pytest.mark.parametrize(
    "plot_fixture, product, ext",
    [
        ("stratification_plot", "stratification", "png"),
        ("mld_plot", "mld", "png"),
    ],
)
def test_plot_output(request, plot_fixture, product, ext):
    tmp_path = request.getfixturevalue(plot_fixture)
    path = Path(tmp_path) / ext / f"{PLOT_STEM.format(product=product)}.{ext}"
    _assert_nonempty(path)


def _bare_stratification(data, climatology):
    """Build a Stratification carrying only what compute_climatology reads."""
    strat = object.new(Stratification)
    strat.data = data
    strat.climatology = climatology
    return strat


@pytest.fixture
def monthly_dataset():
    """Two years of monthly data, enough to group by either month or season."""
    time = xr.date_range("1990-01-01", periods=24, freq="MS")
    return xr.Dataset({"thetao": ("time", np.arange(24.0))}, coords={"time": time})


@pytest.mark.parametrize("climatology, expected_clim_type", [("January", "month"), ("DJF", "season")])
def test_compute_climatology_selects_single_slice(monthly_dataset, climatology, expected_clim_type):
    """A month name or a season name collapses time onto the requested slice."""
    strat = _bare_stratification(monthly_dataset, climatology)
    strat.compute_climatology(climatology=climatology)

    assert strat.clim_type == expected_clim_type
    assert "time" not in strat.data.dims
    assert strat.data.attrs["AQUA_stratification_climatology"] == climatology


@pytest.mark.xfail(
    strict=True,
    reason=(
        "compute_climatology leaves the time axis untouched for these values. clim_type is set "
        "to the truthy string 'Total', so if self.clim_type: enters the branch that only handles "
        "month/year/season and the else computing the time mean is unreachable. The data comes "
        "out unreduced while AQUA_stratification_climatology claims a climatology was computed. "
        "'month' is the default of Stratification.run, so the default is a silent no-op."
    ),
)
@pytest.mark.parametrize("climatology", ["month", "season", "total"])
def test_compute_climatology_reduces_time_for_documented_values(monthly_dataset, climatology):
    """Every value advertised in the run/compute_climatology docstrings must reduce time."""
    strat = _bare_stratification(monthly_dataset, climatology)
    strat.compute_climatology(climatology=climatology)

    assert strat.data.sizes["time"] < monthly_dataset.sizes["time"]
