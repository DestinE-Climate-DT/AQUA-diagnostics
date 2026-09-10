from pathlib import Path

import pytest

from aqua.diagnostics.ocean_stratification import PlotMLD, PlotStratification
from aqua.diagnostics.ocean_stratification.stratification import Stratification
from tests.shared_constants import APPROX_REL, LOGLEVEL

loglevel = LOGLEVEL
approx_rel = APPROX_REL * 10

# --- Constants ---
# Expected values valid with aqua-core >=1.0.0a6, which renames the FESOM/NEMO
# vertical coordinate 'level' -> 'depth' (CoordIdentifier NEMO-layers rule).
EXPECTED_MLD = 25.49270658
EXPECTED_RHO = 26.8719114
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
            "climatology": "January",
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
