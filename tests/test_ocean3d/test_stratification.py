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

pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="dask_operations")]


# --- Fixtures ---


@pytest.fixture(scope="session")
def strat_config():
    return {
        "catalog": "ci",
        "model": "FESOM",
        "exp": "hpz3",
        "source": "monthly-3d",
        "regrid": "r100",
        "loglevel": loglevel,
    }


@pytest.fixture(scope="module")
def strat_dimean_result(tmp_path_factory, strat_config):
    """Run with dim_mean — collapsed scalar, used for value assertions and PlotStratification."""
    tmp_path = tmp_path_factory.mktemp("strat_dimean")
    strat = Stratification(**strat_config)
    strat.run(
        dim_mean=["lat", "lon"],
        var=["thetao", "so"],
        climatology="January",
        region="ls",
        mld=True,
        outputdir=tmp_path,
    )
    return strat, tmp_path


@pytest.fixture(scope="module")
def strat_map_result(tmp_path_factory, strat_config):
    """Run without dim_mean — 2D map data, used for PlotMLD."""
    tmp_path = tmp_path_factory.mktemp("strat_map")
    strat = Stratification(**strat_config)
    strat.run(var=["thetao", "so"], climatology="January", region="ls", mld=True, outputdir=tmp_path)
    return strat, tmp_path


@pytest.fixture(scope="module")
def stratification_plot(strat_dimean_result):
    """Run PlotStratification once, saving PNG and PDF."""
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
    ).plot_stratification(save_format=["png", "pdf"])
    return tmp_path


@pytest.fixture(scope="module")
def mld_plot(strat_dimean_result, strat_map_result):
    """Run PlotMLD once, saving PNG and PDF."""
    _, tmp_path = strat_dimean_result
    strat, _ = strat_map_result
    PlotMLD(
        data=strat.data[["mld"]],
        obs=strat.data[["mld"]] * 2,
        outputdir=tmp_path,
        loglevel=loglevel,
    ).plot_mld(save_format=["png", "pdf"])
    return tmp_path


# --- Tests ---


def test_strat_not_none(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat is not None


def test_mld_value(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat.data["mld"].values == pytest.approx(EXPECTED_MLD, rel=approx_rel)


def test_rho_value(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat.data["rho"].isel({strat.vert_coord: 1}).values == pytest.approx(EXPECTED_RHO, rel=approx_rel)


def test_stratification_png(stratification_plot):
    png = Path(stratification_plot) / "png" / "ocean_stratification.stratification.ci.FESOM.hpz3.r1.labrador_sea.png"
    assert png.is_file(), f"PNG not found: {png}"
    assert png.stat().st_size > 0


def test_stratification_pdf(stratification_plot):
    pdf = Path(stratification_plot) / "pdf" / "ocean_stratification.stratification.ci.FESOM.hpz3.r1.labrador_sea.pdf"
    assert pdf.is_file(), f"PDF not found: {pdf}"
    assert pdf.stat().st_size > 0


def test_mld_png(mld_plot):
    png = Path(mld_plot) / "png" / "ocean_stratification.mld.ci.FESOM.hpz3.r1.labrador_sea.png"
    assert png.is_file(), f"PNG not found: {png}"
    assert png.stat().st_size > 0


def test_mld_pdf(mld_plot):
    pdf = Path(mld_plot) / "pdf" / "ocean_stratification.mld.ci.FESOM.hpz3.r1.labrador_sea.pdf"
    assert pdf.is_file(), f"PDF not found: {pdf}"
    assert pdf.stat().st_size > 0
