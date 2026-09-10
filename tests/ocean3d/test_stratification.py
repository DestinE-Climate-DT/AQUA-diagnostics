from pathlib import Path
from unittest.mock import MagicMock

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
# They refer to the DJF climatology, the season config-ocean3d-en4-stratification pairs
# with both regions used here (Labrador Sea for the profiles, Arctic for the MLD map).
# EXPECTED_MLD_MEAN is the flat average of the 2D MLD field, not the fldmean scalar the
# dim_mean run used to produce: MLD is now taken from the map branch, as in production.
EXPECTED_MLD_MEAN = 45.12769451
EXPECTED_RHO = 26.82583261
PLOT_STEM = "ocean_stratification.{product}.ci.FESOM.hpz3.r1.{region}"
NC_STEM = "stratification.{product}.ci.FESOM.hpz3.r1.{region}"

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
        },
        "dim_mean": ["lat", "lon"],
        # Each production section gets its own region, zipped with its local winter:
        # stratification runs 'ls' with DJF, mld runs 'arctic' with DJF.
        "profile_region": "ls",
        "map_region": "arctic",
    }


@pytest.fixture(scope="module")
def strat_dimean_result(tmp_path_factory, strat_config):
    """Mirror the 'stratification' section of the CLI: dim_mean, mld=False.

    Stratification.run switches on `mld` to decide *which* product it writes, so this
    is the only fixture exercising the 'stratification' netCDF product and the vertical
    profiles PlotStratification consumes.
    """
    tmp_path = tmp_path_factory.mktemp("strat_dimean")
    strat = Stratification(**strat_config["init"])
    strat.run(
        **strat_config["run"],
        region=strat_config["profile_region"],
        dim_mean=strat_config["dim_mean"],
        mld=False,
        outputdir=tmp_path,
    )
    return strat, tmp_path


@pytest.fixture(scope="module")
def strat_map_result(tmp_path_factory, strat_config):
    """Mirror the 'mld' section of the CLI: no dim_mean, mld=True.

    Writes the 'mld' netCDF product and leaves MLD as the 2D field PlotMLD needs.
    Runs on 'arctic': set_extent and set_central_lat_lon in PlotMLD have dedicated
    arctic/antarctic branches, and those are the only regions production plots.
    """
    tmp_path = tmp_path_factory.mktemp("strat_map")
    strat = Stratification(**strat_config["init"])
    strat.run(**strat_config["run"], region=strat_config["map_region"], mld=True, outputdir=tmp_path)
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
def mld_plot(strat_map_result, strat_config):
    """Run PlotMLD once, saving PNG.

    region and proj_name are passed explicitly as cli_ocean_stratification does: the
    polar Orthographic projection is the fragile cartopy path and the only one used
    in production, while the default PlateCarree exercises the generic fallback.
    """
    strat, tmp_path = strat_map_result
    PlotMLD(
        data=strat.data[["mld"]],
        obs=strat.data[["mld"]] * 2,
        outputdir=tmp_path,
        loglevel=loglevel,
    ).plot_mld(region=strat_config["map_region"], proj_name="Orthographic", save_format=["png"])
    return tmp_path


def _assert_nonempty(path):
    assert path.is_file(), f"File not found: {path}"
    assert path.stat().st_size > 0


# --- Tests ---


def test_mld_map(strat_map_result):
    """MLD stays a 2D field, which is what PlotMLD and the 'mld' product need."""
    strat, _ = strat_map_result
    mld = strat.data["mld"]
    assert set(mld.dims) == {"lat", "lon"}
    assert mld.mean().values == pytest.approx(EXPECTED_MLD_MEAN, rel=approx_rel)


def test_rho_value(strat_dimean_result):
    strat, _ = strat_dimean_result
    assert strat.data["rho"].isel({strat.vert_coord: 1}).values == pytest.approx(EXPECTED_RHO, rel=approx_rel)


@pytest.mark.parametrize(
    "result_fixture, product, region",
    [
        ("strat_dimean_result", "stratification", "labrador_sea"),
        ("strat_map_result", "mld", "arctic"),
    ],
)
def test_netcdf_output(request, result_fixture, product, region):
    """The mld flag selects the product name, so both branches need their own file."""
    _, tmp_path = request.getfixturevalue(result_fixture)
    nc = Path(tmp_path) / "netcdf" / f"{NC_STEM.format(product=product, region=region)}.nc"
    _assert_nonempty(nc)


@pytest.mark.parametrize(
    "plot_fixture, product, region, ext",
    [
        ("stratification_plot", "stratification", "labrador_sea", "png"),
        ("mld_plot", "mld", "arctic", "png"),
    ],
)
def test_plot_output(request, plot_fixture, product, region, ext):
    tmp_path = request.getfixturevalue(plot_fixture)
    path = Path(tmp_path) / ext / f"{PLOT_STEM.format(product=product, region=region)}.{ext}"
    _assert_nonempty(path)


def _bare_stratification(data, climatology):
    """Build a Stratification carrying only what compute_climatology reads."""
    strat = object.__new__(Stratification)
    strat.logger = MagicMock()
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
