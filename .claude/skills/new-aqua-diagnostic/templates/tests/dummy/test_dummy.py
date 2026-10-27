"""Tests for the Dummy tool, run on the real `ci` catalog."""

import numpy as np
import pytest

from aqua.diagnostics.dummy import Dummy, PlotDummy
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL, assert_nonempty

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

DATASET = {"catalog": "ci", "model": "ERA5", "exp": "era5-hpz3", "source": "monthly", "regrid": "r100"}


@pytest.fixture(scope="module")
def dummy_run(tmp_path_factory):
    """Run Dummy once and share it: returns (instance, outputdir)."""
    outputdir = tmp_path_factory.mktemp("dummy")
    diag = Dummy(**DATASET, loglevel=LOGLEVEL)
    diag.run(var="tcc", outputdir=str(outputdir))
    return diag, outputdir


def test_run_result(dummy_run):
    """The result is named, in memory, annual and carries the provenance attributes."""
    diag, _ = dummy_run
    assert diag.result.name == "tcc"
    assert diag.result.dims == ("time",)
    assert diag.result.attrs["AQUA_model"] == "ERA5"
    assert diag.result.attrs["AQUA_exp"] == "era5-hpz3"
    assert "AQUA_startdate" in diag.result.attrs
    assert not np.isnan(diag.result.values).any()


def test_region(tmp_path):
    """A region is stored as AQUA_region and ends up in the filename."""
    diag = Dummy(**DATASET, region="tropics", loglevel=LOGLEVEL)
    diag.run(var="tcc", outputdir=str(tmp_path))
    assert diag.result.attrs["AQUA_region"] == diag.region
    assert list(tmp_path.rglob("*.nc"))


def test_not_enough_data():
    """Less than MINIMUM_MONTHS_REQUIRED months raises, it is the CLI that decides to skip."""
    from aqua.core.exceptions import NotEnoughDataError

    diag = Dummy(**DATASET, startdate="1990-01-01", enddate="1990-03-31", loglevel=LOGLEVEL)
    with pytest.raises(NotEnoughDataError):
        diag.run(var="tcc", save_netcdf=False)


def test_save_load_round_trip(dummy_run):
    """A fresh instance that never retrieved anything gets back the same result (plot-only runs)."""
    diag, outputdir = dummy_run
    fresh = Dummy(**DATASET, loglevel=LOGLEVEL)
    fresh.load(var="tcc", outputdir=str(outputdir))
    assert fresh.result is not None
    np.testing.assert_allclose(fresh.result.values, diag.result.values, rtol=APPROX_REL)
    assert fresh.result.attrs["AQUA_model"] == diag.result.attrs["AQUA_model"]


def test_load_checks_the_period(dummy_run):
    """With dates, only a file computed over that period is loaded: this is what plot-only runs ask for."""
    diag, outputdir = dummy_run
    first, last = diag.result.attrs["AQUA_startdate"], diag.result.attrs["AQUA_enddate"]

    same = Dummy(**DATASET, startdate=first, enddate=last, loglevel=LOGLEVEL)
    same.load(var="tcc", outputdir=str(outputdir))
    assert same.result is not None

    other = Dummy(**DATASET, startdate="1991-01-01", enddate="1992-12-31", loglevel=LOGLEVEL)
    other.load(var="tcc", outputdir=str(outputdir))
    assert other.result is None


def test_load_missing_file(tmp_path):
    """A missing file leaves the result untouched."""
    diag = Dummy(**DATASET, loglevel=LOGLEVEL)
    diag.load(var="tcc", outputdir=str(tmp_path))
    assert diag.result is None


def test_plot(dummy_run, tmp_path):
    """The plot class works from the result alone and writes the figure."""
    diag, _ = dummy_run
    plot = PlotDummy(data=diag.result, outputdir=str(tmp_path), loglevel=LOGLEVEL)
    assert "ERA5" in plot.set_title()
    assert plot.set_description().endswith(".")
    plot.plot(save_format="png", dpi=DPI)
    assert_nonempty(next(tmp_path.rglob("*.png")))
