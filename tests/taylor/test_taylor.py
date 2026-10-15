"""Tests for the Taylor diagnostic and its plotting class, on the ERA5 CI data."""

import os

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from aqua.diagnostics import PlotTaylor, Taylor
from tests.shared_constants import APPROX_REL, DPI, LOGLEVEL

pytestmark = [pytest.mark.diagnostics]

# Two different periods of the same dataset, so that the statistics are not trivial
DATASET = {"catalog": "ci", "model": "ERA5", "exp": "era5-hpz3", "source": "monthly", "regrid": "r100"}
DATES = {"startdate": "1990-01-01", "enddate": "1991-12-31"}
DATES_REF = {"startdate": "1992-01-01", "enddate": "1994-12-31"}
PREFIX = "taylor.taylor_statistics.ci.ERA5.era5-hpz3.r1.ci.ERA5.era5-hpz3"


@pytest.fixture(scope="module")
def outputdir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("taylor"))


@pytest.fixture(scope="module")
def reference():
    return Taylor(**DATASET, **DATES_REF, loglevel=LOGLEVEL)


@pytest.fixture(scope="module")
def statistics_2t(reference, outputdir):
    """Statistics of the 2m temperature, saved to disk."""
    reference.run(var="2t")
    taylor = Taylor(**DATASET, **DATES, loglevel=LOGLEVEL)
    taylor.run(var="2t")
    taylor.compute_statistics(reference=reference, outputdir=outputdir)
    return taylor


class TestTaylor:
    """Tests of the Taylor class."""

    def test_climatology(self, statistics_2t):
        """The climatology holds the annual and the seasonal means."""
        climatology = statistics_2t.climatology
        assert list(climatology.period.values) == ["annual", "DJF", "MAM", "JJA", "SON"]
        assert climatology.name == "2t"
        assert climatology.attrs["AQUA_model"] == "ERA5"
        assert statistics_2t.area is not None

    def test_statistics(self, statistics_2t):
        """Statistics are close to the reference, but not identical."""
        stats = statistics_2t.statistics
        assert isinstance(stats, xr.Dataset)
        assert stats["corr"].dims == ("period",)
        annual = stats.sel(period="annual")
        assert 0.99 < float(annual["corr"]) < 1
        assert float(annual["std_norm"]) == pytest.approx(1, rel=0.05)
        assert stats.attrs["AQUA_model_ref"] == "ERA5"
        assert stats.attrs["AQUA_startdate_ref"] == "1992-01-01"
        assert stats.attrs["short_name"] == "2t"

    def test_netcdf_and_load(self, statistics_2t, outputdir):
        """The statistics are saved with the reference in the filename and can be loaded without any retrieve."""
        assert os.path.exists(os.path.join(outputdir, "netcdf", f"{PREFIX}.2t.nc"))

        loaded = Taylor(**DATASET, **DATES, loglevel=LOGLEVEL)
        reference = Taylor(**DATASET, loglevel=LOGLEVEL)
        loaded.load(var="2t", reference=reference, outputdir=outputdir)
        assert loaded.statistics is not None
        assert float(loaded.statistics["corr"].sel(period="annual")) == pytest.approx(
            float(statistics_2t.statistics["corr"].sel(period="annual")), rel=APPROX_REL
        )

        # A different period is not loaded
        other = Taylor(**DATASET, startdate="1990-01-01", enddate="1990-12-31", loglevel=LOGLEVEL)
        other.load(var="2t", reference=reference, outputdir=outputdir)
        assert other.statistics is None

    def test_options(self, outputdir):
        """Units, pressure level, region and formula, annual only."""
        options = [
            ({"var": "tprate", "units": "mm/day"}, "tprate.tropics"),
            ({"var": "q", "plev": 85000}, "q.85000.tropics"),
            ({"var": "tnlwrf+tnswrf", "formula": True, "short_name": "tnr"}, "tnr.tropics"),
        ]
        reference = Taylor(**DATASET, **DATES_REF, region="tropics", loglevel=LOGLEVEL)
        for kwargs, suffix in options:
            reference.run(**kwargs, seasonal=False)
            taylor = Taylor(**DATASET, **DATES, region="tropics", loglevel=LOGLEVEL)
            taylor.run(**kwargs, seasonal=False)
            taylor.compute_statistics(reference=reference, outputdir=outputdir)
            assert list(taylor.statistics.period.values) == ["annual"]
            assert taylor.statistics.attrs["AQUA_region"] == "Tropics"
            assert float(taylor.climatology.lat.max()) <= 15
            assert os.path.exists(os.path.join(outputdir, "netcdf", f"{PREFIX}.{suffix}.nc"))
        assert taylor.statistics.attrs["short_name"] == "tnr"

    def test_errors(self, reference):
        """Statistics need both climatologies, and 3D variables need a level."""
        taylor = Taylor(**DATASET, **DATES, loglevel=LOGLEVEL)
        with pytest.raises(ValueError, match="climatologies"):
            taylor.compute_statistics(reference=reference, save_netcdf=False)
        with pytest.raises(ValueError, match="pressure levels"):
            taylor.run(var="q")
        with pytest.raises(KeyError):
            taylor.run(var="notavar+2t", formula=True)


class TestPlotTaylor:
    """Tests of the PlotTaylor class."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close("all")

    def test_plot_annual_and_seasonal(self, statistics_2t, outputdir):
        """Annual and seasonal diagrams are saved."""
        plot = PlotTaylor(save_format=["png"], dpi=DPI, outputdir=outputdir, loglevel=LOGLEVEL)
        fig, ax = plot.plot_taylor(statistics_2t.statistics)
        assert ax.name == "polar"
        assert "Taylor diagram" in ax.get_title()

        plot.plot_taylor([statistics_2t.statistics], period="seasonal", color_by="period", marker_by="dataset")
        plot.plot_taylor(statistics_2t.statistics, normalize=False, save=False)

        prefix = "taylor.taylor_diagram.ci.ERA5.era5-hpz3.r1.ci.ERA5.era5-hpz3.2t"
        assert os.path.exists(os.path.join(outputdir, "png", f"{prefix}.annual.png"))
        assert os.path.exists(os.path.join(outputdir, "png", f"{prefix}.seasonal.png"))

    def test_plot_errors(self, statistics_2t):
        """Invalid grouping keys, missing periods and non normalised mixed references raise."""
        plot = PlotTaylor(save_format=None, loglevel=LOGLEVEL)
        stats = statistics_2t.statistics
        with pytest.raises(ValueError, match="color_by"):
            plot.plot_taylor(stats, color_by="model")
        with pytest.raises(ValueError, match="different"):
            plot.plot_taylor(stats, color_by="var", marker_by="var")
        with pytest.raises(ValueError, match="No statistics"):
            plot.plot_taylor(stats.sel(period=["annual"]), period="JJA")
        with pytest.raises(ValueError, match="single reference"):
            plot.plot_taylor(stats, period="seasonal", normalize=False)
