"""Tests of the SeaIce analysis class on the FESOM test data (1991-1994, regridded to r100)."""

import pytest
import xarray as xr

from aqua.core.util import time_to_string
from aqua.diagnostics import SeaIce
from tests.shared_constants import APPROX_REL, LOGLEVEL

# pytest approximation, to bear with different machines
approx_rel = APPROX_REL * 10
abs_rel = 1e-4

# pytestmark groups tests that run sequentially on the same worker to avoid conflicts
# These tests repeatedly retrieve the same data
pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="diagnostic_setup_class")]


def make_seaice(regions, startdate="1991-01-01", enddate="2000-01-01", **kwargs):
    """SeaIce on the test dataset; the data end in 1994-12, before the requested end date."""
    return SeaIce(
        catalog="ci",
        model="FESOM",
        exp="hpz3",
        source="monthly-2d",
        startdate=startdate,
        enddate=enddate,
        regions=regions,
        regrid="r100",
        loglevel=LOGLEVEL,
        **kwargs,
    )


class TestSeaIce:
    """Test the SeaIce class."""

    @pytest.mark.parametrize(
        ("method", "region", "value", "expected_units", "variable"),
        [
            ("extent", "arctic", 17.2719, "million km^2", "siconc"),
            ("extent", "weddell_sea_ice", 4.5872, "million km^2", "siconc"),
            ("volume", "arctic", 13.7552, "thousands km^3", "siconc"),
            ("volume", "antarctic", 9.1140, "thousands km^3", "siconc"),
        ],
    )
    def test_integrated_series(self, method, region, value, expected_units, variable):
        """A wrong mask, region selection or cell area changes the integrated value of June 1991."""
        seaice = make_seaice(region)
        seaice.run(method=method, var=variable, save_netcdf=False)

        var_name = f"sea_ice_{method}_{region}"
        assert isinstance(seaice.monthly, xr.Dataset)
        assert list(seaice.monthly.data_vars) == [var_name]
        assert all(c in ["variable", "time"] for c in seaice.monthly.coords)
        assert seaice.monthly.attrs["units"] == expected_units
        assert seaice.monthly[var_name].values[5] == pytest.approx(value, rel=approx_rel)
        # only what was asked for is computed
        assert seaice.annual is None and seaice.std_monthly is None and seaice.seasonalcycle is None

    def test_fully_covered_cells_count_for_the_extent(self):
        """A concentration of exactly 1 is sea ice: with all the cells at 1 the extent is the area of the hemisphere."""
        seaice = make_seaice("arctic")
        seaice.method, seaice.var = "extent", "siconc"
        seaice.retrieve(var="siconc")
        seaice.data = xr.ones_like(seaice.data)
        seaice.compute_timeseries()

        # half of the surface of a sphere with a radius of 6371 km, in million km^2
        assert seaice.monthly["sea_ice_extent_arctic"].values[0] == pytest.approx(255.03, rel=approx_rel)

    def test_result_keeps_provenance(self):
        """The plots read model, region and period from the result: fldarea alone would return the cell-area attributes."""
        seaice = make_seaice("arctic")
        seaice.run(method="extent", save_netcdf=False)

        attrs = seaice.monthly["sea_ice_extent_arctic"].attrs
        assert (attrs["AQUA_model"], attrs["AQUA_exp"], attrs["AQUA_catalog"]) == ("FESOM", "hpz3", "ci")
        assert (attrs["AQUA_region"], attrs["AQUA_method"]) == ("Arctic", "extent")
        assert attrs["long_name"] == "Sea ice extent integrated over Arctic"
        assert attrs["standard_name"] == "arctic_sea_ice_extent"
        # available period on the variable, requested period on the dataset (what load() looks for)
        assert (attrs["AQUA_startdate"], attrs["AQUA_enddate"]) == ("1991-01-01", "1994-12-01")
        assert seaice.monthly.attrs["AQUA_enddate"] == "2000-01-01"
        assert "history" in attrs

    def test_invalid_method_raises(self):
        """An unknown method is rejected before any data is retrieved."""
        with pytest.raises(ValueError, match="Invalid method"):
            make_seaice("antarctic").run(method="wrong_method", save_netcdf=False)

    def test_invalid_region_raises(self):
        """Region names are checked against the regions file when the instance is created."""
        with pytest.raises(ValueError, match="Invalid region name"):
            make_seaice([1, 2, 3])

    def test_single_std_date_raises(self):
        """The std period needs both dates: with one only the base class would fail later with an obscure error."""
        with pytest.raises(ValueError, match="std_startdate and std_enddate"):
            make_seaice("arctic", std_startdate="1992-01-01")

    @pytest.mark.parametrize(
        ("method", "region", "variable", "value"),
        [
            ("extent", "arctic", "siconc", 0.883124),
            ("extent", "antarctic", "siconc", 0.744789),
            ("volume", "antarctic", "siconc", 1.042678),
        ],
    )
    def test_std_by_calendar_month(self, method, region, variable, value):
        """The std of June over the analysis period, used when no std dates are given."""
        seaice = make_seaice(region)
        seaice.run(method=method, var=variable, std=True, save_netcdf=False)

        std = seaice.std_monthly[f"std_sea_ice_{method}_{region}"]
        assert list(seaice.std_monthly.data_vars) == [std.name]
        assert std.dims == ("month",) and std.sizes["month"] == 12
        assert std.values[5] == pytest.approx(value, rel=approx_rel)
        assert seaice.std_monthly.attrs["units"] == seaice.monthly.attrs["units"]
        # the std says which period it covers and nothing about the analysis one
        assert (std.attrs["AQUA_std_startdate"], std.attrs["AQUA_std_enddate"]) == ("1991-01-01", "1994-12-01")
        assert "AQUA_startdate" not in std.attrs

    def test_std_uses_std_window_dates(self):
        """With std dates the std comes from that period only."""
        seaice = make_seaice("arctic", std_startdate="1992-01-01", std_enddate="1993-12-31")
        seaice.run(method="extent", std=True, save_netcdf=False)
        std = seaice.std_monthly["std_sea_ice_extent_arctic"]

        june = seaice.monthly["sea_ice_extent_arctic"].sel(time=["1992-06", "1993-06"])
        assert std.values[5] == pytest.approx(float(june.std()), rel=approx_rel)
        assert std.attrs["AQUA_std_startdate"] == time_to_string(seaice.std_startdate)
        assert std.attrs["AQUA_std_enddate"] == time_to_string(seaice.std_enddate)
        assert "AQUA_startdate" not in std.attrs and "AQUA_enddate" not in std.attrs

    @pytest.mark.parametrize(("region", "value"), [("arctic", 21.4141), ("antarctic", 3.2294)])
    def test_seasonal_cycle(self, region, value):
        """The January mean of the cycle, with its std by calendar month."""
        seaice = make_seaice(region)
        seaice.run(method="extent", seasonal_cycle=True, std=True, save_netcdf=False)

        cycle = seaice.seasonalcycle[f"sea_ice_extent_{region}"]
        assert cycle.dims == ("month",)
        assert cycle.values[0] == pytest.approx(value, rel=approx_rel)
        assert "month" in seaice.std_monthly.coords

    def test_annual_means(self):
        """Annual means and their std: 1991-1994 are four complete years."""
        seaice = make_seaice("arctic")
        seaice.run(method="extent", annual=True, std=True, save_netcdf=False)

        monthly = seaice.monthly["sea_ice_extent_arctic"]
        annual = seaice.annual["sea_ice_extent_arctic"]
        assert annual.sizes["time"] == 4
        assert annual.values[0] == pytest.approx(float(monthly.sel(time="1991").mean()), rel=approx_rel)
        std = seaice.std_annual["std_sea_ice_extent_arctic"]
        assert std.ndim == 0  # a single value, as plot_timeseries expects
        assert float(std) == pytest.approx(float(annual.std()), rel=approx_rel)

    def test_annual_means_need_a_complete_year(self):
        """Ten months are not a year: no annual mean, while the monthly series is still there."""
        seaice = make_seaice("arctic", startdate="1991-02-01", enddate="1991-11-30")
        seaice.run(method="extent", annual=True, std=True, save_netcdf=False)

        assert seaice.monthly.sizes["time"] == 10
        assert seaice.annual is None and seaice.std_annual is None

    @pytest.mark.parametrize(
        ("method", "region", "value", "variable"),
        [("fraction", "arctic", 0.2817, "siconc"), ("thickness", "antarctic", 0.0619, "sithick")],
    )
    def test_2d_monthly_climatology(self, method, region, value, variable):
        """The mean of the monthly climatology maps over the region."""
        seaice = make_seaice(region)
        seaice.run(method=method, var=variable, save_netcdf=False)

        result = seaice.climatology[f"sea_ice_{method}_{region}"]
        assert set(result.dims) == {"month", "lat", "lon"}
        assert result.mean(skipna=True).values == pytest.approx(value, rel=approx_rel, abs=abs_rel)
        assert seaice.monthly is None

    def test_save_and_load_round_trip(self, tmp_path):
        """A new instance finds the files of a run, also when the data are shorter than the requested period."""
        seaice = make_seaice(["arctic", "antarctic"])
        seaice.run(method="extent", annual=True, std=True, outputdir=str(tmp_path))

        loaded = make_seaice(["arctic", "antarctic"])
        loaded.load(method="extent", outputdir=str(tmp_path))
        assert loaded.reader is None  # no data was retrieved
        for name in ("monthly", "annual", "std_monthly", "std_annual"):
            xr.testing.assert_allclose(getattr(loaded, name), getattr(seaice, name))
        assert loaded.monthly["sea_ice_extent_arctic"].attrs["AQUA_region"] == "Arctic"
        assert "SeaIce diagnostic" in loaded.monthly.attrs["history"]
        assert "integrated over arctic" in loaded.monthly["sea_ice_extent_arctic"].attrs["history"]
        assert loaded.seasonalcycle is None

    def test_load_checks_the_period_and_the_regions(self, tmp_path):
        """Files of another period or of other regions are not the result of this configuration."""
        make_seaice(["arctic", "antarctic"]).run(method="extent", outputdir=str(tmp_path))

        other_period = make_seaice(["arctic", "antarctic"], startdate="1992-01-01")
        other_period.load(method="extent", outputdir=str(tmp_path))
        assert other_period.monthly is None

        other_regions = make_seaice(["arctic"])
        other_regions.load(method="extent", outputdir=str(tmp_path))
        assert other_regions.monthly is None

    def test_seasonal_cycle_round_trip(self, tmp_path):
        """The seasonal cycle and its std have their own files, apart from those of the time series."""
        seaice = make_seaice("arctic")
        seaice.run(method="extent", seasonal_cycle=True, std=True, outputdir=str(tmp_path))

        loaded = make_seaice("arctic")
        loaded.load(method="extent", seasonal_cycle=True, outputdir=str(tmp_path))
        xr.testing.assert_allclose(loaded.seasonalcycle, seaice.seasonalcycle)
        xr.testing.assert_allclose(loaded.std_monthly, seaice.std_monthly)
        # the run did not leave a time series file behind
        loaded.load(method="extent", outputdir=str(tmp_path))
        assert loaded.monthly is None
