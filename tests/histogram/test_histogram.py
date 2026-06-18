import pytest

from aqua.diagnostics.histogram import Histogram

loglevel = "DEBUG"
MODEL = dict(model="IFS", exp="test-tco79", source="teleconnections")


@pytest.mark.diagnostics
class TestHistogram:
    """Tests for the Histogram diagnostic class."""

    def setup_method(self):
        self.hist = Histogram(**MODEL, startdate="1990-01-01", enddate="1991-12-31", bins=50, loglevel=loglevel)

    @pytest.mark.parametrize("density,units", [(True, None), (False, "counts")])
    def test_compute_propagates_metadata(self, density, units):
        """retrieve+compute_histogram yields valid hist data with AQUA_* attrs propagated."""
        self.hist.retrieve(var="skt")
        self.hist.compute_histogram(density=density)

        h = self.hist.histogram_data
        assert "center_of_bin" in h.dims and len(h.center_of_bin) == self.hist.bins
        assert h.attrs["AQUA_model"] == "IFS" and h.attrs["AQUA_exp"] == "test-tco79"
        assert h.attrs["AQUA_startdate"] == "1990-01-01" and h.attrs["AQUA_enddate"] == "1991-12-31"
        assert "AQUA_catalog" in h.attrs and "units" in h.center_of_bin.attrs
        if units:
            assert h.attrs["units"] == units

    def test_auto_dates_clean(self):
        """Auto-detected dates are clean YYYY-MM-DD (no timestamp 'T')."""
        hist = Histogram(**MODEL, bins=40, loglevel=loglevel)
        hist.retrieve(var="skt")
        hist.compute_histogram()
        assert "T" not in hist.histogram_data.attrs["AQUA_startdate"]
        assert "T" not in hist.histogram_data.attrs["AQUA_enddate"]

    def test_region_and_custom_range(self):
        """Named region resolves to limits and custom range constrains bin edges."""
        hist = Histogram(**MODEL, region="tropics", bins=25, range=(250, 320), loglevel=loglevel)
        hist.retrieve(var="skt")
        hist.compute_histogram()

        assert hist.region == "Tropics"
        assert hist.lat_limits == [-15, 15]
        bins = hist.histogram_data.center_of_bin
        assert float(bins.min()) >= 250 and float(bins.max()) <= 320

    def test_full_run_with_formula(self, tmp_path):
        """End-to-end run with formula evaluation writes netCDF."""
        self.hist.run(
            var="skt+273.15",
            formula=True,
            long_name="Temperature",
            units="K",
            standard_name="temperature",
            outputdir=str(tmp_path),
            rebuild=True,
        )
        assert self.hist.data.attrs["units"] == "K"
        assert len(list(tmp_path.rglob("*.nc"))) > 0

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="nonexistent_var"):
            self.hist.retrieve(var="nonexistent_var")

    def test_save_without_data_is_noop(self, tmp_path):
        """save_netcdf is a silent no-op when no histogram computed."""
        self.hist.save_netcdf(outputdir=str(tmp_path))
        assert self.hist.histogram_data is None
        assert list(tmp_path.rglob("*.nc")) == []
