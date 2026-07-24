import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics.histogram import PlotHistogram
from tests.shared_constants import DPI, LOGLEVEL

loglevel = LOGLEVEL


def _make_hist_data(
    model="IFS",
    exp="test-tco79",
    region="global",
    startdate="2000-01-01",
    enddate="2005-12-31",
    realization=None,
):
    """Build a PDF-like histogram DataArray with AQUA_* metadata."""
    bins = np.linspace(250, 320, 50)
    attrs = {
        "AQUA_catalog": "test_catalog",
        "AQUA_model": model,
        "AQUA_exp": exp,
        "AQUA_region": region,
        "AQUA_startdate": startdate,
        "AQUA_enddate": enddate,
        "short_name": "skt",
        "standard_name": "skin_temperature",
        "long_name": "Skin Temperature",
        "units": "K",
    }
    if realization is not None:
        attrs["AQUA_realization"] = realization
    da = xr.DataArray(
        np.random.exponential(scale=10, size=50),
        dims=["center_of_bin"],
        coords={"center_of_bin": bins},
        attrs=attrs,
    )
    da.center_of_bin.attrs["units"] = "K"
    return da


@pytest.mark.diagnostics
class TestPlotHistogram:
    """Tests for the PlotHistogram class."""

    def setup_method(self):
        self.data = _make_hist_data()
        self.ref = _make_hist_data(model="ERA5", exp="era5", startdate="1980-01-01", enddate="2020-12-31")

    def test_init_metadata_labels_title(self):
        """Init extracts AQUA_* metadata; labels/title reflect model+region."""
        p = PlotHistogram(data=self.data, ref_data=self.ref, loglevel=loglevel)

        assert (p.len_data, p.len_ref) == (1, 1)
        assert p.models[0] == "IFS" and p.exps[0] == "test-tco79"
        assert p.region == "global"
        assert "IFS test-tco79" in p.set_data_labels()[0]
        assert p.set_ref_label() == "ERA5"
        assert "era5" not in p.set_ref_label()
        title = p.set_title()
        # Variable rendered lower case, but title still starts with a capital letter
        assert "Skin temperature" in title or "skin_temperature" in title
        assert "Skin Temperature" not in title

    def test_realization_propagation(self):
        """AQUA_realization attribute extracted to plotter.realizations."""
        p = PlotHistogram(data=_make_hist_data(realization="r1i1p1f1"), loglevel=loglevel)
        assert p.realizations[0] == "r1i1p1f1"

    @pytest.mark.parametrize("density,prefix", [(True, "Probability density function (PDF)"), (False, "Histogram of")])
    def test_description_prefix(self, density, prefix):
        """density=True/False switches PDF/Histogram prefix."""
        p = PlotHistogram(data=self.data, density=density, loglevel=loglevel)
        assert p.set_description().startswith(prefix)

    def test_description_variable_lowercase_no_units_compared_to(self):
        """Variable rendered lower case, units removed, 'compared to' replaces 'vs'."""
        ref_same = _make_hist_data(model="ERA5", exp="era5")  # same period as self.data
        desc = PlotHistogram(data=self.data, ref_data=ref_same, loglevel=loglevel).set_description()
        assert "skin temperature" in desc
        assert "[K]" not in desc
        assert " vs " not in desc
        assert "compared to ERA5" in desc

    def test_description_different_periods(self):
        """Both date ranges shown in %Y-%m when data/ref periods differ."""
        desc = PlotHistogram(data=self.data, ref_data=self.ref, loglevel=loglevel).set_description()
        assert "2000-01" in desc and "2005-12" in desc
        assert "1980-01" in desc and "2020-12" in desc
        assert "IFS" in desc and "ERA5" in desc

    def test_description_same_period_collapses(self):
        """Description collapses to a single range when data/ref periods match."""
        ref_same = _make_hist_data(model="ERA5", exp="era5")  # default startdate/enddate match self.data
        desc = PlotHistogram(data=self.data, ref_data=ref_same, loglevel=loglevel).set_description()
        assert desc.count("2000-01") == 1 and "compared to" in desc

    def test_description_multi_and_region(self):
        """Multi-dataset description lists pairs; non-global region surfaces in description."""
        p_multi = PlotHistogram(data=[self.data, _make_hist_data(exp="test-tco159")], ref_data=self.ref, loglevel=loglevel)
        desc_multi = p_multi.set_description()
        assert "comparing 2 datasets" in desc_multi
        assert "IFS/test-tco79" in desc_multi and "IFS/test-tco159" in desc_multi

        p_reg = PlotHistogram(data=_make_hist_data(region="Tropics"), loglevel=loglevel)
        assert "over Tropics" in p_reg.set_description()

    @pytest.mark.parametrize("with_ref,smooth,expected_lines", [(False, False, 1), (True, True, 2)])
    def test_plot_smoke(self, with_ref, smooth, expected_lines):
        """plot() returns a figure with the expected number of lines."""
        p = PlotHistogram(data=self.data, ref_data=self.ref if with_ref else None, loglevel=loglevel)
        fig, ax = p.plot(smooth=smooth, smooth_window=5)
        assert len(ax.lines) == expected_lines
        plt.close(fig)

    def test_run_writes_file(self, tmp_path):
        """run() saves an output file."""
        p = PlotHistogram(data=self.data, ref_data=self.ref, loglevel=loglevel)
        p.run(outputdir=str(tmp_path), rebuild=True, dpi=DPI, format="png")
        assert len(list(tmp_path.rglob("*.png"))) > 0
