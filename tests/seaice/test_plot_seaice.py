"""Tests of PlotSeaIce on the sea ice extent of the FESOM test data (1991-1994, regridded to r100)."""

import os

import pytest

from aqua.diagnostics import PlotSeaIce, SeaIce
from aqua.diagnostics.base import OutputSaver
from tests.shared_constants import DPI, LOGLEVEL

# pytestmark groups tests that run sequentially on the same worker to avoid conflicts
# These tests use setup_class with shared resources (data fetching, tmp files)
pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="diagnostic_setup_class")]

PERIOD = "(from 1991-01 to 1994-12)"


def relabel(dataset, model, offset=0.1):
    """A copy of a result shifted by a constant, as if it came from another model."""
    other = dataset + offset
    other.attrs["AQUA_model"] = model
    for name in other.data_vars:
        other[name].attrs["AQUA_model"] = model
    return other


class TestPlotSeaIce:
    @classmethod
    def setup_class(cls):
        seaice = SeaIce(
            catalog="ci",
            model="FESOM",
            exp="hpz3",
            source="monthly-2d",
            regions=["arctic", "antarctic"],
            startdate="1991-01-01",
            enddate="2000-01-01",
            regrid="r100",
            loglevel=LOGLEVEL,
        )
        seaice.run(method="extent", annual=True, std=True, save_netcdf=False)
        cls.monthly, cls.annual, cls.std = seaice.monthly, seaice.annual, seaice.std_monthly
        cls.ref = relabel(cls.monthly, "OBS")
        seaice.run(method="extent", seasonal_cycle=True, save_netcdf=False)
        cls.cycle = seaice.seasonalcycle
        cls.cycle_ref = relabel(cls.cycle, "OBS")

    @pytest.fixture
    def saved(self, monkeypatch):
        """The calls to OutputSaver.save_figure, captured instead of writing files."""
        calls = []

        def fake_save_figure(self, fig=None, **kwargs):
            calls.append({"saver": self, "fig": fig, **kwargs})

        monkeypatch.setattr(OutputSaver, "save_figure", fake_save_figure)
        return calls

    def test_timeseries_figure(self, saved):
        """One panel per region with model, reference and std band, described by the attributes of the data."""
        psi = PlotSeaIce(monthly_models=self.monthly, monthly_ref=self.ref, monthly_std_ref=self.std, dpi=DPI)
        psi.plot_seaice(plot_type="timeseries", save_format=["png", "pdf"])

        assert len(saved) == 1
        call = saved[0]
        assert call["diagnostic_product"] == "timeseries"
        assert call["extension"] == ["png", "pdf"]
        assert call["extra_keys"] == {"method": "extent", "region": "Arctic_Antarctic"}
        # the model is read from the data: the catalog is the catalog, not the model name
        assert (call["saver"].catalog, call["saver"].model, call["saver"].exp) == ("ci", "FESOM", "hpz3")
        assert call["metadata"]["Description"] == (
            "Time series of sea ice extent integrated over Arctic and Antarctic regions "
            f"for FESOM hpz3 {PERIOD}  OBS hpz3 {PERIOD} is used as a reference."
            f" Shaded areas represent ±2σ uncertainty bands {PERIOD}."
        )

        axes = call["fig"].axes
        assert [ax.get_title() for ax in axes] == ["Sea ice extent in Arctic", "Sea ice extent in Antarctic"]
        for ax in axes:
            assert [text.get_text() for text in ax.get_legend().get_texts()] == ["FESOM hpz3", "OBS hpz3"]
            assert [len(line.get_xdata()) for line in ax.get_lines()] == [48, 48]  # model and reference
            assert len(ax.collections) == 1  # the std band
        assert "Arctic" in axes[0].get_ylabel() and "Antarctic" in axes[1].get_ylabel()

    def test_annual_means_are_dashed(self, saved):
        """The annual means of the four complete years are added to the monthly series."""
        psi = PlotSeaIce(monthly_models=self.monthly, annual_models=self.annual, dpi=DPI)
        psi.plot_seaice(plot_type="timeseries", save_format="png")

        lines = saved[0]["fig"].axes[0].get_lines()
        assert [(line.get_linestyle(), len(line.get_xdata())) for line in lines] == [("-", 48), ("--", 4)]

    def test_regions_to_plot_selects_the_panels(self, saved):
        """Only the requested regions are drawn and named in the file."""
        psi = PlotSeaIce(monthly_models=self.monthly, monthly_ref=self.ref, regions_to_plot=["Antarctic"], dpi=DPI)
        psi.plot_seaice(plot_type="timeseries", save_format="png")

        assert [ax.get_title() for ax in saved[0]["fig"].axes] == ["Sea ice extent in Antarctic"]
        assert saved[0]["extra_keys"]["region"] == "Antarctic"
        assert "Antarctic region for" in saved[0]["metadata"]["Description"]

    def test_none_entries_are_skipped(self, saved):
        """A missing reference in a list (a dataset the CLI could not compute) does not break the figure."""
        psi = PlotSeaIce(monthly_models=[self.monthly], monthly_ref=[None, self.ref], monthly_std_ref=[None], dpi=DPI)
        psi.plot_seaice(plot_type="timeseries", save_format="png")

        ax = saved[0]["fig"].axes[0]
        assert len(ax.get_lines()) == 2 and len(ax.collections) == 0
        assert "uncertainty" not in saved[0]["metadata"]["Description"]

    def test_seasonal_cycle_figure(self, saved):
        """The seasonal cycle is drawn over the twelve months, with its reference."""
        psi = PlotSeaIce(monthly_models=self.cycle, monthly_ref=[self.cycle_ref], dpi=DPI)
        psi.plot_seaice(plot_type="seasonalcycle", save_format="png")

        call = saved[0]
        assert call["diagnostic_product"] == "seasonalcycle"
        assert call["metadata"]["Description"].startswith("Seasonal cycle of sea ice extent integrated over")
        for ax in call["fig"].axes:
            assert ax.get_xlim() == (0.5, 12.5)
            assert [text.get_text() for text in ax.get_legend().get_texts()] == ["FESOM hpz3", "OBS hpz3"]
            assert len(ax.get_lines()) == 2

    def test_models_are_listed_once_each(self, saved):
        """With two models the caption names both and the file is saved for both."""
        psi = PlotSeaIce(monthly_models=[self.monthly, relabel(self.monthly, "OTHER")], dpi=DPI)
        psi.plot_seaice(plot_type="timeseries", save_format="png")

        assert f"for FESOM hpz3 {PERIOD} and OTHER hpz3 {PERIOD} " in saved[0]["metadata"]["Description"]
        assert saved[0]["saver"].model == ["FESOM", "OTHER"]

    def test_no_data_draws_nothing(self, saved):
        """Without model data there is no figure to save."""
        PlotSeaIce(dpi=DPI).plot_seaice(plot_type="seasonalcycle", save_format="png")
        assert saved == []

    def test_invalid_regions_type_raises(self):
        """regions_to_plot must be a list of names."""
        with pytest.raises(TypeError, match="to be a list"):
            PlotSeaIce(monthly_models=self.monthly, regions_to_plot="arctic")
        with pytest.raises(TypeError, match="to be a list"):
            PlotSeaIce(monthly_models=self.monthly, regions_to_plot=123)

    def test_invalid_plot_type_raises(self):
        """Only time series and seasonal cycles exist."""
        with pytest.raises(ValueError, match="Invalid plot_type"):
            PlotSeaIce(monthly_models=self.monthly, dpi=DPI).plot_seaice(plot_type="bad_type", save_format=[])

    def test_plot_saves_outputs(self, tmp_path):
        """The files are named after diagnostic, product, catalog, model, experiment, method and regions."""
        psi = PlotSeaIce(monthly_models=self.monthly, monthly_ref=self.ref, dpi=DPI, outputdir=str(tmp_path))
        psi.plot_seaice(plot_type="timeseries", save_format=["png", "pdf"])

        name = "seaice.timeseries.ci.FESOM.hpz3.r1.extent.arctic_antarctic"
        assert os.path.isfile(tmp_path / "png" / f"{name}.png")
        assert os.path.isfile(tmp_path / "pdf" / f"{name}.pdf")
