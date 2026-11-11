"""Tests of Plot2DSeaIce on the sea ice climatologies of the FESOM test data (1991-1994, regridded to r100)."""

import os

import numpy as np
import pytest
import xarray as xr

from aqua.diagnostics import Plot2DSeaIce, SeaIce
from aqua.diagnostics.base import OutputSaver
from tests.shared_constants import DPI, LOGLEVEL

# pytestmark groups tests that run sequentially on the same worker to avoid conflicts
# These tests use setup_class with shared resources (data fetching, tmp files)
pytestmark = [pytest.mark.diagnostics, pytest.mark.xdist_group(name="diagnostic_setup_class")]

PERIOD = "(from 1991-01 to 1994-12)"


def climatology(regions, method, offset=0.0, model=None):
    """Climatology of the test dataset; with an offset and another model name it stands for a reference."""
    seaice = SeaIce(
        catalog="ci",
        model="FESOM",
        exp="hpz3",
        source="monthly-2d",
        regions=regions,
        startdate="1991-01-01",
        enddate="2000-01-01",
        regrid="r100",
        loglevel=LOGLEVEL,
    )
    seaice.run(method=method, save_netcdf=False)
    result = seaice.climatology + offset
    for name in result.data_vars:
        result[name].attrs["AQUA_model"] = model or "FESOM"
    return result


class TestPlot2DSeaIce:
    @classmethod
    def setup_class(cls):
        # the model has both regions in one dataset, the references one hemisphere each, as in the CLI
        cls.frac_model = climatology(["arctic", "antarctic"], "fraction")
        cls.frac_ref = [climatology([region], "fraction", 0.1, "OBS") for region in ("arctic", "antarctic")]
        cls.thick_model = climatology(["arctic", "antarctic"], "thickness")
        cls.thick_ref_arctic = climatology(["arctic"], "thickness", 0.5, "OBS")

    @pytest.fixture
    def saved(self, monkeypatch):
        """The calls to OutputSaver.save_figure, captured instead of writing files."""
        calls = []

        def fake_save_figure(self, fig, diagnostic_product, **kwargs):
            calls.append({"saver": self, "fig": fig, "diagnostic_product": diagnostic_product, **kwargs})

        monkeypatch.setattr(OutputSaver, "save_figure", fake_save_figure)
        return calls

    def test_fraction_bias_figures(self, saved):
        """One figure per region: a row of reference, model and difference maps for each month."""
        p2d = Plot2DSeaIce(ref=self.frac_ref, models=self.frac_model, dpi=DPI)
        assert p2d.regions_to_plot == ["Arctic", "Antarctic"]
        p2d.plot_2d_seaice(method="fraction", months=[3, 9], plot_ref_contour=True, save_format=["png"])

        assert [call["extra_keys"] for call in saved] == [
            {"method": "fraction", "region": "Arctic"},
            {"method": "fraction", "region": "Antarctic"},
        ]
        arctic, antarctic = saved
        assert arctic["diagnostic_product"] == "bias"
        assert (arctic["saver"].model, arctic["saver"].model_ref, arctic["saver"].exp_ref) == ("FESOM", "OBS", "hpz3")
        assert arctic["metadata"]["Description"] == (
            "Climatology and differences against the reference dataset for sea ice fraction in the Arctic region. "
            f"Model data is FESOM, experiment hpz3 {PERIOD}. Reference dataset is OBS, experiment hpz3 {PERIOD}. "
            "The red contour line represents where the sea ice fraction is equal to 0.2."
        )

        fig = arctic["fig"]
        assert [sub.get_suptitle() for sub in fig.subfigs] == [
            "Sea ice fraction in Arctic Month: March",
            "Sea ice fraction in Arctic Month: September",
        ]
        maps = [ax for ax in fig.axes if hasattr(ax, "projection")]
        assert [ax.get_title() for ax in maps[:3]] == [" OBS hpz3", " FESOM hpz3", " FESOM -  OBS"]
        assert len(maps) == 6 and len(fig.axes) == 12  # every map has its colorbar
        # each figure looks at the pole of its region: the model alone, which holds both, would not tell which
        assert maps[0].projection.proj4_params["lat_0"] == pytest.approx(89.5)
        assert antarctic["fig"].axes[0].projection.proj4_params["lat_0"] == pytest.approx(-89.5)

    def test_thickness_bias_skips_region_without_reference(self, saved):
        """Thickness maps are drawn for the Arctic only when the Antarctic has no reference."""
        p2d = Plot2DSeaIce(ref=self.thick_ref_arctic, models=self.thick_model, dpi=DPI)
        p2d.plot_2d_seaice(method="thickness", months=[3], save_format="png")

        assert [call["extra_keys"]["region"] for call in saved] == ["Arctic"]
        fig = saved[0]["fig"]
        assert "contour" not in saved[0]["metadata"]["Description"]
        colorbars = [ax for ax in fig.axes if not hasattr(ax, "projection")]
        assert [ax.get_ylabel() for ax in colorbars] == ["Sea-ice thickness [m]"] * 3

    @pytest.mark.parametrize(("region", "pole"), [("Arctic", 90), ("Antarctic", -90)])
    def test_azimuthal_view_is_a_disc_around_the_pole(self, saved, region, pole):
        """The maps are centred on the pole of the region and reach 50 degrees of latitude in its hemisphere."""
        p2d = Plot2DSeaIce(ref=self.frac_ref, models=self.frac_model, regions_to_plot=[region], dpi=DPI)
        p2d.plot_2d_seaice(method="fraction", months=[3], projection="azimuthal_equidistant", save_format="png")

        ax = saved[0]["fig"].axes[0]
        assert ax.projection.proj4_params["lat_0"] == pole
        # 40 degrees of meridian (about 111.3 km each) on both sides of the pole: a centre off the pole
        # collapsed the Arctic maps to a line, the window of the other hemisphere would be 3.5 times wider
        assert np.diff(ax.get_xlim())[0] == pytest.approx(2 * 40 * 111.3e3, rel=0.01)

    def test_plot_saves_outputs(self, tmp_path):
        """The file is named after model and reference, method and region."""
        p2d = Plot2DSeaIce(
            ref=self.frac_ref, models=self.frac_model, regions_to_plot=["Arctic"], outputdir=str(tmp_path), dpi=DPI
        )
        p2d.plot_2d_seaice(method="fraction", months=[3], save_format=["png", "pdf"])

        name = "seaice.bias.ci.FESOM.hpz3.r1.OBS.hpz3.fraction.arctic"
        assert os.path.isfile(tmp_path / "png" / f"{name}.png")
        assert os.path.isfile(tmp_path / "pdf" / f"{name}.pdf")

    def test_invalid_input_data_raises(self):
        """Only Datasets and DataArrays can be plotted."""
        with pytest.raises(TypeError, match="Unsupported type"):
            Plot2DSeaIce(models="string")
        with pytest.raises(TypeError, match="Unsupported type"):
            Plot2DSeaIce(models=[self.frac_model, 123])

    def test_bad_months_raise(self):
        """Months are integers from 1 to 12."""
        p2d = Plot2DSeaIce(ref=self.frac_ref, models=self.frac_model, dpi=DPI)
        with pytest.raises(ValueError, match="Invalid month value"):
            p2d.plot_2d_seaice(months=[0], save_format=[])
        with pytest.raises(TypeError):
            p2d.plot_2d_seaice(months=["Feb"], save_format=[])

    def test_bad_method_raises(self):
        """Only fraction and thickness have maps."""
        p2d = Plot2DSeaIce(ref=self.frac_ref, models=self.frac_model)
        with pytest.raises(ValueError, match="Unsupported method"):
            p2d.plot_2d_seaice(method="extent", save_format=[])

    def test_missing_reference_raises(self):
        """The difference maps cannot be drawn without a reference."""
        with pytest.raises(ValueError, match="Missing models or reference data"):
            Plot2DSeaIce(models=self.frac_model).plot_2d_seaice(method="fraction", save_format=[])

    def test_removed_arguments_raise(self):
        """The arguments of the previous interface are not silently swallowed by the map options."""
        p2d = Plot2DSeaIce(ref=self.frac_ref, models=self.frac_model)
        with pytest.raises(TypeError, match="plot_type"):
            p2d.plot_2d_seaice(plot_type="var", save_format=[])
        with pytest.raises(TypeError, match="projkw"):
            p2d.plot_2d_seaice(projkw={"projname": "orthographic"}, save_format=[])

    @pytest.mark.parametrize(("method", "expected"), [("fraction", [0.0, 0.0, 0.5]), ("thickness", [np.nan, np.nan, 0.5])])
    def test_mask_ice_at_mid_lats(self, method, expected):
        """Mid latitudes are filled with zero fraction, or left without thickness, whatever the data say there."""
        data = xr.DataArray(
            [[np.nan], [0.3], [0.5]],
            coords={"lat": [-30.0, 20.0, 80.0], "lon": [0.0]},
            dims=("lat", "lon"),
        )
        p2d = Plot2DSeaIce(loglevel=LOGLEVEL)
        p2d.method = method
        np.testing.assert_array_equal(p2d._mask_ice_at_mid_lats(data).values[:, 0], expected)
