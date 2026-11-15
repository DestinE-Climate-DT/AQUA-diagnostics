"""Tests for the PlotBase class."""

import matplotlib.pyplot as plt
import pytest
import xarray as xr
from PIL import Image

from aqua.diagnostics.base import PlotBase

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def result(catalog="ci", model="IFS", exp="test-tco79", **attrs):
    """A result as a diagnostic leaves it: a DataArray with AQUA_* attributes."""
    attrs = {"catalog": catalog, "model": model, "exp": exp, **attrs}
    return xr.DataArray([1.0, 2.0], dims="time", attrs={f"AQUA_{key}": value for key, value in attrs.items()})


ERA5 = {"catalog": "obs", "model": "ERA5", "exp": "era5"}


@pytest.fixture
def plot(tmp_path):
    """A PlotBase of one dataset, saving small PNG files only."""
    plot = PlotBase("dummy", outputdir=tmp_path, save_format="png", dpi=50)
    plot.get_data_info(result())
    return plot


def test_get_data_info_reads_datasets_and_references():
    data = [
        result(region="Europe", startdate="1990-01-01", enddate="1999-12-31"),
        result(model="ICON", exp="test-r2b0", realization="r2"),
    ]
    plot = PlotBase("dummy")
    plot.get_data_info(data, result(**ERA5, startdate="1980-01-01", enddate="2010-12-31"))

    assert (plot.catalogs, plot.models, plot.exps) == (["ci", "ci"], ["IFS", "ICON"], ["test-tco79", "test-r2b0"])
    assert plot.realizations == ["r1", "r2"]
    assert (plot.startdates, plot.enddates) == (["1990-01-01", None], ["1999-12-31", None])
    assert plot.region == "Europe"
    assert (plot.ref_catalogs, plot.ref_models, plot.ref_exps) == (["obs"], ["ERA5"], ["era5"])
    assert (plot.ref_startdates, plot.ref_enddates) == (["1980-01-01"], ["2010-12-31"])


def test_get_data_info_completes_a_dataset_with_its_first_variable():
    data = xr.Dataset({"tas": result(region="Global")}, attrs={"AQUA_region": "Europe", "AQUA_startdate": "1990-01-01"})
    plot = PlotBase("dummy")
    plot.get_data_info(data)

    assert (plot.catalogs, plot.models, plot.exps) == (["ci"], ["IFS"], ["test-tco79"])
    assert (plot.region, plot.startdates) == ("Europe", ["1990-01-01"])


@pytest.mark.parametrize(
    "data,ref_data,expected",
    [
        (result(), None, "dummy.prod.ci.IFS.test-tco79.r1"),
        (result(realization="r2"), result(**ERA5), "dummy.prod.ci.IFS.test-tco79.r2.obs.ERA5.era5"),
        ([result(), result(model="ICON", exp="test-r2b0")], result(**ERA5), "dummy.prod.multimodel.obs.ERA5.era5"),
    ],
    ids=["single", "reference", "multimodel"],
)
def test_file_name(data, ref_data, expected):
    plot = PlotBase("dummy")
    plot.get_data_info(data, ref_data)
    assert plot.outputsaver.generate_name("prod") == expected


def test_outputsaver_follows_the_datasets(plot):
    saver = plot.outputsaver
    assert plot.outputsaver is saver

    plot.get_data_info(result(model="ICON"))
    assert plot.outputsaver is not saver
    assert plot.outputsaver.model == "ICON"


def test_set_title_fills_the_datasets_and_takes_titlebuilder_arguments():
    plot = PlotBase("dummy")
    plot.get_data_info(result(region="Europe"), result(**ERA5))

    title = "Bias of temperature in Europe for IFS test-tco79 relative to ERA5 era5"
    assert plot.set_title(diagnostic="Bias", variable="Temperature") == title
    assert plot.set_title(diagnostic="Bias", variable="Temperature", max_chars=30) == (
        "Bias of temperature in Europe\nfor IFS test-tco79\nrelative to ERA5 era5"
    )
    # A keyword argument adds a field or replaces a default
    title = plot.set_title(diagnostic="Bias", catalog=plot.catalogs, startyear=1990, ref_model=None, ref_exp=None)
    assert title == "Bias in Europe for ci IFS test-tco79 1990"


def test_set_title_leaves_out_datasets_without_attributes():
    plot = PlotBase("dummy")
    plot.get_data_info(xr.DataArray([1.0]))
    assert plot.set_title(diagnostic="Trend", catalog=plot.catalogs) == "Trend"


def test_save_plot_writes_the_description_and_closes_the_figure(plot, tmp_path):
    fig = plt.figure()
    plot.save_plot(fig, "prod", extra_keys={"var": "tas"}, description="A caption.", metadata={"note": "extra"})

    assert [folder.name for folder in tmp_path.iterdir()] == ["png"]
    (path,) = (tmp_path / "png").iterdir()
    assert path.name == "dummy.prod.ci.IFS.test-tco79.r1.tas.png"
    with Image.open(path) as image:
        assert (image.text["description"], image.text["note"]) == ("A caption.", "extra")
    assert not plt.fignum_exists(fig.number)


def test_save_plot_defaults_come_from_the_instance(plot, mocker):
    save = mocker.patch.object(plot.outputsaver, "save_figure")
    fig = plt.figure()

    plot.save_plot(fig, "prod", close=False)
    assert plt.fignum_exists(fig.number)
    kwargs = save.call_args.kwargs
    assert (kwargs["extension"], kwargs["rebuild"], kwargs["dpi"], kwargs["metadata"]) == ("png", True, 50, None)

    plot.save_plot(fig, "prod", rebuild=False, save_format="pdf", dpi=100)
    kwargs = save.call_args.kwargs
    assert (kwargs["extension"], kwargs["rebuild"], kwargs["dpi"]) == ("pdf", False, 100)


def test_save_plot_closes_the_figure_when_saving_fails(plot, mocker):
    mocker.patch.object(plot.outputsaver, "save_figure", side_effect=OSError("write failed"))
    fig = plt.figure()
    with pytest.raises(OSError, match="write failed"):
        plot.save_plot(fig, "prod")
    assert not plt.fignum_exists(fig.number)
