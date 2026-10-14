"""Single-variable trend maps preserve their input and reuse core graphics."""

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from aqua.core.graphics import plot_single_map
from aqua.core.reader import Reader
from aqua.diagnostics.trends import PlotTrends

PLOT_MODULE = "aqua.diagnostics.trends.plot_trends"
pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


@pytest.fixture(scope="module")
def trend_map_data():
    """Retrieve a small surface patch from the existing FESOM test data once."""
    reader = Reader(catalog="ci", model="FESOM", exp="hpz3", source="monthly-2d", regrid="r100", loglevel="ERROR")
    data = reader.regrid(reader.retrieve(var=["siconc", "sithick"]))
    data = data[["siconc", "sithick"]].isel(time=0, drop=True).sel(lat=slice(-80, -60), lon=slice(0, 20))
    data = data.isel(lat=slice(None, None, 5), lon=slice(None, None, 5))
    data = data.assign_attrs(AQUA_region="Southern Ocean", AQUA_startdate="1990-01-01", AQUA_enddate="1994-12-31")
    for name in data:
        data[name].attrs = {**data[name].attrs, "units": f"{data[name].attrs['units']}/year"}
    return data


def test_maps_write_per_variable_and_preserve_lazy_input(trend_map_data, tmp_path, mocker):
    before = trend_map_data.copy(deep=True)
    plots = PlotTrends(trend_map_data, outputdir=tmp_path)
    render = mocker.patch(f"{PLOT_MODULE}.plot_single_map", wraps=plot_single_map)
    save = mocker.spy(plots.outputsaver, "save_figure")
    plots.plot_trend(save_format="png", dpi=40)
    outputs = list((tmp_path / "png").glob("*.png"))
    assert len(outputs) == render.call_count == save.call_count == 2
    for name, call in zip(trend_map_data.data_vars, save.call_args_list):
        kwargs = call.kwargs
        assert kwargs["diagnostic_product"] == "map_trend"
        assert kwargs["extra_keys"] == {"var": name, "region": "Southern Ocean"}
        assert "1990-01" in kwargs["metadata"]["description"] and "1994-12" in kwargs["metadata"]["description"]
        assert not plt.fignum_exists(call.args[0].number)
        (path,) = [path for path in outputs if f".{name}." in path.name]
        assert path.stat().st_size > 0
    xr.testing.assert_identical(trend_map_data, before)
    assert trend_map_data.chunks


@pytest.mark.parametrize("invalid", ["missing_coordinates", "extra_dimension"])
def test_plotter_rejects_non_horizontal_fields(trend_map_data, tmp_path, mocker, invalid):
    data = trend_map_data.drop_vars("lat") if invalid == "missing_coordinates" else trend_map_data.expand_dims(depth=[10, 100])
    render = mocker.patch(f"{PLOT_MODULE}.plot_single_map")
    message = "latitude and longitude" if invalid == "missing_coordinates" else "horizontal field"
    with pytest.raises(ValueError, match=message):
        PlotTrends(data, outputdir=tmp_path).plot_trend()
    render.assert_not_called()


def test_write_failure_closes_figure_and_forwards_explicit_limits(trend_map_data, tmp_path, mocker):
    figures = []

    def render(**kwargs):
        fig, ax = plt.subplots()
        figures.append(fig)
        return fig, ax

    draw = mocker.patch(f"{PLOT_MODULE}.plot_single_map", side_effect=render)
    plots = PlotTrends(trend_map_data, outputdir=tmp_path)
    mocker.patch.object(plots.outputsaver, "save_figure", side_effect=OSError("write failed"))
    with pytest.raises(OSError, match="write failed"):
        plots.plot_trend(var="siconc", vmin=-2, vmax=3)
    assert (draw.call_args.kwargs["vmin"], draw.call_args.kwargs["vmax"], draw.call_args.kwargs["sym"]) == (-2, 3, False)
    assert len(figures) == 1 and not plt.fignum_exists(figures[0].number)
