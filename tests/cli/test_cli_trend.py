"""Tests for the generic trend CLI and shared ocean/generic configuration."""

from pathlib import Path

import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.core.util import dump_yaml, load_yaml
from aqua.diagnostics.base import DiagnosticCLI
from aqua.diagnostics.ocean_trends.cli_ocean_trends import main as ocean_main
from aqua.diagnostics.trends import Trends
from aqua.diagnostics.trends.cli_trend import main

CLI_MODULE = "aqua.diagnostics.trends.cli_trend"
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = {"run": True, "var": ["tas"], "regions": [None, "tropics"], "plot": True}
pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


@pytest.fixture
def mock_trend(mocker):
    """Keep data access mocked while retaining real xarray output shapes."""
    trend_cls = mocker.patch(f"{CLI_MODULE}.Trends")
    plot_cls = mocker.patch(f"{CLI_MODULE}.PlotTrends")
    coefficients = xr.Dataset(
        {"tas": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [-30, 30], "lon": [0, 90]},
    )
    instance = trend_cls.return_value
    instance.diagnostic_name = "trends"
    instance.compute_trend.return_value = coefficients
    instance.select_region.side_effect = lambda data, region: {"data": data.copy(), "region": region}
    instance._set_region.side_effect = lambda region: ("Tropics" if region == "tropics" else region, None, None)
    instance.load_netcdf.return_value = coefficients
    return trend_cls, plot_cls


def test_disabled_generic_ignores_enabled_ocean(build_config, mock_cluster, mock_trend, mocker):
    default_path = mocker.patch(f"{CLI_MODULE}.get_diagnostic_configpath")
    config = build_config({"trends": {**BASE_CONFIG, "run": False}, "ocean_trends": {"multilevel": {"run": True}}})
    main(["--config", config])
    trend_cls, plot_cls = mock_trend
    trend_cls.assert_not_called()
    plot_cls.assert_not_called()
    mock_cluster[0].assert_not_called()
    default_path.assert_not_called()


@pytest.mark.parametrize("dim_mean", [None, "lon"])
def test_regional_computation_and_cli_overrides(build_config, mock_cluster, mock_trend, dim_mean):
    config = build_config({"trends": {**BASE_CONFIG, "dim_mean": dim_mean, "plot": dim_mean is None}})
    main(["--config", config, "--model", "IFS", "--realization", "r2", "--regrid", "r100"])
    trend_cls, plot_cls = mock_trend
    assert trend_cls.call_args.kwargs["model"] == "IFS"
    assert trend_cls.call_args.kwargs["regrid"] == "r100"
    instance = trend_cls.return_value
    instance.retrieve.assert_called_once_with(var=["tas"], reader_kwargs={"realization": "r2"})
    if dim_mean is None:
        instance.compute_trend.assert_called_once_with()
        assert [call.kwargs["region"] for call in instance.select_region.call_args_list] == ["tropics"]
    else:
        assert [call.kwargs for call in instance.compute_trend.call_args_list] == [
            {"region": None, "dim_mean": "lon"},
            {"region": "tropics", "dim_mean": "lon"},
        ]
        instance.select_region.assert_not_called()
    assert instance.save_netcdf.call_count == 2
    assert plot_cls.return_value.plot_trend.call_count == (2 if dim_mean is None else 0)


def test_multiple_datasets_are_independent(build_config, mock_cluster, mock_trend, mocker):
    datasets = [{"catalog": "ci", "model": model, "exp": "exp", "source": "monthly"} for model in ["model1", "model2"]]
    trend_cls, _ = mock_trend
    first = trend_cls.return_value
    second = mocker.MagicMock()
    second.diagnostic_name = "trends"
    second.compute_trend.return_value = first.compute_trend.return_value
    second.select_region.return_value = {"data": first.compute_trend.return_value}
    trend_cls.side_effect = [first, second]
    config = build_config({"trends": {**BASE_CONFIG, "regions": [None]}}, datasets=datasets)
    main(["--config", config])
    assert [call.kwargs["model"] for call in trend_cls.call_args_list] == ["model1", "model2"]
    for instance in [first, second]:
        instance.retrieve.assert_called_once_with(var=["tas"], reader_kwargs={})
        instance.compute_trend.assert_called_once_with()
        instance.save_netcdf.assert_called_once()
        instance.select_region.assert_not_called()


def test_plotting_without_saving_coefficients(build_config, mock_cluster, mock_trend):
    config = build_config({"trends": BASE_CONFIG}, output_overrides={"save_netcdf": False})
    main(["--config", config])
    trend_cls, plot_cls = mock_trend
    trend_cls.return_value.save_netcdf.assert_not_called()
    assert plot_cls.return_value.plot_trend.call_count == 2


def test_plot_only_locates_regional_mean_files(build_config, mock_cluster, mock_trend):
    # Real horizontal-file plotting is covered by test_plot_only_reads_real_coefficients.
    config = build_config(
        {"trends": {**BASE_CONFIG, "dim_mean": "lon", "plot": False}},
        output_overrides={"plot_only": True},
    )
    main(["--config", config, "--realization", "r3"])
    trend_cls, plot_cls = mock_trend
    instance = trend_cls.return_value
    instance.retrieve.assert_not_called()
    instance.compute_trend.assert_not_called()
    instance.save_netcdf.assert_not_called()
    assert instance.realization == "r3"
    assert [call.kwargs["extra_keys"] for call in instance.load_netcdf.call_args_list] == [
        {"dim_mean": "lon"},
        {"dim_mean": "lon", "region": "Tropics"},
    ]
    plot_cls.assert_not_called()


def test_retrieval_failure_closes_cluster(build_config, mock_cluster, mock_trend, mocker):
    config = build_config({"trends": BASE_CONFIG})
    mock_trend[0].return_value.retrieve.side_effect = NotEnoughDataError("Unavailable input")
    close = mocker.spy(DiagnosticCLI, "close_dask_cluster")
    with pytest.raises(NotEnoughDataError, match="Unavailable input"):
        main(["--config", config])
    close.assert_called_once()


def test_missing_plot_only_output_fails(build_config, mock_cluster, mock_trend, mocker):
    config = build_config({"trends": BASE_CONFIG}, output_overrides={"plot_only": True})
    mock_trend[0].return_value.load_netcdf.return_value = None
    close = mocker.spy(DiagnosticCLI, "close_dask_cluster")
    with pytest.raises(NoDataError, match="No saved trend"):
        main(["--config", config])
    close.assert_called_once()


def test_non_horizontal_output_can_be_saved_without_plotting(build_config, mock_cluster, mock_trend):
    coefficients = mock_trend[0].return_value.compute_trend.return_value.expand_dims(depth=[10, 100])
    mock_trend[0].return_value.compute_trend.return_value = coefficients
    config = build_config({"trends": {**BASE_CONFIG, "plot": False}})
    main(["--config", config])
    assert mock_trend[0].return_value.save_netcdf.call_count == 2
    mock_trend[1].assert_not_called()


def test_default_shared_template_loads(mock_cluster, mock_trend, mocker):
    config_dir = REPO_ROOT / "aqua/diagnostics/templates/collections"
    default_path = mocker.patch(f"{CLI_MODULE}.get_diagnostic_configpath", return_value=str(config_dir))
    main([])
    default_path.assert_called_once_with("trends", folder="templates", loglevel="WARNING")
    mock_trend[0].return_value.retrieve.assert_called_once_with(var=["thetao", "so"], reader_kwargs={"chunks": {"time": 12}})
    mock_trend[1].assert_not_called()


def test_separate_2d_template_runs_generic_maps(mock_cluster, mock_trend):
    config = REPO_ROOT / "aqua/diagnostics/templates/collections/config-trends-2d.yaml"
    main(["--config", str(config)])
    trend_cls, plot_cls = mock_trend
    assert trend_cls.call_args.kwargs["source"] == "monthly-2d"
    assert trend_cls.call_args.kwargs["regrid"] == "r100"
    trend_cls.return_value.retrieve.assert_called_once_with(var=["siconc", "sithick"], reader_kwargs={"chunks": {"time": 12}})
    plot_cls.return_value.plot_trend.assert_called_once()


@pytest.mark.parametrize("kind,suffix", [("jinja", "j2"), ("legacy", "yaml")], ids=["jinja", "legacy"])
@pytest.mark.parametrize(
    "collection,variables",
    [("atmosphere2d", ["2t", "msl"]), ("water_cycle", ["tprate"]), ("ocean2d", ["tos"])],
)
def test_collection_uses_model_period_and_output_prefix(
    mock_cluster, mock_trend, tmp_path, kind, suffix, collection, variables
):
    config = REPO_ROOT / f"aqua/diagnostics/config/collections/{kind}/{collection}/config-{collection}-trends.{suffix}"
    if kind == "jinja":
        # render the template with the default experiment kind, as aqua analysis does
        kinds = load_yaml(str(REPO_ROOT / "aqua/diagnostics/config/analysis/climatedt-experiment-kind.yaml"))
        rendered = tmp_path / f"config-{collection}-trends.yaml"
        dump_yaml(str(rendered), load_yaml(str(config), definitions=kinds["default"]))
        config = rendered
    trend_cls, plot_cls = mock_trend
    trend_cls.return_value.diagnostic_name = collection
    main(
        [
            "--config",
            str(config),
            "--catalog",
            "climatedt-gen2",
            "--model",
            "IFS-NEMO-5km",
            "--exp",
            "baseline-hist",
            "--source",
            "lra-r100-monthly",
            "--outputdir",
            str(tmp_path),
            "--startdate",
            "2001-01-01",
            "--enddate",
            "2005-12-31",
        ]
    )
    assert trend_cls.call_args.kwargs == {
        "catalog": "climatedt-gen2",
        "model": "IFS-NEMO-5km",
        "exp": "baseline-hist",
        "source": "lra-r100-monthly",
        "regrid": None,
        "startdate": "2001-01-01",
        "enddate": "2005-12-31",
        "diagnostic_name": collection,
        "loglevel": "WARNING",
    }
    instance = trend_cls.return_value
    instance.retrieve.assert_called_once_with(var=variables, reader_kwargs={"chunks": {"time": 12}})
    instance.compute_trend.assert_called_once_with()
    instance.select_region.assert_not_called()
    instance.save_netcdf.assert_called_once()
    assert plot_cls.call_args.kwargs["diagnostic_name"] == collection
    plot_cls.return_value.plot_trend.assert_called_once()


def test_shared_template_remains_usable_by_ocean_cli(mock_cluster, mock_trend, mocker):
    ocean_cls = mocker.patch("aqua.diagnostics.ocean_trends.cli_ocean_trends.Trends")
    mocker.patch("aqua.diagnostics.ocean_trends.cli_ocean_trends.PlotTrends")
    data = mock_trend[0].return_value.compute_trend.return_value
    ocean_cls.return_value.select_region.return_value = (data, "Global Ocean")
    config = REPO_ROOT / "aqua/diagnostics/templates/collections/config-trends.yaml"
    ocean_main(["--config", str(config)])
    ocean_cls.return_value.run.assert_called_once()
    assert ocean_cls.return_value.run.call_args.kwargs["var"] == ["thetao", "so"]
    mock_trend[0].assert_not_called()


def test_plot_only_reads_real_coefficients(build_config, mock_cluster, mocker):
    """A saved file is located by its identity, plotted, and left unchanged."""
    dataset = {"catalog": "ci", "model": "model", "exp": "exp", "source": "monthly"}
    config_path = build_config(
        {"trends": {**BASE_CONFIG, "regions": [None]}},
        datasets=[dataset],
        output_overrides={"plot_only": True, "save_format": ["png"]},
    )
    outputdir = load_yaml(config_path)["output"]["outputdir"]
    diagnostic = Trends(**dataset)
    diagnostic.realization = "r2"
    coefficients = xr.Dataset(
        {"tas": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [-30, 30], "lon": [0, 90]},
        attrs={"AQUA_startdate": "2000-03-01", "AQUA_enddate": "2001-06-30", "history": "fit history"},
    )
    coefficients.tas.attrs = {
        "AQUA_catalog": "ci",
        "AQUA_model": "model",
        "AQUA_exp": "exp",
        "AQUA_realization": "r2",
        "short_name": "tas",
        "long_name": "Temperature",
        "units": "K/year",
    }
    diagnostic.save_netcdf(data=coefficients, outputdir=outputdir)
    (saved_path,) = Path(outputdir).glob("netcdf/*.nc")
    original_bytes = saved_path.read_bytes()
    retrieve = mocker.patch.object(Trends, "retrieve", side_effect=AssertionError("plot-only retrieved source data"))
    compute = mocker.patch.object(Trends, "compute_trend", side_effect=AssertionError("plot-only computed a new trend"))
    main(["--config", config_path, "--realization", "r2"])
    retrieve.assert_not_called()
    compute.assert_not_called()
    (figure_path,) = Path(outputdir).glob("png/*.png")
    assert figure_path.stat().st_size > 0
    assert saved_path.read_bytes() == original_bytes
