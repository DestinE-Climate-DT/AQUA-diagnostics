"""Tests for the Ocean Drift CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_drift.cli_ocean_drift import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_drift.cli_ocean_drift"

BASE_BLOCK = {
    "run": True,
    "regions": ["global_ocean"],
    "diagnostic_name": "ocean_drift",
    "var": ["thetao"],
    "dim_mean": ["lat", "lon"],
    "vert_coord": "lev",
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    """Verify parse_arguments parses CLI options."""
    args = parse_arguments(["--model", "IFS", "--nworkers", "2"])
    assert args.model == "IFS"
    assert args.nworkers == 2
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestMainExecutionFlow:
    """Test main() execution flow with mocked Hovmoller and PlotHovmoller."""

    @pytest.fixture
    def mock_od(self, mocker):
        mock_hov_cls = mocker.patch(f"{CLI_MODULE}.Hovmoller")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotHovmoller")
        return mock_hov_cls, mock_plot_cls

    def test_hovmoller_disabled_skips_processing(self, build_config, mock_cluster, mock_od):
        """When run=False, diagnostic and plot classes are not instantiated."""
        mock_hov_cls, mock_plot_cls = mock_od
        config_file = build_config({"ocean_drift": {"hovmoller": {"run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_hovmoller_full_pipeline(self, build_config, mock_cluster, mock_od):
        """With run=True, Hovmoller.run and both plotting methods are called."""
        mock_hov_cls, mock_plot_cls = mock_od
        mock_hov_instance = mock_hov_cls.return_value
        mock_hov_instance.processed_data_list = [object()]
        config_file = build_config({"ocean_drift": {"hovmoller": BASE_BLOCK}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_called_once()
        mock_hov_instance.run.assert_called_once()
        run_call = mock_hov_instance.run.call_args
        assert run_call.kwargs["region"] == "global_ocean"
        assert run_call.kwargs["var"] == ["thetao"]

        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot_hovmoller.assert_called_once()
        mock_plot_cls.return_value.plot_timeseries.assert_called_once()
