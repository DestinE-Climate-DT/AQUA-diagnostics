"""Tests for the Ocean Drift CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_drift.cli_ocean_drift import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_drift.cli_ocean_drift"

BASE_DRIFT = {
    "hovmoller": {
        "run": True,
        "regions": ["global_ocean"],
        "diagnostic_name": "ocean_drift",
        "var": ["thetao"],
        "dim_mean": ["lat", "lon"],
        "vert_coord": "lev",
    }
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
        config_file = build_config({"ocean_drift": {"hovmoller": {**BASE_DRIFT["hovmoller"], "run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_hovmoller_full_pipeline(self, build_config, mock_cluster, mock_od):
        """With run=True, Hovmoller.run and both plotting methods are called."""
        mock_hov_cls, mock_plot_cls = mock_od
        mock_hov_instance = mock_hov_cls.return_value
        mock_hov_instance.processed_data = {"global_ocean": [object()]}
        config_file = build_config({"ocean_drift": BASE_DRIFT})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hov_cls.assert_called_once()
        mock_hov_instance.run.assert_called_once()
        run_call = mock_hov_instance.run.call_args
        assert run_call.kwargs["regions"] == ["global_ocean"]
        assert run_call.kwargs["var"] == ["thetao"]

        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot_hovmoller.assert_called_once()
        mock_plot_cls.return_value.plot_timeseries.assert_called_once()

    @pytest.mark.parametrize(
        "output, runs, saves, loads",
        [
            ({}, True, True, True),  # compute, save and plot from the files just written
            ({"save_netcdf": False}, True, False, False),  # compute and plot from memory only
            ({"plot_only": True}, False, None, True),  # no evaluation, plot from previous files
        ],
    )
    def test_plot_only_and_save_netcdf(self, build_config, mock_cluster, mock_od, output, runs, saves, loads):
        """plot_only skips the evaluation, save_netcdf only decides whether the results are written."""
        mock_hov_cls, mock_plot_cls = mock_od
        mock_hov_instance = mock_hov_cls.return_value
        mock_hov_instance.processed_data = {"global_ocean": [object()]}
        config_file = build_config({"ocean_drift": BASE_DRIFT}, output_overrides=output)

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_hov_instance.run.call_count == (1 if runs else 0)
        for call in mock_hov_instance.run.call_args_list:
            assert call.kwargs["save_netcdf"] is saves
        # Loading unsaved results would replace them with older files on disk
        assert mock_hov_instance.load.call_count == (1 if loads else 0)
        # load rebuilds the filenames from the same regions and anomaly reference that run uses
        for call in mock_hov_instance.load.call_args_list:
            assert call.kwargs["regions"] == ["global_ocean"]
            assert call.kwargs["anomaly_ref"] == "t0"
        mock_plot_cls.assert_called_once()
