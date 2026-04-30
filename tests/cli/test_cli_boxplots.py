"""Tests for the Boxplots CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.boxplots.cli_boxplots import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.boxplots.cli_boxplots"

# One variable group with a couple of plot kwargs. The CLI treats every
# item in 'variables' as a group dict with 'vars' plus any plot kwargs.
BASE_BP = {
    "run": True,
    "variables": [
        {"vars": ["2t"], "ylabel": "temperature"},
    ],
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


# ======================================================================
# Argument parsing
# ======================================================================
def test_parse_arguments_cli_options():
    """
    Verify parse_arguments parses CLI options.
    Detailed flag coverage lives in test_base_util.py
    """
    args = parse_arguments(["--model", "IFS", "--loglevel", "WARNING"])
    assert args.model == "IFS"
    assert args.loglevel == "WARNING"
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked Boxplots and PlotBoxplots."""

    @pytest.fixture
    def mock_bp(self, mocker):
        """
        Patch Boxplots and PlotBoxplots at the CLI module path.
        Returns (mock_bp_cls, mock_plot_cls).
        """
        mock_bp_cls = mocker.patch(f"{CLI_MODULE}.Boxplots")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotBoxplots")
        return mock_bp_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_bp):
        """When run=False, neither Boxplots nor PlotBoxplots is instantiated."""
        mock_bp_cls, mock_plot_cls = mock_bp
        config_file = build_config({"boxplots": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_bp_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mock_bp):
        """
        Verify the full execution flow: Boxplots is created for each dataset
        and each reference, run() is called, and PlotBoxplots.plot_boxplots
        is invoked once per variable group.
        """
        mock_bp_cls, mock_plot_cls = mock_bp
        mock_bp_instance = mock_bp_cls.return_value
        config_file = build_config({"boxplots": BASE_BP})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 1 dataset + 1 reference = 2 Boxplots instantiations per group
        assert mock_bp_cls.call_count == 2
        assert mock_bp_instance.run.call_count == 2
        first_run = mock_bp_instance.run.call_args_list[0]
        assert first_run.kwargs["var"] == ["2t"]

        mock_plot_cls.assert_called_once()
        plot_call = mock_plot_cls.return_value.plot_boxplots.call_args
        assert plot_call.kwargs["var"] == ["2t"]
        assert plot_call.kwargs["ylabel"] == "temperature"

    def test_not_enough_data_skips_dataset(self, build_config, mock_cluster, mock_bp, mocker):
        """When run() raises NotEnoughDataError, that dataset is skipped but the plot still runs."""
        from aqua.core.exceptions import NotEnoughDataError

        mock_bp_cls, mock_plot_cls = mock_bp
        # First call (dataset) raises, second (reference) succeeds
        mock_bp_cls.return_value.run.side_effect = [NotEnoughDataError("no data"), mocker.DEFAULT]
        config_file = build_config({"boxplots": BASE_BP})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # With no valid datasets (fldmeans empty), plot_boxplots must NOT be called.
        mock_plot_cls.return_value.plot_boxplots.assert_not_called()
