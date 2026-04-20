"""Tests for the Histogram CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.histogram.cli_histogram import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.histogram.cli_histogram"

BASE_DICT = {
    "run": True,
    "variables": ["2t"],
    "formulae": [],
    "bins": 50,
    "weighted": True,
    "density": True,
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
    args = parse_arguments(["--model", "IFS", "--nworkers", "4"])
    assert args.model == "IFS"
    assert args.nworkers == 4
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked Histogram and PlotHistogram."""

    @pytest.fixture
    def mock_hist(self, mocker):
        """
        Patch Histogram and PlotHistogram at the CLI module path.
        Returns (mock_hist_cls, mock_plot_cls).
        """
        mock_hist_cls = mocker.patch(f"{CLI_MODULE}.Histogram")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotHistogram")
        return mock_hist_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_hist):
        """When run=False, no Histogram instance should be created."""
        mock_hist_cls, mock_plot_cls = mock_hist
        config_file = build_config({"histogram": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_hist_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline_with_variables_and_formulae(self, build_config, mock_cluster, mock_hist):
        """
        Verify the full execution flow: Histogram is created for each
        (variable, dataset/reference) pair, run() is called, and
        PlotHistogram.run is invoked once per variable.
        """
        mock_hist_cls, mock_plot_cls = mock_hist
        mock_hist_instance = mock_hist_cls.return_value
        config_file = build_config(
            {
                "histogram": {
                    **BASE_DICT,
                    "variables": ["2t"],
                    "formulae": ["net_toa"],
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 2 vars * (1 dataset + 1 reference) = 4 Histogram instantiations
        assert mock_hist_cls.call_count == 4
        assert mock_hist_instance.run.call_count == 4

        # First run is for the first dataset of '2t' (formula=False)
        first_run = mock_hist_instance.run.call_args_list[0]
        assert first_run.kwargs["var"] == "2t"
        assert first_run.kwargs["formula"] is False

        # Third run is for the first dataset of 'net_toa' (formula=True)
        third_run = mock_hist_instance.run.call_args_list[2]
        assert third_run.kwargs["var"] == "net_toa"
        assert third_run.kwargs["formula"] is True

        # PlotHistogram created once per variable
        assert mock_plot_cls.call_count == 2
        assert mock_plot_cls.return_value.run.call_count == 2

    def test_dict_variable_uses_embedded_config(self, build_config, mock_cluster, mock_hist):
        """When a variable is a dict (not a string), its fields flow into Histogram kwargs."""
        mock_hist_cls, _ = mock_hist
        config_file = build_config(
            {
                "histogram": {
                    **BASE_DICT,
                    "variables": [{"name": "2t", "units": "K", "long_name": "Temperature"}],
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # dataset + reference call for the single variable
        assert mock_hist_cls.call_count == 2
        first_run = mock_hist_cls.return_value.run.call_args_list[0]
        assert first_run.kwargs["var"] == "2t"
        assert first_run.kwargs["units"] == "K"
        assert first_run.kwargs["long_name"] == "Temperature"
