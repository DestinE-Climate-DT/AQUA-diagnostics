"""Tests for the GlobalBiases CLI (parse_arguments + main orchestration)."""

from unittest.mock import MagicMock

import pytest

from aqua.diagnostics.global_biases.cli_global_biases import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.global_biases.cli_global_biases"

# Base configuration dictionary for GlobalBiases diagnostic
BASE_SET = {
    "run": True,
    "variables": ["2t"],
    "formulae": [],
    "params": {"default": {}},
    "plot_params": {"default": {}},
}


def _mock_data(*var_names):
    """Create a dict mimicking GlobalBiases.data with no plev dimension."""
    return {v: MagicMock(dims=()) for v in var_names}


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
    """Test main() execution flow with mocked GlobalBiases and PlotGlobalBiases."""

    @pytest.fixture
    def mock_gb(self, mocker):
        """
        Patch GlobalBiases and PlotGlobalBiases with sensible defaults.
        Returns (mock_gb_cls, mock_plot_cls). The GlobalBiases instance
        is pre-configured with .data for "2t" and an empty .climatology.
        """
        mock_gb_cls = mocker.patch(f"{CLI_MODULE}.GlobalBiases")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotGlobalBiases")
        mock_gb_cls.return_value.data = _mock_data("2t")
        mock_gb_cls.return_value.climatology = {}
        return mock_gb_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_gb):
        """When run=False, no GlobalBiases instance should be created."""
        mock_gb_cls, _ = mock_gb
        config_file = build_config({"globalbiases": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_gb_cls.assert_not_called()

    def test_single_variable_full_pipeline(self, build_config, mock_cluster, mock_gb):
        """
        With one variable and run=True, verify the full execution flow:
        GlobalBiases created for dataset+reference, retrieve and
        compute_climatology called, PlotGlobalBiases.plot_bias called.
        """
        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_instance = mock_gb_cls.return_value
        config_file = build_config({"globalbiases": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_gb_cls.call_count == 2
        dataset_call, reference_call = mock_gb_cls.call_args_list

        assert dataset_call.kwargs["catalog"] == "test-catalog"
        assert dataset_call.kwargs["model"] == "TestModel"
        assert reference_call.kwargs["catalog"] == "ref-catalog"
        assert reference_call.kwargs["model"] == "RefModel"

        assert mock_gb_instance.retrieve.call_count == 2
        retrieve_call = mock_gb_instance.retrieve.call_args_list[0]
        assert retrieve_call.kwargs["var"] == "2t"
        assert retrieve_call.kwargs["formula"] is False

        assert mock_gb_instance.compute_climatology.call_count == 2

        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot_bias.assert_called_once()

    def test_retrieve_nodata_skips_variable(self, build_config, mock_cluster, mock_gb):
        """If retrieve raises NoDataError, the variable is skipped gracefully."""
        from aqua.core.exceptions import NoDataError

        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_cls.return_value.retrieve.side_effect = NoDataError("no data")
        config_file = build_config({"globalbiases": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_gb_cls.return_value.compute_climatology.assert_not_called()
        mock_plot_cls.return_value.plot_bias.assert_not_called()

    def test_seasonal_bias_plotted_when_seasons_enabled(self, build_config, mock_cluster, mock_gb):
        """When seasons=True, plot_seasonal_bias should be called."""
        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_cls.return_value.seasonal_climatology = {}
        config_file = build_config(
            {
                "globalbiases": {
                    **BASE_SET,
                    "params": {"default": {"seasons": True, "seasons_stat": "mean"}},
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_plot_cls.return_value.plot_seasonal_bias.assert_called_once()

    def test_multiple_variables_and_formulae(self, build_config, mock_cluster, mock_gb):
        """Both variables and formulae are processed in order."""
        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_instance = mock_gb_cls.return_value
        mock_gb_instance.data = _mock_data("2t", "tprate", "net_toa")
        config_file = build_config(
            {
                "globalbiases": {
                    **BASE_SET,
                    "variables": ["2t", "tprate"],
                    "formulae": ["net_toa"],
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 3 vars * 2 (dataset + reference) = 6 retrieve calls
        assert mock_gb_instance.retrieve.call_count == 6

        # 3 vars * 1 plot_bias call each = 3
        assert mock_plot_cls.return_value.plot_bias.call_count == 3

        # Check formula flag for the third pair of retrieve calls (net_toa)
        fifth_call = mock_gb_instance.retrieve.call_args_list[4]
        assert fifth_call.kwargs["var"] == "net_toa"
        assert fifth_call.kwargs["formula"] is True

    def test_custom_diagnostic_name_passed_to_globalbiases(self, build_config, mock_cluster, mock_gb):
        """A custom diagnostic_name in config should be forwarded to GlobalBiases."""
        mock_gb_cls, _ = mock_gb
        config_file = build_config(
            {
                "globalbiases": {
                    **BASE_SET,
                    "diagnostic_name": "my_custom_gb",
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_gb_cls.call_args_list[0].kwargs["diagnostic"] == "my_custom_gb"
