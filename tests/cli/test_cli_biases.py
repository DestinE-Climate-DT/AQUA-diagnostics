"""Tests for the Biases CLI (parse_arguments + main orchestration)."""

from unittest.mock import MagicMock

import pytest

from aqua.diagnostics.biases.cli_biases import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.biases.cli_biases"

# Base configuration dictionary for Biases diagnostic
BASE_SET = {
    "run": True,
    "variables": ["2t"],
    "formulae": [],
    "params": {"default": {}},
    "plot_params": {"default": {}},
}


def _mock_data(*var_names):
    """Create a dict mimicking Biases.data with no plev dimension."""
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
    assert args.climatology is False
    assert parse_arguments(["--climatology"]).climatology is True

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked Climatology and PlotBias."""

    @pytest.fixture
    def mock_gb(self, mocker):
        """
        Patch Climatology and PlotBias with sensible defaults.
        Returns (mock_gb_cls, mock_plot_cls). The Climatology instance
        is pre-configured with .data for "2t" and an empty .climatology.
        """
        mock_gb_cls = mocker.patch(f"{CLI_MODULE}.Climatology")
        mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotBias")
        mock_gb_cls.return_value.data = _mock_data("2t")
        mock_gb_cls.return_value.climatology = {}
        return mock_gb_cls, mock_plot_cls

    def test_diagnostic_disabled_skips_processing(self, build_config, mock_cluster, mock_gb):
        """When run=False, no Climatology instance should be created."""
        mock_gb_cls, _ = mock_gb
        config_file = build_config({"biases": {"run": False}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_gb_cls.assert_not_called()

    def test_single_variable_full_pipeline(self, build_config, mock_cluster, mock_gb):
        """
        With one variable and run=True, verify the full execution flow:
        Climatology created for dataset+reference, retrieve and
        compute_climatology called, PlotBias.plot_bias called.
        """
        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_instance = mock_gb_cls.return_value
        config_file = build_config({"biases": BASE_SET})

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
        config_file = build_config({"biases": BASE_SET})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_gb_cls.return_value.compute_climatology.assert_not_called()
        mock_plot_cls.return_value.plot_bias.assert_not_called()

    def test_seasonal_bias_plotted_when_seasons_enabled(self, build_config, mock_cluster, mock_gb):
        """When seasons=True, plot_seasonal_bias should be called."""
        mock_gb_cls, mock_plot_cls = mock_gb
        mock_gb_cls.return_value.seasonal_climatology = {}
        config_file = build_config(
            {
                "biases": {
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
                "biases": {
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

    def test_custom_diagnostic_name_passed_to_climatology(self, build_config, mock_cluster, mock_gb):
        """A custom diagnostic_name in config should be forwarded to Climatology."""
        mock_gb_cls, _ = mock_gb
        config_file = build_config(
            {
                "biases": {
                    **BASE_SET,
                    "diagnostic_name": "my_custom_gb",
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        assert mock_gb_cls.call_args_list[0].kwargs["diagnostic"] == "my_custom_gb"


class TestClimatologyOnly:
    """Test main() in climatology-only mode (no reference or --climatology flag)."""

    @pytest.fixture
    def mocks(self, mocker):
        """Patch Climatology, PlotBias and PlotClimatology. Data has a plev dimension."""
        mock_gb_cls = mocker.patch(f"{CLI_MODULE}.Climatology")
        mock_bias_cls = mocker.patch(f"{CLI_MODULE}.PlotBias")
        mock_clim_cls = mocker.patch(f"{CLI_MODULE}.PlotClimatology")
        mock_gb_cls.return_value.data = {"t": MagicMock(dims=("plev",))}
        return mock_gb_cls, mock_bias_cls, mock_clim_cls

    @pytest.mark.parametrize(
        "references, extra_args",
        [([], []), (None, ["--climatology"])],
        ids=["no_references", "climatology_flag"],
    )
    def test_climatology_only(self, build_config, mock_cluster, mocks, references, extra_args):
        """Only the dataset is processed and PlotClimatology is used instead of PlotBias."""
        mock_gb_cls, mock_bias_cls, mock_clim_cls = mocks
        config_file = build_config(
            {
                "biases": {
                    **BASE_SET,
                    "variables": ["t"],
                    "params": {"default": {"plev": [85000, 20000], "seasons": True, "vertical": True}},
                },
            },
            references=references,
        )

        main(["--config", config_file, "--loglevel", "WARNING", *extra_args])

        mock_gb_cls.assert_called_once()
        assert mock_gb_cls.call_args.kwargs["model"] == "TestModel"
        assert mock_gb_cls.return_value.retrieve.call_count == 1
        assert mock_gb_cls.return_value.compute_climatology.call_count == 1

        mock_bias_cls.assert_not_called()
        plot_clim = mock_clim_cls.return_value
        assert plot_clim.plot_climatology.call_count == 2
        assert [c.kwargs["plev"] for c in plot_clim.plot_climatology.call_args_list] == [85000, 20000]
        assert plot_clim.plot_seasonal_climatology.call_count == 2
        plot_clim.plot_vertical_climatology.assert_called_once()
