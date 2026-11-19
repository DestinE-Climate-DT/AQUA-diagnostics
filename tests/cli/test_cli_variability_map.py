"""Tests for the VariabilityMap CLI (parse_arguments + main orchestration), with everything mocked."""

import pytest

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.variability_map.cli_variability_map import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.variability_map.cli_variability_map"

BASE_VM = {"run": True, "diagnostic_name": "ocean2d", "variables": ["zos"]}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    args = parse_arguments(["--model", "IFS", "--nworkers", "4"])
    assert args.model == "IFS"
    assert args.nworkers == 4


class TestMain:
    """main() with VariabilityMap and PlotVariabilityMap patched at the CLI module path."""

    @pytest.fixture
    def mocks(self, mocker):
        mock_diag_cls = mocker.patch(f"{CLI_MODULE}.VariabilityMap")
        mock_diag_cls.return_value.startdate = "1990-01-01"
        mock_diag_cls.return_value.enddate = "1990-12-31"
        return mock_diag_cls, mocker.patch(f"{CLI_MODULE}.PlotVariabilityMap")

    def test_disabled_skips_processing(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"variabilitymap": {"run": False}})])
        mock_diag_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mocks):
        """Dataset and reference are computed, each one plotted, then their difference."""
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"variabilitymap": BASE_VM}), "--model", "CLIModel"])
        models = [call.kwargs["model"] for call in mock_diag_cls.call_args_list]
        # The --model override applies to the dataset only, never to the reference
        assert models == ["CLIModel", "RefModel"]
        assert mock_diag_cls.call_args.kwargs["diagnostic_name"] == "ocean2d"
        assert mock_plot_cls.return_value.plot.call_count == 2
        mock_plot_cls.return_value.plot_diff.assert_called_once()

    def test_failure_skips_the_variable(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        mock_diag_cls.return_value.run.side_effect = NoDataError("missing")
        main(["--config", build_config({"variabilitymap": BASE_VM})])
        mock_plot_cls.return_value.plot.assert_not_called()
