"""Tests for the Budget CLI (parse_arguments + main orchestration), with everything mocked."""

import pytest

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.budget.cli_budget import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.budget.cli_budget"

BASE_BUDGET = {
    "run": True,
    "diagnostic_name": "budget",
    "budgets": ["ocean", "atmosphere"],
    "params": {
        "default": {"expected": 0.0},
        "ocean": {"first": "a+b", "second": "hc", "tendency": True, "mask": "tos"},
        "atmosphere": {"first": "c", "second": "a+b", "expected": 0.7},
    },
}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    """Detailed flag coverage lives in test_base_util.py."""
    args = parse_arguments(["--model", "IFS", "--nworkers", "4"])
    assert args.model == "IFS"
    assert args.nworkers == 4

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestMain:
    """main() with Budget and PlotBudget patched at the CLI module path."""

    @pytest.fixture
    def mocks(self, mocker):
        return mocker.patch(f"{CLI_MODULE}.Budget"), mocker.patch(f"{CLI_MODULE}.PlotBudget")

    def test_disabled_skips_processing(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"budget": {"run": False}})])
        mock_diag_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mocks):
        """Each budget runs with its params, and the plot gets its expected value."""
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"budget": BASE_BUDGET})])
        runs = mock_diag_cls.return_value.run.call_args_list
        assert [c.kwargs["name"] for c in runs] == ["ocean", "atmosphere"]
        assert runs[0].kwargs["tendency"] is True and runs[0].kwargs["mask"] == "tos"
        assert runs[1].kwargs["tendency"] is False and runs[1].kwargs["mask"] is None
        assert [c.kwargs["expected"] for c in mock_plot_cls.call_args_list] == [0.0, 0.7]

    def test_plot_only_does_not_run(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        config_file = build_config({"budget": BASE_BUDGET}, output_overrides={"plot_only": True})
        main(["--config", config_file])
        mock_diag_cls.return_value.run.assert_not_called()
        assert mock_diag_cls.return_value.load.call_count == 2

    def test_failure_skips_the_budget(self, build_config, mock_cluster, mocks):
        """One failing budget must not stop the run, nor produce a plot."""
        mock_diag_cls, mock_plot_cls = mocks
        mock_diag_cls.return_value.run.side_effect = [NoDataError("missing"), None]
        main(["--config", build_config({"budget": BASE_BUDGET})])
        assert mock_plot_cls.call_count == 1
