"""Tests for the OceanHeatBudget CLI (parse_arguments + main orchestration), with everything mocked."""

import pytest

from aqua.core.exceptions import NoDataError
from aqua.diagnostics.ocean_heat_budget.cli_ocean_heat_budget import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_heat_budget.cli_ocean_heat_budget"

BASE_OCEAN_HEAT_BUDGET = {"run": True, "diagnostic_name": "ocean_heat_budget", "variables": ["hc700m"]}

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


def test_parse_arguments_cli_options():
    """Detailed flag coverage lives in test_base_util.py."""
    args = parse_arguments(["--model", "IFS", "--nworkers", "4"])
    assert args.model == "IFS"
    assert args.nworkers == 4

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestMain:
    """main() with OceanHeatBudget and PlotOceanHeatBudget patched at the CLI module path."""

    @pytest.fixture
    def mocks(self, mocker):
        return mocker.patch(f"{CLI_MODULE}.OceanHeatBudget"), mocker.patch(f"{CLI_MODULE}.PlotOceanHeatBudget")

    def test_disabled_skips_processing(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"ocean_heat_budget": {"run": False}})])
        mock_diag_cls.assert_not_called()
        mock_plot_cls.assert_not_called()

    def test_full_pipeline(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        main(["--config", build_config({"ocean_heat_budget": BASE_OCEAN_HEAT_BUDGET})])
        mock_diag_cls.return_value.run.assert_called_once()
        mock_plot_cls.assert_called_once()
        mock_plot_cls.return_value.plot.assert_called_once()

    def test_plot_only_does_not_run(self, build_config, mock_cluster, mocks):
        mock_diag_cls, mock_plot_cls = mocks
        config_file = build_config({"ocean_heat_budget": BASE_OCEAN_HEAT_BUDGET}, output_overrides={"plot_only": True})
        main(["--config", config_file])
        mock_diag_cls.return_value.run.assert_not_called()
        mock_diag_cls.return_value.load.assert_called_once()

    def test_failure_skips_the_variable(self, build_config, mock_cluster, mocks):
        """One missing variable must not stop the run, nor produce a plot."""
        mock_diag_cls, mock_plot_cls = mocks
        mock_diag_cls.return_value.run.side_effect = NoDataError("missing")
        main(["--config", build_config({"ocean_heat_budget": BASE_OCEAN_HEAT_BUDGET})])
        mock_plot_cls.assert_not_called()
