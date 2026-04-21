"""Tests for the Ocean Stratification CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.ocean_stratification.cli_ocean_stratification import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.ocean_stratification.cli_ocean_stratification"

BASE_BLOCK = {
    "run": True,
    "regions": ["global_ocean"],
    "climatology": ["annual"],
    "diagnostic_name": "ocean_stratification",
    "var": ["thetao", "so"],
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
    """Test main() execution flow with mocked Stratification and plot classes."""

    @pytest.fixture
    def mock_os(self, mocker):
        mocks = {
            "Stratification": mocker.patch(f"{CLI_MODULE}.Stratification"),
            "PlotStratification": mocker.patch(f"{CLI_MODULE}.PlotStratification"),
            "PlotMLD": mocker.patch(f"{CLI_MODULE}.PlotMLD"),
        }
        # Allow data[["thetao", "so", "rho"]] and data[["mld"]] in CLI.
        mocks["Stratification"].return_value.data = mocker.MagicMock()
        return mocks

    def test_stratification_disabled_skips_processing(self, build_config, mock_cluster, mock_os):
        """When run=False, diagnostic and plot classes are not instantiated."""
        config_file = build_config({"ocean_stratification": {"stratification": {"run": False}}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_os["Stratification"].assert_not_called()
        mock_os["PlotStratification"].assert_not_called()
        mock_os["PlotMLD"].assert_not_called()

    def test_stratification_full_pipeline(self, build_config, mock_cluster, mock_os):
        """
        With run=True and one reference, per-region flow creates:
        - 4 Stratification runs (model+obs for stratification and MLD)
        - 1 PlotStratification and 1 PlotMLD call.
        """
        config_file = build_config({"ocean_stratification": {"stratification": BASE_BLOCK}})

        main(["--config", config_file, "--loglevel", "WARNING"])

        strat_cls = mock_os["Stratification"]
        assert strat_cls.call_count == 4
        assert strat_cls.return_value.run.call_count == 4
        mock_os["PlotStratification"].return_value.plot_stratification.assert_called_once()
        mock_os["PlotMLD"].return_value.plot_mld.assert_called_once()
