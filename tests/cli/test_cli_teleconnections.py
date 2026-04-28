"""Tests for the Teleconnections CLI (parse_arguments + main orchestration)."""

import pytest

from aqua.diagnostics.teleconnections.cli_teleconnections import main, parse_arguments

CLI_MODULE = "aqua.diagnostics.teleconnections.cli_teleconnections"

# Minimal block shared by NAO and ENSO (they have identical orchestration shape).
BASE_TC_BLOCK = {
    "run": True,
    "seasons": ["annual"],
    "months_window": 3,
}

BASE_TC = {
    "NAO": BASE_TC_BLOCK,
    "ENSO": BASE_TC_BLOCK,
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
    """Test main() execution flow with mocked NAO, ENSO and plot classes."""

    @pytest.fixture
    def mock_tc(self, mocker):
        """
        Patch NAO, ENSO and their plot counterparts.
        Returns a dict of mocks keyed by class name.

        plot_index() is configured to return a 2-tuple so that the
        ``fig_index, _ = plot_*.plot_index()`` unpacking in main() succeeds.
        """
        mocks = {
            "NAO": mocker.patch(f"{CLI_MODULE}.NAO"),
            "ENSO": mocker.patch(f"{CLI_MODULE}.ENSO"),
            "PlotNAO": mocker.patch(f"{CLI_MODULE}.PlotNAO"),
            "PlotENSO": mocker.patch(f"{CLI_MODULE}.PlotENSO"),
        }
        mocks["PlotNAO"].return_value.plot_index.return_value = (mocker.MagicMock(), mocker.MagicMock())
        mocks["PlotENSO"].return_value.plot_index.return_value = (mocker.MagicMock(), mocker.MagicMock())
        return mocks

    def test_all_diagnostics_disabled_skip_processing(self, build_config, mock_cluster, mock_tc):
        """When both NAO and ENSO have run=False, no diagnostic class is instantiated."""
        config_file = build_config(
            {
                "teleconnections": {
                    "NAO": {"run": False},
                    "ENSO": {"run": False},
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_tc["NAO"].assert_not_called()
        mock_tc["ENSO"].assert_not_called()
        mock_tc["PlotNAO"].assert_not_called()
        mock_tc["PlotENSO"].assert_not_called()

    def test_nao_full_pipeline(self, build_config, mock_cluster, mock_tc):
        """
        With NAO enabled, verify NAO is created for each dataset and reference,
        compute_index + compute_regression + compute_correlation are called,
        and PlotNAO produces the index and map plots.
        """
        config_file = build_config(
            {
                "teleconnections": {"NAO": BASE_TC["NAO"]},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        nao_instance = mock_tc["NAO"].return_value
        # 1 dataset + 1 reference = 2 NAO instantiations
        assert mock_tc["NAO"].call_count == 2
        assert nao_instance.retrieve.call_count == 2
        assert nao_instance.compute_index.call_count == 2

        # 1 season * 2 (dataset + reference) = 2 regression/correlation calls
        assert nao_instance.compute_regression.call_count == 2
        assert nao_instance.compute_correlation.call_count == 2

        mock_tc["PlotNAO"].assert_called_once()
        plot_nao_instance = mock_tc["PlotNAO"].return_value
        plot_nao_instance.plot_index.assert_called_once()
        # plot_maps called once per season for regression and once for correlation
        assert plot_nao_instance.plot_maps.call_count == 2
        mock_tc["ENSO"].assert_not_called()

    def test_enso_full_pipeline(self, build_config, mock_cluster, mock_tc):
        """
        With ENSO enabled, verify ENSO is created for each dataset and reference,
        compute_index + compute_regression + compute_correlation are called,
        and PlotENSO produces the index and map plots.
        """
        config_file = build_config(
            {
                "teleconnections": {"ENSO": BASE_TC["ENSO"]},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        enso_instance = mock_tc["ENSO"].return_value
        assert mock_tc["ENSO"].call_count == 2
        assert enso_instance.compute_regression.call_count == 2
        assert enso_instance.compute_correlation.call_count == 2

        mock_tc["PlotENSO"].assert_called_once()
        plot_enso_instance = mock_tc["PlotENSO"].return_value
        plot_enso_instance.plot_index.assert_called_once()
        assert plot_enso_instance.plot_maps.call_count == 2
        mock_tc["NAO"].assert_not_called()
