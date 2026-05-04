"""Tests for the Timeseries CLI (parse_arguments + main orchestration)."""

import pandas as pd
import pytest

from aqua.diagnostics.timeseries.cli_timeseries import main, parse_arguments
from tests.cli.conftest import DEFAULT_REFERENCE

CLI_MODULE = "aqua.diagnostics.timeseries.cli_timeseries"

# Minimal timeseries block. `params.default` must include hourly/daily/monthly/annual
# because load_var_config pops those keys when building the freq list.
BASE_TIMESERIES = {
    "run": True,
    "variables": ["2t"],
    "formulae": [],
    "params": {
        "default": {"hourly": False, "daily": False, "monthly": False, "annual": True},
    },
}

BASE_SEASONALCYCLES = {
    "run": True,
    "variables": ["2t"],
    "params": {"default": {}},
}

BASE_GREGORY = {
    "run": True,
    "annual": True,
    "monthly": False,
    "std": True,
    "t2m_name": "2t",
    "net_toa_name": "tnlwrf+tnswrf",
    "std_startdate": "2000-01-01",
    "std_enddate": "2010-12-31",
    "t2m_ref": DEFAULT_REFERENCE,
    "net_toa_ref": DEFAULT_REFERENCE,
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
    args = parse_arguments(["--model", "IFS", "--nworkers", "2"])
    assert args.model == "IFS"
    assert args.nworkers == 2
    assert args.catalog is None

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


# ======================================================================
# CLI execution flow (main)
# ======================================================================
class TestMainExecutionFlow:
    """Test main() execution flow with mocked diagnostic and plot classes."""

    @pytest.fixture
    def mock_ts(self, mocker):
        """
        Patch Timeseries, SeasonalCycles, Gregory and their plot counterparts.
        Returns a dict of mocks keyed by class name.

        Timeseries/SeasonalCycles instances get pd.Timestamp plt_startdate and
        plt_enddate so the min()/max() aggregations in main() succeed.
        """
        mocks = {
            "Timeseries": mocker.patch(f"{CLI_MODULE}.Timeseries"),
            "SeasonalCycles": mocker.patch(f"{CLI_MODULE}.SeasonalCycles"),
            "Gregory": mocker.patch(f"{CLI_MODULE}.Gregory"),
            "PlotTimeseries": mocker.patch(f"{CLI_MODULE}.PlotTimeseries"),
            "PlotSeasonalCycles": mocker.patch(f"{CLI_MODULE}.PlotSeasonalCycles"),
            "PlotGregory": mocker.patch(f"{CLI_MODULE}.PlotGregory"),
        }
        start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
        mocks["Timeseries"].return_value.plt_startdate = start
        mocks["Timeseries"].return_value.plt_enddate = end
        mocks["SeasonalCycles"].return_value.plt_startdate = start
        mocks["SeasonalCycles"].return_value.plt_enddate = end
        return mocks

    def test_all_diagnostics_disabled_skip_processing(self, build_config, mock_cluster, mock_ts):
        """When every block has run=False, no diagnostic class is instantiated."""
        config_file = build_config(
            {
                "timeseries": {"run": False},
                "seasonalcycles": {"run": False},
                "gregory": {"run": False},
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        mock_ts["Timeseries"].assert_not_called()
        mock_ts["SeasonalCycles"].assert_not_called()
        mock_ts["Gregory"].assert_not_called()
        mock_ts["PlotTimeseries"].assert_not_called()
        mock_ts["PlotSeasonalCycles"].assert_not_called()
        mock_ts["PlotGregory"].assert_not_called()

    def test_timeseries_full_pipeline(self, build_config, mock_cluster, mock_ts):
        """
        With the timeseries block enabled (variables + formulae), verify that
        Timeseries is created for each dataset and reference, run() is called,
        and PlotTimeseries.plot_timeseries is called per variable.
        """
        config_file = build_config(
            {
                "timeseries": {
                    **BASE_TIMESERIES,
                    "variables": ["2t"],
                    "formulae": ["net_toa"],
                },
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        ts_instance = mock_ts["Timeseries"].return_value
        # 2 variables * (1 dataset + 1 reference) = 4 Timeseries instantiations
        assert mock_ts["Timeseries"].call_count == 4
        assert ts_instance.run.call_count == 4

        # 2 variables * 1 plot each = 2
        assert mock_ts["PlotTimeseries"].call_count == 2
        assert mock_ts["PlotTimeseries"].return_value.plot_timeseries.call_count == 2

        # Second run corresponds to the reference for variable '2t': formula=False
        first_run = ts_instance.run.call_args_list[0]
        assert first_run.kwargs["var"] == "2t"
        assert first_run.kwargs["formula"] is False

        # Third run corresponds to the dataset for formula 'net_toa': formula=True
        third_run = ts_instance.run.call_args_list[2]
        assert third_run.kwargs["var"] == "net_toa"
        assert third_run.kwargs["formula"] is True

    def test_seasonalcycles_full_pipeline(self, build_config, mock_cluster, mock_ts):
        """
        With seasonalcycles enabled, verify SeasonalCycles + PlotSeasonalCycles
        are wired correctly. The timeseries block is also enabled because the
        seasonalcycles orchestration relies on its Timeseries instances.
        """
        config_file = build_config(
            {
                "timeseries": BASE_TIMESERIES,
                "seasonalcycles": BASE_SEASONALCYCLES,
            }
        )

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 1 variable * (1 dataset + 1 reference) = 2 SeasonalCycles instantiations
        assert mock_ts["SeasonalCycles"].call_count == 2
        assert mock_ts["SeasonalCycles"].return_value.run.call_count == 2

        mock_ts["PlotSeasonalCycles"].assert_called_once()
        mock_ts["PlotSeasonalCycles"].return_value.plot_seasonalcycles.assert_called_once()

    def test_gregory_full_pipeline(self, build_config, mock_cluster, mock_ts):
        """
        With the gregory block enabled (std=True), verify Gregory is created
        for the dataset plus the t2m and net_toa references, and that
        PlotGregory.plot is called.
        """
        config_file = build_config({"gregory": BASE_GREGORY})

        main(["--config", config_file, "--loglevel", "WARNING"])

        # 1 dataset + 2 references (t2m_ref + net_toa_ref) = 3 Gregory instantiations
        assert mock_ts["Gregory"].call_count == 3
        assert mock_ts["Gregory"].return_value.run.call_count == 3

        mock_ts["PlotGregory"].assert_called_once()
        mock_ts["PlotGregory"].return_value.plot.assert_called_once()
