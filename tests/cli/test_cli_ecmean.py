"""Tests for the ECmean CLI (parse_arguments + main orchestration)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NoDataError, NotEnoughDataError
from aqua.core.util import load_yaml
from aqua.diagnostics.ecmean.cli_ecmean import (
    data_check,
    main,
    parse_arguments,
    reader_data,
    set_description,
    set_title,
    time_check,
)

CLI_MODULE = "aqua.diagnostics.ecmean.cli_ecmean"

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]


BASE_ECMEAN = {
    "nprocs": 2,
    "interface_file": "interface.yaml",
    "config_file": "ecmean-default.yaml",
    "global_mean": {
        "diagnostic_name": "ecmean_gm",
        "atm_vars": ["2t"],
        "oce_vars": ["tos"],
        "regions": ["Global"],
        "year1": 2000,
        "year2": 2001,
    },
    "performance_indices": {
        "diagnostic_name": "ecmean_pi",
        "atm_vars": ["2t"],
        "oce_vars": ["tos"],
        "regions": ["Global"],
        "year1": 2000,
        "year2": 2001,
    },
}


def _tools_config():
    return {"dirs": {}, "global_mean": {"regions": ["Global"]}, "performance_indices": {"regions": ["Global"]}}


def _prepare_config_load(mock_ecmean, build_config, *, save_format=None, catalog="test-catalog"):
    config_file = build_config(
        {"ecmean": BASE_ECMEAN},
        output_overrides={"save_format": save_format if save_format is not None else []},
    )
    config_dict = load_yaml(config_file)
    config_dict["datasets"][0]["catalog"] = catalog
    mock_ecmean["load"].side_effect = [config_dict, _tools_config()]


def test_parse_arguments_cli_options():
    """Verify parse_arguments parses ECmean-specific options."""
    args = parse_arguments(["--model", "IFS", "--nprocs", "4", "--source_oce", "oce-source"])
    assert args.model == "IFS"
    assert args.nprocs == 4
    assert args.source_oce == "oce-source"

    with pytest.raises(SystemExit):
        parse_arguments(["--help"])


class TestMainExecutionFlow:
    """Test main() execution flow with mocked ECmean components."""

    @pytest.fixture
    def mock_ecmean(self, mocker):
        mock_load = mocker.patch(f"{CLI_MODULE}.load_diagnostic_config")
        mock_merge = mocker.patch(f"{CLI_MODULE}.merge_config_args", side_effect=lambda cfg, _args: cfg)
        mocker.patch(f"{CLI_MODULE}.get_diagnostic_configpath", return_value="/tmp/ecmean-tools")
        mock_reader_data = mocker.patch(f"{CLI_MODULE}.reader_data", return_value=MagicMock(name="dataset"))
        mocker.patch(f"{CLI_MODULE}.data_check", return_value=MagicMock(name="checked_dataset"))
        mock_time_check = mocker.patch(f"{CLI_MODULE}.time_check", return_value=(2000, 2001))
        mock_configpath = mocker.patch(f"{CLI_MODULE}.ConfigPath")
        mock_outputsaver = mocker.patch(f"{CLI_MODULE}.OutputSaver")
        mock_perf = mocker.patch(f"{CLI_MODULE}.PerformanceIndices")
        mock_gm = mocker.patch(f"{CLI_MODULE}.GlobalMean")
        return {
            "load": mock_load,
            "merge": mock_merge,
            "reader_data": mock_reader_data,
            "time_check": mock_time_check,
            "configpath": mock_configpath,
            "outputsaver": mock_outputsaver,
            "performance": mock_perf,
            "global_mean": mock_gm,
        }

    def test_main_runs_both_ecmean_diagnostics(self, mock_ecmean, build_config):
        """Verify both global_mean and performance_indices are executed end-to-end."""
        _prepare_config_load(mock_ecmean, build_config, save_format=[])

        main(["--config", "dummy.yaml", "--loglevel", "WARNING"])

        mock_ecmean["merge"].assert_called_once()
        mock_ecmean["global_mean"].assert_called_once()
        mock_ecmean["performance"].assert_called_once()

        for instance in (mock_ecmean["global_mean"].return_value, mock_ecmean["performance"].return_value):
            for method_name in ("prepare", "run", "store", "plot"):
                getattr(instance, method_name).assert_called_once()

        # No save_figure calls when save_format is empty.
        assert mock_ecmean["outputsaver"].return_value.save_figure.call_count == 0

    def test_main_saves_figures_when_save_format_is_set(self, mock_ecmean, build_config):
        """When output.save_format is non-empty, figures are saved for both diagnostics."""
        _prepare_config_load(mock_ecmean, build_config, save_format=["png"])

        main(["--config", "dummy.yaml", "--loglevel", "WARNING"])

        assert mock_ecmean["outputsaver"].return_value.save_figure.call_count == 2

    def test_main_uses_catalog_fallback_when_missing(self, mock_ecmean, build_config):
        """If dataset catalog is missing, ConfigPath fallback should provide it."""
        _prepare_config_load(mock_ecmean, build_config, save_format=[], catalog=None)
        catalog_obj = MagicMock()
        catalog_obj.name = "fallback-catalog"
        mock_ecmean["configpath"].return_value.deliver_intake_catalog.return_value = (catalog_obj, None, None)

        main(["--config", "dummy.yaml", "--loglevel", "WARNING"])

        mock_ecmean["configpath"].assert_called_once()
        first_outputsaver_call = mock_ecmean["outputsaver"].call_args_list[0]
        assert first_outputsaver_call.kwargs["catalog"] == "fallback-catalog"

    def test_main_applies_realization_source_oce_and_dates_overrides(self, mock_ecmean, build_config):
        """CLI overrides should flow into reader_data and diagnostic constructor years."""
        _prepare_config_load(mock_ecmean, build_config, save_format=[])

        main(
            [
                "--config",
                "dummy.yaml",
                "--loglevel",
                "WARNING",
                "--source_oce",
                "oce-cli",
                "--realization",
                "r1i1p1f1",
                "--startdate",
                "2003-01-01",
                "--enddate",
                "2004-12-31",
            ]
        )

        # Atmospheric load uses dataset source, oceanic load uses source_oce override.
        first_atm_load, first_oce_load = mock_ecmean["reader_data"].call_args_list[:2]
        assert first_atm_load.kwargs["source"] == "test-source"
        assert first_oce_load.kwargs["source"] == "oce-cli"

        # realization is injected into reader_kwargs passed to reader_data.
        assert first_atm_load.kwargs["reader_kwargs"]["realization"] == "r1i1p1f1"

        # Dates override year1/year2 from config before time_check is called.
        first_time_check = mock_ecmean["time_check"].call_args_list[0]
        assert first_time_check.args[1] == 2003
        assert first_time_check.args[2] == 2004


def test_data_check_raises_with_no_data():
    """When both atmospheric and oceanic data are missing, NoDataError is raised."""
    with pytest.raises(NoDataError):
        data_check(None, None)


def test_set_title_rejects_unknown_diagnostic():
    """Unknown diagnostic names should fail fast during title generation."""
    with pytest.raises(ValueError):
        set_title("unknown_diag", "Model", "exp", 2000, 2001)


def test_set_description_global_mean():
    """Description text includes core context fields for known diagnostics."""
    config = {"global_mean": {"regions": ["Global", "NH"]}}
    description = set_description("global_mean", "IFS", "hist", 2000, 2001, config)

    assert "IFS hist from 2000-01-01 to 2001-12-31" in description
    assert "Global (90°S-90°N)" in description
    assert "NH (20°N-90°N)" in description


def test_time_check_infers_years_and_validates_minimum_months():
    """time_check should infer bounds and reject timeseries shorter than 12 unique months."""
    monthly = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS")})
    assert time_check(monthly, None, None) == (2000, 2000)

    too_short = xr.Dataset(coords={"time": pd.date_range("2001-01-01", periods=11, freq="MS")})
    with pytest.raises(NotEnoughDataError):
        time_check(too_short, None, None)


def test_reader_data_returns_none_for_model_false():
    """reader_data should short-circuit when model is explicitly False."""
    assert reader_data(model=False, exp="exp", source="source") is None
