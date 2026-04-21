import argparse
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NotEnoughDataError
from aqua.core.util import dump_yaml
from aqua.diagnostics.base import (
    Diagnostic,
    close_cluster,
    get_diagnostic_configpath,
    load_diagnostic_config,
    merge_config_args,
    open_cluster,
    round_enddate,
    round_startdate,
    start_end_dates,
    template_parse_arguments,
)
from aqua.diagnostics.lat_lon_profiles import LatLonProfiles
from tests.shared_constants import LOGLEVEL

loglevel = LOGLEVEL

pytestmark = pytest.mark.aqua


def test_template_parse_arguments():
    """Test the template_parse_arguments function"""
    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    # fmt: off
    args = parser.parse_args([
            "--loglevel", "DEBUG",
            "--catalog", "test_catalog",
            "--model", "test_model",
            "--exp", "test_exp",
            "--source", "test_source",
            "--config", "test_config.yaml",
            "--regrid", "r100",
            "--outputdir", "test_outputdir",
            "--cluster", "test_cluster",
            "--nworkers", "2"]
    )
    # fmt: on
    assert args.loglevel == "DEBUG"
    assert args.catalog == "test_catalog"
    assert args.model == "test_model"
    assert args.exp == "test_exp"
    assert args.source == "test_source"
    assert args.config == "test_config.yaml"
    assert args.regrid == "r100"
    assert args.outputdir == "test_outputdir"
    assert args.cluster == "test_cluster"
    assert args.nworkers == 2

    with pytest.raises(FileNotFoundError):
        load_diagnostic_config(diagnostic="pippo", config=None, loglevel=loglevel)


def test_load_diagnostic_config_from_args(tmp_path):
    """Test loading configuration from a file specified in arguments."""

    # Create a minimal config file
    config_data = {"datasets": [{"model": "TestModel", "exp": "test-exp"}], "output": {"outputdir": str(tmp_path / "output")}}

    config_file = os.path.join(str(tmp_path), "test_config_args.yaml")
    dump_yaml(outfile=config_file, cfg=config_data)

    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    args = parser.parse_args(["--config", config_file, "--loglevel", "DEBUG"])

    result = load_diagnostic_config(diagnostic="ignored_name", config=args.config, loglevel=loglevel)

    assert result["datasets"][0]["model"] == "TestModel"
    assert result["output"]["outputdir"] == str(tmp_path / "output")


@patch("aqua.diagnostics.base.util.Client")
@patch("aqua.diagnostics.base.util.LocalCluster")
def test_cluster(mock_cluster, mock_client):
    """Test the cluster functions with mocking"""

    # Test case 1: No workers specified
    client, cluster, private_cluster = open_cluster(None, None, loglevel)
    assert client is None
    assert cluster is None
    assert private_cluster is False

    # # Test case 2: New cluster creation
    client, cluster, private_cluster = open_cluster(2, None, loglevel)
    assert client is not None
    assert cluster is not None
    assert private_cluster is True

    # Test case 3: Using existing cluster
    previous_cluster = mock_cluster
    client, cluster, private_cluster = open_cluster(2, previous_cluster, loglevel)
    assert client is not None
    assert cluster is previous_cluster
    assert private_cluster is False

    close_cluster(client, cluster, private_cluster)


def test_load_diagnostic_config(monkeypatch):
    """Test the load_diagnostic_config function"""

    # Isolate this test from local AQUA catalog setup.
    class _DummyConfigPath:
        def __init__(self, loglevel=None):
            self.configdir = "config"

    monkeypatch.setattr("aqua.diagnostics.base.util.ConfigPath", _DummyConfigPath)
    monkeypatch.setattr(
        "aqua.diagnostics.base.util.load_yaml",
        lambda _: {
            "datasets": [
                {
                    "catalog": None,
                    "exp": None,
                    "model": None,
                    "source": "lra-r100-monthly",
                    "regrid": None,
                    "reader_kwargs": None,
                }
            ]
        },
    )

    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    args = parser.parse_args(["--loglevel", "DEBUG"])
    ts_dict = load_diagnostic_config(
        diagnostic="timeseries",
        default_config="config-timeseries.yaml",
        folder="templates",
        config=args.config,
        loglevel=loglevel,
    )

    assert ts_dict["datasets"] == [
        {"catalog": None, "exp": None, "model": None, "source": "lra-r100-monthly", "regrid": None, "reader_kwargs": None}
    ]


def test_get_diagnostic_configpath(monkeypatch):
    """Path resolver handles collections/tools/templates and rejects invalid folder names."""

    class _DummyConfigPath:
        def __init__(self, loglevel=None):
            self.configdir = "/tmp/aqua-config"

    monkeypatch.setattr("aqua.diagnostics.base.util.ConfigPath", _DummyConfigPath)

    assert get_diagnostic_configpath("timeseries", folder="collections") == "/tmp/aqua-config/collections/timeseries"
    assert get_diagnostic_configpath("timeseries", folder="tools") == "/tmp/aqua-config/tools/timeseries"
    assert get_diagnostic_configpath("timeseries", folder="templates") == "/tmp/aqua-config/templates/collections"

    with pytest.raises(ValueError, match="Invalid folder name"):
        get_diagnostic_configpath("timeseries", folder="invalid")


def test_load_diagnostic_config_default_filename(monkeypatch):
    """When no explicit config/default_config is provided, config-{diagnostic}.yaml is used."""
    calls = {}

    def _fake_get_path(diagnostic, folder="collections", loglevel="WARNING"):
        calls["path_args"] = (diagnostic, folder, loglevel)
        return "/tmp/aqua-config/collections/timeseries"

    def _fake_load_yaml(path):
        calls["loaded_path"] = path
        return {"ok": True}

    monkeypatch.setattr("aqua.diagnostics.base.util.get_diagnostic_configpath", _fake_get_path)
    monkeypatch.setattr("aqua.diagnostics.base.util.load_yaml", _fake_load_yaml)

    out = load_diagnostic_config("timeseries", config=None, default_config=None, folder="collections", loglevel="DEBUG")

    assert out == {"ok": True}
    assert calls["path_args"] == ("timeseries", "collections", "DEBUG")
    assert calls["loaded_path"] == "/tmp/aqua-config/collections/timeseries/config-timeseries.yaml"


def test_merge_config_args():
    """Test the merge_config_args function"""
    parser = argparse.ArgumentParser()
    parser = template_parse_arguments(parser)
    # fmt: off
    args = parser.parse_args(
        [
            "--loglevel", "DEBUG",
            "--catalog", "test_catalog",
            "--model", "test_model",
            "--exp", "test_exp",
            "--source", "test_source",
            "--outputdir", "test_outputdir",
        ]
    )
    # fmt: on
    ts_dict = {
        "datasets": [{"catalog": None, "model": None, "exp": None, "source": "lra-r100-monthly"}],
        "references": [{"catalog": "obs", "model": "ERA5", "exp": "era5", "source": "monthly"}],
        "output": {"outputdir": "./"},
    }

    merged_config = merge_config_args(config=ts_dict, args=args, loglevel=loglevel)

    assert merged_config["datasets"] == [
        {"catalog": "test_catalog", "exp": "test_exp", "model": "test_model", "source": "test_source"}
    ]
    assert merged_config["output"]["outputdir"] == "test_outputdir"


def test_close_cluster_closes_private_cluster_only_when_flag_true():
    """close_cluster always closes client, and closes cluster only if private_cluster=True."""

    class _Dummy:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    client = _Dummy()
    cluster = _Dummy()

    close_cluster(client, cluster, private_cluster=False, loglevel=loglevel)
    assert client.closed == 1
    assert cluster.closed == 0

    close_cluster(client, cluster, private_cluster=True, loglevel=loglevel)
    assert client.closed == 2
    assert cluster.closed == 1


def test_start_end_dates():
    # All None inputs
    assert start_end_dates() == (None, None)

    # Only startdate provided
    assert start_end_dates(startdate="2020-01-01") == (pd.Timestamp("2020-01-01"), None)

    # Two dates provided
    assert start_end_dates(startdate="2020-01-01", enddate="2020-01-02") == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    )
    assert start_end_dates(startdate="20200101", enddate="20200102") == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    )
    assert start_end_dates(startdate="20200101", start_std="20200102") == (pd.Timestamp("2020-01-01"), None)
    assert start_end_dates(startdate="2020-01-01", enddate="20200102") == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    )

    assert start_end_dates(start_std="2020-01-01", end_std="2020-01-02") == (None, None)

    assert start_end_dates(startdate="2020-01-01", end_std="2020-01-02") == (pd.Timestamp("2020-01-01"), None)


@pytest.mark.parametrize(
    "date,freq,expected",
    [
        ("2020-03-15 14:30:00", "monthly", "2020-03-01 00:00:00"),
        ("2020-06-15 14:30:00", "annual", "2020-01-01 00:00:00"),
    ],
)
def test_round_startdate(date, freq, expected):
    """Test rounding to start of month/year"""
    rounded = round_startdate(pd.Timestamp(date), freq=freq)
    assert rounded == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "date,freq,expected",
    [
        ("2020-02-15 14:30:00", "monthly", "2020-02-29 23:59:59"),
        ("2020-06-15 14:30:00", "annual", "2020-12-31 23:59:59"),
    ],
)
def test_round_enddate(date, freq, expected):
    """Test rounding to end of month/year"""
    rounded = round_enddate(pd.Timestamp(date), freq=freq)
    assert rounded == pd.Timestamp(expected)


def test_round_invalid_freq():
    """Test error handling for invalid frequency"""
    with pytest.raises(ValueError):
        round_startdate(pd.Timestamp("2020-03-15"), freq="weekly")
    with pytest.raises(ValueError):
        round_enddate(pd.Timestamp("2020-03-15"), freq="weekly")


def _make_monthly_dataset(n_months: int, start: str = "2000-01-01") -> xr.Dataset:
    """Helper: synthetic monthly dataset with n_months timesteps."""
    times = pd.date_range(start, periods=n_months, freq="MS")
    return xr.Dataset({"2t": xr.DataArray(np.ones(n_months), dims=["time"], coords={"time": times})})


@patch("aqua.diagnostics.base.diagnostic.Reader")
def test_minimum_months_not_enough(mock_reader_class):
    """NotEnoughDataError is raised when available months < months_required."""
    mock_reader_class.return_value.retrieve.return_value = _make_monthly_dataset(6)
    mock_reader_class.return_value.catalog = "test"

    diag = Diagnostic(model="M", exp="E", source="S")
    with pytest.raises(NotEnoughDataError):
        diag._retrieve(model="M", exp="E", source="S", months_required=12)


@patch("aqua.diagnostics.base.diagnostic.Reader")
def test_minimum_months_enough(mock_reader_class):
    """No error when available months >= months_required."""
    mock_reader_class.return_value.retrieve.return_value = _make_monthly_dataset(12)
    mock_reader_class.return_value.catalog = "test"

    diag = Diagnostic(model="M", exp="E", source="S")
    result, _, _ = diag._retrieve(model="M", exp="E", source="S", months_required=12)
    assert len(result.time) == 12


def test_minimum_months_required_class_attribute():
    """Concrete diagnostics expose MINIMUM_MONTHS_REQUIRED as a positive int class attribute."""
    # LatLonProfiles is selected since is one of the easiest diagnostics on this perspective
    assert hasattr(LatLonProfiles, "MINIMUM_MONTHS_REQUIRED")
    assert isinstance(LatLonProfiles.MINIMUM_MONTHS_REQUIRED, int)
    assert LatLonProfiles.MINIMUM_MONTHS_REQUIRED > 0
