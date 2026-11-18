import argparse
import os
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aqua.core.exceptions import NotEnoughDataError
from aqua.core.logger import log_configure
from aqua.core.util import dump_yaml
from aqua.diagnostics import get_install_dirs
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
from aqua.diagnostics.teleconnections.base import BaseMixin
from tests.shared_constants import LOGLEVEL

loglevel = LOGLEVEL

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]
REPO_DIAGNOSTICS_DIR = Path(__file__).resolve().parents[2] / "aqua" / "diagnostics"
REAL_CONFIG_DIR = REPO_DIAGNOSTICS_DIR / "config"


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


def test_load_diagnostic_config():
    """Test loading a real diagnostic configuration from the repository config path."""

    default_config = os.path.join(
        REAL_CONFIG_DIR, "collections", "jinja", "climate_metrics", "config-climate_metrics-gregory.j2"
    )
    ts_dict = load_diagnostic_config(
        diagnostic="climate_metrics",
        config=default_config,
        loglevel=loglevel,
    )

    assert ts_dict["datasets"][0]["source"] == "lra-r100-monthly"
    assert "gregory" in ts_dict["diagnostics"]


@pytest.fixture
def aqua_configdir(tmp_path, monkeypatch):
    """Return a factory of AQUA config folders with or without the ``project`` key (pre-project layout)."""
    monkeypatch.delenv("AQUA_PROJECT", raising=False)

    def _make(project):
        configdir = tmp_path / "aqua"
        cfg = {"machine": "test-machine"}
        if project:
            cfg["project"] = project
        tools = configdir / "tools" / project if project else configdir / "tools"
        (tools / "teleconnections").mkdir(parents=True)
        (configdir / "templates" / "collections").mkdir(parents=True)
        dump_yaml(outfile=str(configdir / "config-aqua.yaml"), cfg=cfg)
        monkeypatch.setenv("AQUA_CONFIG", str(configdir))
        return configdir, tools

    return _make


@pytest.mark.parametrize("project", ["diagnostics", None], ids=["project", "pre-project"])
def test_get_diagnostic_configpath(aqua_configdir, project):
    """Tools are looked up in the active project folder, templates in the flat templates folder."""
    configdir, tools = aqua_configdir(project)

    assert get_diagnostic_configpath("teleconnections", folder="tools") == str(tools / "teleconnections")
    assert get_diagnostic_configpath("timeseries") == str(configdir / "templates" / "collections")

    with pytest.raises(ValueError, match="Invalid folder name"):
        get_diagnostic_configpath("timeseries", folder="collections")


def test_get_diagnostic_configpath_env_project(aqua_configdir, monkeypatch):
    """AQUA_PROJECT (exported by aqua analysis) selects the tools folder of another project."""
    configdir, _ = aqua_configdir("diagnostics")
    monkeypatch.setenv("AQUA_PROJECT", "other")

    with pytest.raises(FileNotFoundError, match="other"):
        get_diagnostic_configpath("teleconnections", folder="tools")

    (configdir / "tools" / "other").mkdir()
    expected = configdir / "tools" / "other" / "teleconnections"
    assert get_diagnostic_configpath("teleconnections", folder="tools") == str(expected)


@pytest.mark.parametrize("project", ["diagnostics", None], ids=["project", "pre-project"])
def test_teleconnections_definition_from_tools(aqua_configdir, project):
    """Teleconnections read their default definitions from the tools folder of the active project."""
    _, tools = aqua_configdir(project)
    (tools / "teleconnections" / "definitions").mkdir()
    dump_yaml(outfile=str(tools / "teleconnections" / "definitions" / "test.yaml"), cfg={"NAO": {"ok": True}})
    mixin = BaseMixin.__new__(BaseMixin)
    mixin.logger = log_configure(log_level=loglevel, log_name="test")

    assert mixin.load_definition(definition="test", telecname="NAO") == {"ok": True}


@pytest.mark.parametrize(
    "default_config, expected",
    [(None, "config-timeseries.yaml"), ("config-biases.yaml", "config-biases.yaml")],
)
def test_load_diagnostic_config_default(aqua_configdir, default_config, expected):
    """Without --config, the default file is read from templates/collections."""
    configdir, _ = aqua_configdir("diagnostics")
    dump_yaml(outfile=str(configdir / "templates" / "collections" / expected), cfg={"ok": expected})

    out = load_diagnostic_config("timeseries", config=None, default_config=default_config, loglevel=loglevel)

    assert out == {"ok": expected}


@pytest.mark.parametrize(
    "cli_module",
    [
        "biases.cli_biases",
        "boxplots.cli_boxplots",
        "ensemble.cli_multi_model_maps_ensemble",
        "ensemble.cli_multi_model_timeseries_ensemble",
        "ensemble.cli_zonal_ensemble",
        "histogram.cli_histogram",
        "lat_lon_profiles.cli_lat_lon_profiles",
        "ocean_drift.cli_ocean_drift",
        "ocean_stratification.cli_ocean_stratification",
        "ocean_trends.cli_ocean_trends",
        "seaice.cli_seaice",
        "teleconnections.cli_teleconnections",
        "timeseries.cli_timeseries",
        "trends.cli_trends",
    ],
)
def test_cli_default_config_has_template(cli_module):
    """The default_config of each CLI exists in templates/collections, where it is read without --config."""
    source = (REPO_DIAGNOSTICS_DIR / cli_module.replace(".", os.sep)).with_suffix(".py").read_text()
    templates = {path.name for path in (REPO_DIAGNOSTICS_DIR / "templates" / "collections").glob("*.yaml")}

    names = set(re.findall(r'"(config[-_][\w.-]+\.yaml)"', source))
    assert names, f"No default config found in {cli_module}"
    assert names <= templates, f"{cli_module}: {sorted(names - templates)} not in templates/collections"


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


@pytest.mark.parametrize("configured_kwargs", [None, {"chunks": {"time": 12}}])
def test_merge_config_args_normalizes_and_preserves_reader_kwargs(configured_kwargs):
    """CLI realization is merged into the first dataset without leaking to others."""
    parser = argparse.ArgumentParser()
    args = template_parse_arguments(parser).parse_args(["--realization", "r2"])
    config = {
        "datasets": [
            {
                "catalog": "test",
                "model": "Model1",
                "exp": "exp1",
                "source": "source1",
                "reader_kwargs": configured_kwargs,
            },
            {
                "catalog": "test",
                "model": "Model2",
                "exp": "exp2",
                "source": "source2",
                "reader_kwargs": {"realization": "r3"},
            },
        ],
        "output": {"outputdir": "./"},
    }

    merged_config = merge_config_args(config=config, args=args, loglevel=loglevel)

    expected = {"realization": "r2"}
    if configured_kwargs:
        expected["chunks"] = {"time": 12}
    assert merged_config["datasets"][0]["reader_kwargs"] == expected
    assert merged_config["datasets"][1]["reader_kwargs"] == {"realization": "r3"}


def test_close_private_cluster_when_flag_true():
    """close_cluster always closes client, and closes cluster only if private_cluster=True."""

    class _DummyCluster:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    client = _DummyCluster()
    cluster = _DummyCluster()

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

    assert start_end_dates(start_std="2020-01-01", end_std="2020-01-02") == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    )

    assert start_end_dates(startdate="2020-01-01", end_std="2020-01-02") == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    )


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


@patch("aqua.diagnostics.base.diagnostic.Reader")
def test_minimum_months_enough_1month(mock_reader_class):
    """No error when available months >= months_required."""
    mock_reader_class.return_value.retrieve.return_value = _make_monthly_dataset(1)
    mock_reader_class.return_value.catalog = "test"

    diag = Diagnostic(model="M", exp="E", source="S")
    result, _, _ = diag._retrieve(model="M", exp="E", source="S", months_required=1)
    assert len(result.time) == 1


def test_minimum_months_required_class_attribute():
    """Concrete diagnostics expose MINIMUM_MONTHS_REQUIRED as a positive int class attribute."""
    # LatLonProfiles is selected since is one of the easiest diagnostics on this perspective
    assert hasattr(LatLonProfiles, "MINIMUM_MONTHS_REQUIRED")
    assert isinstance(LatLonProfiles.MINIMUM_MONTHS_REQUIRED, int)
    assert LatLonProfiles.MINIMUM_MONTHS_REQUIRED > 0


def test_get_install_dirs():
    """get_install_dirs follows the plugin contract version 2 of aqua-core."""
    assert get_install_dirs() == {
        "contract": 2,
        "config": ["analysis", "collections", "tools"],
        "flat": ["definitions"],
        "templates": ["collections"],
        "files": [],
    }
    for key in ("config", "flat"):
        for folder in get_install_dirs()[key]:
            assert (REAL_CONFIG_DIR / folder).is_dir()
