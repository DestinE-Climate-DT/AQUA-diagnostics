"""Tests for Tropical Cyclones FDB configuration and CLI precedence."""

from pathlib import Path

import pytest
from tropical_cyclones import TCs
from tropical_cyclones.cli_tropical_cyclones import main, parse_arguments

from aqua.core.util import dump_yaml, load_yaml

CLI_MODULE = "tropical_cyclones.cli_tropical_cyclones"
TC_MODULE = "tropical_cyclones.tropical_cyclones"
CONFIG_FILE = Path(__file__).resolve().parents[2] / "frontier-diagnostics/tropical_cyclones/config/config_tcs.yaml"

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics, pytest.mark.frontier]


@pytest.fixture
def tc_config(tmp_path):
    """Use the shipped TC configuration with isolated output paths."""
    config = load_yaml(str(CONFIG_FILE))
    config["paths"] = {key: str(tmp_path / key) for key in config["paths"]}
    config["output"]["outputdir"] = str(tmp_path / "output")
    return config


@pytest.mark.parametrize("option", ["--config-fdb", "--config_fdb"])
def test_parse_config_fdb(option):
    """Accept both flag spellings and retain the default when omitted."""
    assert parse_arguments([option, "/fdb/config.yaml"]).config_fdb == "/fdb/config.yaml"
    assert parse_arguments([]).config_fdb is None


@pytest.mark.parametrize(
    "reader_kwargs, cli_args, expected_kwargs",
    [
        ({"chunks": {"time": 6}}, [], {"chunks": {"time": 6}}),
        (
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
            [],
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
        ),
        (None, ["--config-fdb", "/cli/config.yaml"], {"config_fdb": "/cli/config.yaml"}),
        (
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
            ["--config_fdb", "/cli/config.yaml"],
            {"chunks": {"time": 6}, "config_fdb": "/cli/config.yaml"},
        ),
        (
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
            ["--reader_kwargs", '{"config_fdb": "/json/config.yaml"}', "--config-fdb", "/cli/config.yaml"],
            {"chunks": {"time": 6}, "config_fdb": "/cli/config.yaml"},
        ),
    ],
    ids=["default", "yaml", "cli-with-null-kwargs", "cli-over-yaml", "dedicated-flag-over-json"],
)
def test_cli_config_fdb(tc_config, tmp_path, mock_cluster, mocker, reader_kwargs, cli_args, expected_kwargs):
    """Pass the merged FDB setting to TCs without dropping other Reader options."""
    tc_config["datasets"][0]["reader_kwargs"] = reader_kwargs
    config_file = tmp_path / "config_tcs.yaml"
    dump_yaml(outfile=str(config_file), cfg=tc_config)
    diagnostic = mocker.patch(f"{CLI_MODULE}.TCs")

    main(["--config", str(config_file), *cli_args])

    diagnostic.assert_called_once()
    merged_config = diagnostic.call_args.kwargs["tdict"]
    assert merged_config["datasets"][0]["reader_kwargs"] == expected_kwargs
    diagnostic.return_value.loop_streaming.assert_called_once_with(merged_config)


def test_config_fdb_forwarded_to_all_readers(tc_config, mocker):
    """The orography Reader must use the same FDB configuration as the atmospheric Readers."""
    tc_config["datasets"][0]["engine"] = "z3fdb"
    reader_kwargs = {"config_fdb": "/fdb/config.yaml", "chunks": {"time": 6}}
    tc_config["datasets"][0]["reader_kwargs"] = reader_kwargs
    tc_config["detect"]["write_fullres"] = True
    tc_config["orography"]["file_path"] = None
    reader = mocker.patch(f"{TC_MODULE}.Reader")

    TCs(
        tdict=tc_config,
        streaming=True,
        stream_step="1D",
        stream_startdate=tc_config["time"]["startdate"],
        orography=True,
    )

    assert reader.call_count == 4
    assert [call.kwargs["source"] for call in reader.call_args_list] == [
        tc_config["datasets"][0]["source2d"],
        tc_config["datasets"][0]["source3d"],
        tc_config["datasets"][0]["source2d"],
        tc_config["orography"]["source_oro"],
    ]
    for call in reader.call_args_list:
        assert call.kwargs["engine"] == "z3fdb"
        assert call.kwargs["config_fdb"] == reader_kwargs["config_fdb"]
        assert call.kwargs["chunks"] == reader_kwargs["chunks"]
