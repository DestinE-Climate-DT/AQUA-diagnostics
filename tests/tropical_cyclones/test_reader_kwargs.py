"""Regression tests for Reader options used by catalog-based TC orography."""

from pathlib import Path

import pytest
from tropical_cyclones import TCs
from tropical_cyclones.cli_tropical_cyclones import parse_arguments

from aqua.core.util import dump_yaml, load_yaml
from aqua.diagnostics.base import DiagnosticCLI

TC_MODULE = "tropical_cyclones.tropical_cyclones"
CONFIG_FILE = Path(__file__).resolve().parents[2] / "frontier-diagnostics/tropical_cyclones/config/config_tcs.yaml"

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics, pytest.mark.frontier]


@pytest.fixture
def tc_config(tmp_path):
    """Use the shipped TC configuration with isolated output paths."""
    config = load_yaml(str(CONFIG_FILE))
    config["paths"] = {key: str(tmp_path / key) for key in config["paths"]}
    config["output"]["outputdir"] = str(tmp_path / "output")
    config["datasets"][0]["engine"] = "z3fdb"
    config["detect"]["write_fullres"] = True
    config["orography"]["file_path"] = None
    return config


@pytest.mark.parametrize(
    "reader_kwargs, cli_args, expected_kwargs",
    [
        (
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
            [],
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
        ),
        (
            {"chunks": {"time": 6}, "config_fdb": "/yaml/config.yaml"},
            ["--reader_kwargs", '{"config_fdb": "/cli/config.yaml"}'],
            {"chunks": {"time": 6}, "config_fdb": "/cli/config.yaml"},
        ),
        (
            None,
            ["--reader_kwargs", '{"config_fdb": "/cli/config.yaml"}'],
            {"config_fdb": "/cli/config.yaml"},
        ),
        (None, [], {}),
        ({}, [], {}),
    ],
    ids=["yaml", "cli-over-yaml", "cli-with-null-kwargs", "default-null", "default-empty"],
)
def test_reader_kwargs_forwarded_to_orography(tc_config, tmp_path, mocker, reader_kwargs, cli_args, expected_kwargs):
    """Orography must receive the same merged Reader kwargs as atmospheric fields."""
    tc_config["datasets"][0]["reader_kwargs"] = reader_kwargs
    config_file = tmp_path / "config_tcs.yaml"
    dump_yaml(outfile=str(config_file), cfg=tc_config)
    args = parse_arguments(["--config", str(config_file), *cli_args])
    cli = DiagnosticCLI(args, diagnostic_name="tcs", default_config="config_tcs.yaml").prepare()
    reader = mocker.patch(f"{TC_MODULE}.Reader")

    TCs(
        tdict=cli.config_dict,
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
        for key in ("config_fdb", "chunks"):
            assert call.kwargs.get(key) == expected_kwargs.get(key)
