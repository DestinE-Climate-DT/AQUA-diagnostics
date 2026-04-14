"""Shared fixtures for CLI tests.

These fixtures provide minimal YAML configs and mock arguments that
satisfy DiagnosticCLI.prepare() without needing real data or catalogs.
They are designed to be reused across all diagnostic CLI test modules.
"""

import os

import pytest

from aqua.core.util import dump_yaml

CLI_BASE_MODULE = "aqua.diagnostics.base.cli_base"


@pytest.fixture
def cli_outputdir(tmp_path):
    """Provide a temporary output directory for CLI tests."""
    outdir = tmp_path / "output"
    outdir.mkdir()
    return str(outdir)


def _build_config(
    tmp_path,
    diagnostic_key,
    diagnostic_block,
    *,
    datasets=None,
    references=None,
    output_overrides=None,
):
    """Build a YAML config file and return its path.

    Args:
        tmp_path: pytest tmp_path fixture.
        diagnostic_key: key under 'diagnostics' (e.g. 'globalbiases').
        diagnostic_block: dict for the diagnostic section.
        datasets: list of dataset dicts (sensible default provided).
        references: list of reference dicts (sensible default provided).
        output_overrides: dict merged into the 'output' section.
    """
    outputdir = str(tmp_path / "output")
    os.makedirs(outputdir, exist_ok=True)

    config = {
        "setup": {"loglevel": "WARNING"},
        "datasets": datasets or [
            {
                "catalog": "test-catalog",
                "model": "TestModel",
                "exp": "test-exp",
                "source": "test-source",
            }
        ],
        "references": references or [
            {
                "catalog": "ref-catalog",
                "model": "RefModel",
                "exp": "ref-exp",
                "source": "ref-source",
            }
        ],
        "output": {
            "outputdir": outputdir,
            "rebuild": False,
            "save_format": ["pdf"],
            "save_netcdf": True,
            "dpi": 50,
            "create_catalog_entry": False,
            **(output_overrides or {}),
        },
        "diagnostics": {diagnostic_key: diagnostic_block},
    }

    config_file = tmp_path / f"config_{diagnostic_key}.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def build_config(tmp_path):
    """Factory fixture: call with (diagnostic_key, diagnostic_block, **kwargs)."""
    def _factory(diagnostic_key, diagnostic_block, **kwargs):
        return _build_config(tmp_path, diagnostic_key, diagnostic_block, **kwargs)
    return _factory


@pytest.fixture
def mock_cluster(mocker):
    """Mock open_cluster/close_cluster where DiagnosticCLI uses them.

    Returns (mock_open, mock_close) so tests can assert on them.
    """
    mock_open = mocker.patch(
        f"{CLI_BASE_MODULE}.open_cluster",
        return_value=(None, None, False),
    )
    mock_close = mocker.patch(f"{CLI_BASE_MODULE}.close_cluster")
    return mock_open, mock_close
