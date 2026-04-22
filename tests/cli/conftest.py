"""Shared fixtures for CLI tests.

These fixtures provide minimal YAML configs and mock arguments that
satisfy DiagnosticCLI.prepare() without needing real data or catalogs.
They are designed to be reused across all diagnostic CLI test modules.
"""

import os

import pytest

from aqua.core.util import dump_yaml

CLI_BASE_MODULE = "aqua.diagnostics.base.cli_base"


def _build_config(
    tmp_path,
    diagnostics,
    *,
    datasets=None,
    references=None,
    output_overrides=None,
):
    """Build a YAML config file and return its path.

    Args:
        tmp_path: pytest tmp_path fixture.
        diagnostics: mapping of diagnostic keys to their config dicts,
            e.g. ``{"globalbiases": {...}}`` or
            ``{"seaice_timeseries": {...}, "seaice_2d_bias": {...}}``.
        datasets: list of dataset dicts (sensible default provided).
        references: list of reference dicts (sensible default provided).
        output_overrides: dict merged into the 'output' section.
    """
    outputdir = str(tmp_path / "output")
    os.makedirs(outputdir, exist_ok=True)

    default_datasets = [
        {
            "catalog": "test-catalog",
            "model": "TestModel",
            "exp": "test-exp",
            "source": "test-source",
        }
    ]
    default_references = [
        {
            "catalog": "ref-catalog",
            "model": "RefModel",
            "exp": "ref-exp",
            "source": "ref-source",
        }
    ]
    config = {
        "setup": {"loglevel": "WARNING"},
        "datasets": default_datasets if datasets is None else datasets,
        "references": default_references if references is None else references,
        "output": {
            "outputdir": outputdir,
            "rebuild": False,
            "save_format": ["pdf"],
            "save_netcdf": True,
            "dpi": 50,
            "create_catalog_entry": False,
            **(output_overrides or {}),
        },
        "diagnostics": diagnostics,
    }

    first_key = next(iter(diagnostics), "test")
    config_file = tmp_path / f"config_{first_key}.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def build_config(tmp_path):
    """Build a minimal YAML config file for a diagnostic CLI test.

    Usage::

        config_file = build_config({"globalbiases": {...}})
        config_file = build_config({"seaice_timeseries": {...}, "seaice_2d_bias": {...}})
    """

    def _build_config_file(diagnostics, **kwargs):
        return _build_config(tmp_path, diagnostics, **kwargs)

    return _build_config_file


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
