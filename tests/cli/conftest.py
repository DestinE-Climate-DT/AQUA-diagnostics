"""Shared fixtures for CLI tests.

These fixtures build test configs from real repository collection YAML files,
then override only the sections needed by each test. This keeps tests aligned
with the live CLI config schema and helps catch breaking config changes early.
"""

import os
from copy import deepcopy
from pathlib import Path

import pytest

from aqua.core.util import dump_yaml, load_yaml

CLI_BASE_MODULE = "aqua.diagnostics.base.cli_base"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_COLLECTIONS_DIR = REPO_ROOT / "aqua" / "diagnostics" / "config" / "collections"
DEFAULT_DATASET = {
    "catalog": "test-catalog",
    "model": "TestModel",
    "exp": "test-exp",
    "source": "test-source",
}
DEFAULT_REFERENCE = {
    "catalog": "ref-catalog",
    "model": "RefModel",
    "exp": "ref-exp",
    "source": "ref-source",
}
DEFAULT_OUTPUT = {
    "rebuild": False,
    "save_format": ["pdf"],
    "save_netcdf": True,
    "dpi": 50,
    "create_catalog_entry": False,
}


def _find_matching_template(diagnostics):
    """Find the repository config that best matches requested diagnostics.

    The template is only used to decide whether a `references` section should
    be injected by default (its other top-level sections are overwritten by
    `_build_config`). Stricter schema/contract checks live in
    `tests/config/test_diagnostic_key_coverage.py`.
    """
    collection_config_files = tuple(sorted(CONFIG_COLLECTIONS_DIR.rglob("config-*.yaml")))

    requested_keys = frozenset(diagnostics)
    best_template_cfg = None
    best_overlap_size = -1

    for template_file in collection_config_files:
        cfg = load_yaml(str(template_file))
        template_keys = set((cfg.get("diagnostics") or {}).keys())
        overlap_size = len(requested_keys & template_keys)
        if overlap_size > best_overlap_size:
            best_overlap_size = overlap_size
            best_template_cfg = cfg

    if best_template_cfg is None:
        raise ValueError("Could not select a repository collection configuration template.")

    return best_template_cfg


def _build_config(
    tmp_path,
    diagnostics,
    *,
    datasets=None,
    references=None,
    output_overrides=None,
    setup_overrides=None,
):
    """Build a YAML config from repository templates and return its path.

    Args:
        tmp_path: pytest tmp_path fixture.
        diagnostics: mapping of diagnostic keys to their config dicts,
            e.g. ``{"globalbiases": {...}}`` or
            ``{"seaice_timeseries": {...}, "seaice_2d_bias": {...}}``.
        datasets: list of dataset dicts (override template datasets).
        references: list of reference dicts (override template references).
        output_overrides: dict merged into the 'output' section.
        setup_overrides: dict merged into the 'setup' section.
    """
    outputdir = str(tmp_path / "output")
    os.makedirs(outputdir, exist_ok=True)

    # Use a real collection template so CLI tests match config structure.
    config = deepcopy(_find_matching_template(diagnostics))

    # Force deterministic values for test isolation
    config["setup"] = {"loglevel": "WARNING", **(setup_overrides or {})}
    config["datasets"] = datasets or [deepcopy(DEFAULT_DATASET)]
    if references is None:
        if "references" in config:
            config["references"] = [deepcopy(DEFAULT_REFERENCE)]
    else:
        config["references"] = references

    config["output"] = {"outputdir": outputdir, **deepcopy(DEFAULT_OUTPUT), **(output_overrides or {})}
    config["diagnostics"] = diagnostics

    first_key = next(iter(diagnostics), "test")
    config_file = tmp_path / f"config_{first_key}.yaml"
    dump_yaml(outfile=str(config_file), cfg=config)
    return str(config_file)


@pytest.fixture
def build_config(tmp_path):
    """Build a temporary config file for a diagnostic CLI test.

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

    This keeps CLI tests fast and deterministic by avoiding real Dask startup.
    Returns (mock_open, mock_close) so tests can assert on cluster orchestration.
    """
    mock_open = mocker.patch(
        f"{CLI_BASE_MODULE}.open_cluster",
        return_value=(None, None, False),
    )
    mock_close = mocker.patch(f"{CLI_BASE_MODULE}.close_cluster")
    return mock_open, mock_close
