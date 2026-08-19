"""Contract tests for diagnostic keys in repository collection configs."""

from pathlib import Path

import pytest

from aqua.core.util import load_yaml

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

COLLECTIONS_DIR = Path(__file__).resolve().parents[2] / "aqua" / "diagnostics" / "config" / "collections"

# Keep this explicit: when a new diagnostic key appears in collection configs,
# we want this test to fail so corresponding CLI coverage is added on purpose.
KNOWN_DIAGNOSTIC_KEYS = {
    "boxplots",
    "ecmean",
    "ensemble",
    "globalbiases",
    "gregory",
    "histogram",
    "lat_lon_profiles",
    "ocean_drift",
    "ocean_stratification",
    "ocean_trends",
    "seaice_2d_bias",
    "seaice_seasonal_cycle",
    "seaice_timeseries",
    "seasonalcycles",
    "teleconnections",
    "timeseries",
}


def _collection_files():
    return tuple(sorted(COLLECTIONS_DIR.rglob("config-*.yaml")))


def _contains_run_flag(diagnostic_cfg):
    """Return True when a diagnostic block defines a run switch at any depth."""
    if not isinstance(diagnostic_cfg, dict):
        return False
    if "run" in diagnostic_cfg:
        return True
    return any(_contains_run_flag(value) for value in diagnostic_cfg.values())


@pytest.mark.parametrize("config_path", _collection_files(), ids=lambda p: p.name)
def test_collection_have_valid_diag_keys(config_path):
    """Collection YAMLs with diagnostic keys expose valid diagnostic blocks."""
    cfg = load_yaml(str(config_path))
    diagnostics = cfg.get("diagnostics")
    if diagnostics is None:
        pytest.skip(f"{config_path} does not expose a 'diagnostics' section.")

    assert isinstance(diagnostics, dict), f"{config_path}: 'diagnostics' must be a mapping."
    assert diagnostics, f"{config_path}: 'diagnostics' cannot be empty."

    for diagnostic_key, diagnostic_cfg in diagnostics.items():
        assert isinstance(diagnostic_cfg, dict), (
            f"{config_path}: diagnostic '{diagnostic_key}' must map to a dictionary configuration."
        )
        assert _contains_run_flag(diagnostic_cfg), (
            f"{config_path}: diagnostic '{diagnostic_key}' must define a 'run' key at top level or nested level."
        )


def test_diag_keys_coverage_is_explicit():
    """Keep diagnostic key coverage intentional and reviewed."""
    discovered_keys = set()
    for config_path in _collection_files():
        cfg = load_yaml(str(config_path))
        discovered_keys.update((cfg.get("diagnostics") or {}).keys())

    unexpected = discovered_keys - KNOWN_DIAGNOSTIC_KEYS
    missing = KNOWN_DIAGNOSTIC_KEYS - discovered_keys

    assert not unexpected, (
        f"New diagnostic keys found in collection YAMLs: {sorted(unexpected)}. "
        "Add or adapt CLI tests, then update KNOWN_DIAGNOSTIC_KEYS intentionally."
    )
    assert not missing, (
        f"Known diagnostic keys not found in collection YAMLs: {sorted(missing)}. "
        "If removed intentionally, update KNOWN_DIAGNOSTIC_KEYS."
    )
