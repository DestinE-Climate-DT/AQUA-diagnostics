"""Contract tests for the analysis orchestration configuration."""

from pathlib import Path

import pytest

from aqua.core.util import load_yaml

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_ROOT = REPO_ROOT / "aqua" / "diagnostics"
ANALYSIS_CONFIG = DIAGNOSTICS_ROOT / "config" / "analysis" / "config.aqua-analysis.yaml"


def _load_analysis_config():
    return load_yaml(str(ANALYSIS_CONFIG))


def _flatten_run_groups(run_groups):
    return [diagnostic_group for run_group in run_groups for diagnostic_group in run_group]


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _resolve_collection_path(path_value):
    prefix = "${AQUA_CONFIG}/collections/"
    if not isinstance(path_value, str) or not path_value.startswith(prefix):
        return None
    relative_path = path_value[len(prefix) :]
    return DIAGNOSTICS_ROOT / "config" / "collections" / relative_path


def test_analysis_config_run_groups_reference_existing_diagnostics():
    """Entries in run groups must exist under diagnostics."""
    cfg = _load_analysis_config()
    run_groups = _flatten_run_groups(cfg["run"])
    diagnostics = cfg["diagnostics"]

    missing_groups = sorted(group for group in set(run_groups) if group not in diagnostics)
    assert not missing_groups, f"Run groups reference undefined diagnostics sections: {missing_groups}."


def test_analysis_config_diagnostics_reference_known_cli_entries():
    """Each diagnostic entry must reference a known CLI alias."""
    cfg = _load_analysis_config()
    cli_aliases = set(cfg["cli"])

    missing_cli_aliases = []
    for diagnostic_group, diagnostic_entries in cfg["diagnostics"].items():
        for cli_alias in diagnostic_entries:
            if cli_alias not in cli_aliases:
                missing_cli_aliases.append((diagnostic_group, cli_alias))

    assert not missing_cli_aliases, f"Diagnostics reference unknown cli aliases: {missing_cli_aliases}."


def test_analysis_config_cli_paths_exist():
    """CLI aliases must point to existing diagnostic CLI scripts."""
    cfg = _load_analysis_config()

    missing_paths = []
    for cli_alias, relative_cli_path in cfg["cli"].items():
        cli_path = DIAGNOSTICS_ROOT / relative_cli_path
        if not cli_path.is_file():
            missing_paths.append((cli_alias, str(relative_cli_path)))

    assert not missing_paths, f"CLI aliases point to missing scripts: {missing_paths}."


def test_analysis_config_collection_files_exist():
    """Collection config references in diagnostics must exist."""
    cfg = _load_analysis_config()

    missing_collection_files = []
    for diagnostic_group, diagnostic_entries in cfg["diagnostics"].items():
        for cli_alias, options in diagnostic_entries.items():
            config_option = options.get("config")
            if config_option is None:
                continue

            for config_path in _as_list(config_option):
                resolved_path = _resolve_collection_path(config_path)
                if resolved_path is not None and not resolved_path.is_file():
                    missing_collection_files.append((diagnostic_group, cli_alias, str(config_path)))

    assert not missing_collection_files, f"Diagnostics reference missing collection config files: {missing_collection_files}."
