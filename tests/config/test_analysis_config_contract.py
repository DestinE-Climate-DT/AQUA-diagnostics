"""Contract tests for the analysis orchestration configuration."""

from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined
from ruamel.yaml import YAML

from aqua.core.util import load_yaml

pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_ROOT = REPO_ROOT / "aqua" / "diagnostics"
ANALYSIS_CONFIG = DIAGNOSTICS_ROOT / "config" / "analysis" / "config.aqua-analysis.yaml"
EXPERIMENT_KIND_CONFIG = DIAGNOSTICS_ROOT / "config" / "analysis" / "climatedt-experiment-kind.yaml"
JINJA_COLLECTIONS_DIR = DIAGNOSTICS_ROOT / "config" / "collections" / "jinja"


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


# TODO(aqua-core #2834): this local renderer exists only because the CI-pinned aqua-core (from PyPI)
# does not yet expose load_yaml(strict=...). Once the pinned aqua-core ships #2834, remove this block
# and the jinja2/ruamel imports, and render via the real production function:
#     rendered = load_yaml(str(template), definitions=kind_definitions, strict=True)
# Mirror of aqua-core's strict jinja env (StrictUndefined + trim/lstrip/keep):
_STRICT_JINJA_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)


# Kind resolution: use the production code from aqua-core when available.
# TODO(aqua-core #3266): drop the fallback mirror once the pinned aqua-core ships resolve_experiment_kind.
try:
    from aqua.core.analysis import resolve_experiment_kind
except ImportError:

    def _deep_merge(base, override):
        """Recursively merge ``override`` on top of ``base`` without modifying the inputs."""
        merged = dict(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def resolve_experiment_kind(kinds, kind):
        """Mirror of ``aqua.core.analysis.resolve_experiment_kind`` for older aqua-core."""
        if kind not in kinds:
            raise KeyError(f"Experiment kind '{kind}' not found.")
        resolved = _deep_merge(kinds["default"], kinds[kind])
        if "kind" in resolved:
            raise ValueError("'kind' is a reserved key in the experiment kind file.")
        resolved["kind"] = kind
        return resolved


def _resolved_experiment_kinds():
    """Resolve every kind of the experiment kind file, as passed to the templates by ``aqua analysis``."""
    experiment_kinds = load_yaml(str(EXPERIMENT_KIND_CONFIG))
    return {name: resolve_experiment_kind(experiment_kinds, name) for name in experiment_kinds}


def _render_template_strict(template_path, definitions):
    """Render a jinja template with StrictUndefined and parse the result as YAML."""
    rendered = _STRICT_JINJA_ENV.from_string(template_path.read_text()).render(definitions)
    return YAML(typ="rt").load(rendered)


def test_jinja_collection_templates_render_with_experiment_kinds():
    """Every jinja collection template must render to a valid config for all experiment kinds.

    Mirrors production (``load_yaml(strict=True)``): a variable missing from the kind raises
    ``UndefinedError``, only logged, which silently skips the collection at runtime. Guards
    render-ability and YAML validity, not the rendered values.
    """
    experiment_kinds = _resolved_experiment_kinds()
    templates = sorted(JINJA_COLLECTIONS_DIR.glob("**/*.j2"))
    assert templates, f"No jinja templates found under {JINJA_COLLECTIONS_DIR}."

    render_failures = []
    for template in templates:
        relative_template = template.relative_to(JINJA_COLLECTIONS_DIR)
        for kind_name, kind_definitions in experiment_kinds.items():
            label = f"{relative_template} [{kind_name}]"
            # Catch broadly: both UndefinedError and YAML parse errors are silent failures at runtime.
            try:
                rendered = _render_template_strict(template, kind_definitions)
            except Exception as error:
                render_failures.append(f"{label}: {type(error).__name__}: {error}")
                continue
            if not isinstance(rendered, dict):
                render_failures.append(f"{label}: rendered to {type(rendered).__name__}, not a mapping")

    assert not render_failures, "Jinja templates failed to render to a valid config:\n" + "\n".join(render_failures)


def _assert_same_keys(resolved, default, label):
    """Recursively check that ``resolved`` has exactly the keys of ``default``."""
    assert resolved.keys() == default.keys(), label
    for key, value in default.items():
        if isinstance(value, dict):
            _assert_same_keys(resolved[key], value, f"{label}.{key}")


def test_experiment_kinds_define_same_keys_as_default():
    """Overrides must not introduce keys unknown to ``default`` (typos would be silently unused)."""
    default = load_yaml(str(EXPERIMENT_KIND_CONFIG))["default"]
    for kind_name, resolved in _resolved_experiment_kinds().items():
        assert resolved.pop("kind") == kind_name
        _assert_same_keys(resolved, default, kind_name)


def test_sea_ice_concentration_reference_per_kind():
    """Storylines use OSI-SAF AMSR, HighResMIP ERA5 (back to 1950), the other kinds OSI-SAF SSMIS."""
    expected_exp = {"storyline": "osi-saf-amsr", "highresmip-hist-cmip6": "era5", "highresmip-hist-cmip7": "era5"}
    for kind_name, resolved in _resolved_experiment_kinds().items():
        for slot in ("siconc_nh", "siconc_sh"):
            assert resolved["references"][slot]["exp"] == expected_exp.get(kind_name, "osi-saf-ssmis"), f"{kind_name}.{slot}"


@pytest.mark.parametrize("kind_name", ["highresmip-hist-cmip6", "highresmip-hist-cmip7"])
def test_highresmip_references_cover_1950(kind_name):
    """HighResMIP kinds start in 1950 and replace the satellite-era references with ERA5."""
    resolved = _resolved_experiment_kinds()[kind_name]
    references = resolved["references"]
    for slot in ("radiation_toa", "radiation_surface", "sst", "precipitation", "siconc_nh", "siconc_sh"):
        assert references[slot] == references["atmosphere"], slot
    for block in ("period", "radiation", "timeseries"):
        assert resolved[block]["startdate"] == resolved[block]["std_startdate"] == "1950-01-01", block


@pytest.mark.parametrize("collection", ["radiation_toa", "radiation_surface"])
def test_boxplots_skip_duplicate_reference(collection):
    """Boxplots list the satellite reference only when it differs from the atmosphere one (ERA5)."""
    template = JINJA_COLLECTIONS_DIR / collection / f"config-{collection}-boxplots.j2"
    for kind_name, resolved in _resolved_experiment_kinds().items():
        references = _render_template_strict(template, resolved)["references"]
        expected = 1 if kind_name.startswith("highresmip") else 2
        assert len(references) == expected, kind_name
        assert references[0]["exp"] == "era5", kind_name
