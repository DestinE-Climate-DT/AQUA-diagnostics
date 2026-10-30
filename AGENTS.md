# AQUA-diagnostics Agent Guidelines

`aqua.diagnostics` holds the scientific diagnostics of the AQUA framework (ClimateDT monitoring).
Data access, fixing, regridding, statistics and plotting primitives live in **aqua-core** (`aqua.core`, repo
[DestinE-Climate-DT/AQUA](https://github.com/DestinE-Climate-DT/AQUA), usually checked out as `../AQUA`).
The core coding rules in its `AGENTS.md` apply here too; this file covers what is specific to diagnostics.

**Before writing code**: reuse `aqua.core` (Reader methods, `EvaluateFormula`, `aqua.core.graphics`, statistics,
Dask cluster management via `aqua.core.daskcluster.DaskCluster`), then `aqua.diagnostics.base`.
Anything generic (not tied to one tool) belongs in aqua-core, not here.

---

## Philosophy: Tools and Collections

- A **tool** (`aqua/diagnostics/<tool>/`, e.g. `biases`, `timeseries`) is generic, agnostic of the variable and of the
  scientific question, and must run on its own: from Python (compute and plot classes) or from its CLI with a config file.
- A **collection** (e.g. `atmosphere2d`, `ocean3d`) is a superset of tools set up for a scientific purpose by ad hoc
  config files in `config/collections/` (variables, references, periods, regions, plot limits). `aqua analysis` runs collections.
- Hence: variables, references, periods and regions come from the config, never from the tool code; outputs are named
  after the collection (`diagnostic_name`). The code still says "diagnostic" for both (`Diagnostic`, `diagnostics:` block).

---

## Repository Map

| Path | Content |
|---|---|
| `aqua/diagnostics/base/` | Shared infrastructure: `Diagnostic`, `DiagnosticCLI`, `OutputSaver`, `TitleBuilder`, `template_parse_arguments`, `load_var_config`, regions, `cli_setup_checker.py` |
| `aqua/diagnostics/<tool>/` | One package per tool: compute class(es), `plot_*.py`, `cli_<tool>.py`, optional `util.py`. Public classes re-exported in `aqua/diagnostics/__init__.py` with unique names (generic `Trends` vs `OceanTrends`) |
| `aqua/diagnostics/config/analysis/` | `aqua analysis` setup: `config.aqua-analysis.yaml`, `config.aqua-analysis-ensemble.yaml` (`aqua analysis ensemble`), `climatedt-experiment-kind.yaml` (Jinja values per `--kind`), `config.grouping.yaml` (dashboard grouping) |
| `aqua/diagnostics/config/collections/` | Run configs grouped by collection (`atmosphere2d`, `ocean3d`, ...): `jinja/*.j2` (used by `aqua analysis`) and `legacy/*.yaml` (same content, hard-coded dates) |
| `aqua/diagnostics/config/definitions/regions.yaml` | Central region registry, resolved by name |
| `aqua/diagnostics/config/tools/<tool>/` | Tool-specific static files (ECmean, teleconnection indices, SSH) |
| `aqua/diagnostics/templates/collections/` | Blank per-tool config templates for users |
| `frontier-diagnostics/` | **Deprecated** (`tropical_rainfall`, `tropical_cyclones`): not exported, not in `aqua analysis`, not tested in CI. Never use as a reference; modify only if the task explicitly targets them |
| `notebooks/diagnostics/<tool>/`, `docs/sphinx/source/` | Usage examples and per-tool documentation |

`config/` and `templates/` are installed into `$AQUA_CONFIG` (default `~/.aqua`) by `aqua install <machine>` through the
`aqua.plugins` entry point (`get_install_dirs()`). Repo edits are visible at runtime only with an editable install
(`aqua install <machine> --diagnostics <repo-path>`, which symlinks); otherwise reinstall.

Docs under `docs/sphinx/source/new_diagnostics/guidelines/` are partly outdated (e.g. `diagnostic_config.py`): trust the code.

---

## Anatomy of a Tool

Three layers, kept separate (see `histogram/` for a compact reference):

1. **Compute class** `MyDiag(Diagnostic)` — one instance per dataset (model and reference are two instances)
   - `super().retrieve(var=..., reader_kwargs=..., months_required=self.MINIMUM_MONTHS_REQUIRED)` sets `self.data`,
     `self.std_data`, `self.reader`, `self.catalog`, `self.realization`; clips dates to the available data and raises
     `NotEnoughDataError` when too short
   - Use `self.reader` methods (`select_area`, `fldmean`, `timmean`, ...) and `self._check_data()` for unit conversion
   - Regions by name via `self._set_region()` / `self.select_region()`: never hard-code lat/lon boxes
   - Store results as attributes; carry plot metadata in `AQUA_*` attrs (`AQUA_catalog/model/exp/realization/region`);
     after Dataset→DataArray, call `self._set_date_attrs()` again
   - `save_netcdf()` via `Diagnostic.save_netcdf`; `load()` must mirror it through one shared filename-key helper
     (this is what makes `output.plot_only` work)
2. **Plot class** `PlotMyDiag` (not a `Diagnostic`): takes computed data for one or more models plus reference,
   reads labels from attrs, draws with `aqua.core.graphics`, titles with `TitleBuilder`, writes a dashboard
   description, saves with `OutputSaver.save_figure(fig, diagnostic_product, extra_keys=..., metadata=...)`
3. **CLI** `cli_<tool>.py`: thin orchestration only (next section)

---

## CLI Contract

```python
def parse_arguments(args):
    parser = argparse.ArgumentParser(description="MyDiag CLI")
    # from aqua.diagnostics.base, not aqua.core.util: standard flags, add extra ones after this
    parser = template_parse_arguments(parser)
    return parser.parse_args(args)


def main(argv=None):
    args = parse_arguments(argv if argv is not None else sys.argv[1:])
    cli = DiagnosticCLI(args, diagnostic_name="mydiag", default_config="...").prepare()
    cli.open_dask_cluster()
    tool_dict = cli.config_dict["diagnostics"].get("mydiag", {})
    if tool_dict.get("run", False):
        ...  # loop over variables: catch NoDataError/NotEnoughDataError/KeyError/ValueError, log a warning, continue
    cli.close_dask_cluster()


if __name__ == "__main__":
    main()
```

- `prepare()` loads the YAML, merges CLI flags, sets the logger and `cli.outputdir/rebuild/save_format/save_netcdf/plot_only/dpi`
- CLI flags override **only `datasets[0]`**; `cli.dataset_args()` applies CLI regrid/dates, `cli.reference_args()` never does
- Per-variable settings: `load_var_config()` with precedence `params.default < params.<var> < inline dict`
- `--reader_kwargs` takes JSON and rejects keys that have a dedicated flag
- Always pass `--config`: the `default_config` fallback points to `$AQUA_CONFIG/collections/<tool>/`, which no longer exists
- `.j2` files contain unrendered `{{ ... }}` dates: for standalone runs use the `legacy/` YAML or a custom config

```bash
python aqua/diagnostics/biases/cli_biases.py -l INFO --outputdir /tmp/out \
  --config aqua/diagnostics/config/collections/legacy/atmosphere2d/config-atmosphere2d-era5.yaml \
  --catalog <catalog> --model <model> --exp <exp> --source <source>
```

### Collection YAML schema

```yaml
datasets: [{catalog, model, exp, source, regrid, reader_kwargs, startdate, enddate}]
references: [{catalog, model, exp, source, regrid, startdate, enddate}]
setup: {loglevel}
output: {outputdir, rebuild, save_netcdf, save_format, dpi, plot_only}
diagnostics:
  <tool_key>:                  # e.g. biases, timeseries, seaice_timeseries
    run: true                  # mandatory switch
    diagnostic_name: atmosphere2d   # output filename prefix = dashboard group
    variables: [...]
    formulae: [...]
    params: {default: {...}, <var>: {...}}
    plot_params: {default: {...}, <var>: {...}}
```

---

## `aqua analysis`

The runner lives in aqua-core (`aqua/core/console/analysis.py`, `aqua/core/analysis/analysis.py`); this repo provides its
configuration in `config/analysis/config.aqua-analysis.yaml`:

- `job`: defaults (catalog/model/exp/source, outputdir), `script_path_base: ${AQUA_DIAGNOSTICS}`, `experiment_kind`
- `cluster`: one shared Dask cluster passed to every tool as `--cluster` (unless `nocluster: true`, e.g. ECmean)
- `cli`: tool alias → script path relative to `aqua/diagnostics/`
- `run`: list of groups, run in sequence; collections inside a group run in parallel; tools inside a collection run serially
- `diagnostics.<collection>.<tool>`: `config` (one or a list, one run each), `outname`, `extra`, `nocluster`, `source_oce`, `nworkers`

For each tool it renders the `.j2` with the `--kind` section of `climatedt-experiment-kind.yaml` (strict Jinja), then runs
`python <cli> --model --exp --source --catalog --outputdir [--realization --startdate --enddate --regrid --cluster --source_oce] --config <rendered> -l <level>`.
Every CLI must accept these flags. Under `<outputdir>/<catalog>/<model>/<exp>/<realization>/`, outputs go to `<outname>/`
and each tool logs to `<collection>-<tool>[-N].log`.

- `job.run_setup_checker` (or `--setup_checker`) first runs `base/cli_setup_checker.py`: exit code 1 aborts the whole analysis
- **A failing tool does not stop the run**: check `aqua-analysis-report_<date>.md` (written at the end in the same folder) and the per-tool logs
- `aqua analysis ensemble` uses the same engine with `config.aqua-analysis-ensemble.yaml` (ensemble CLIs, no setup checker)

```bash
aqua analysis --catalog <cat> -m <model> -e <exp> -s <source> -o <outdir> [-k default|scenario|storyline] [--config <file>] [--serial] -l INFO
```

### Output naming and dashboard

Files are written as `<outputdir>/<png|pdf|svg|netcdf>/<diagnostic>.<diagnostic_product>.<catalog>.<model>.<exp>.<realization>[.<ref triplet>][.<extra_keys>].<ext>`,
where `<diagnostic>` is the YAML `diagnostic_name`. `config.grouping.yaml` maps these prefixes to dashboard sections:
renaming a `diagnostic_name` or `diagnostic_product` requires updating it.

### Adding a tool to the analysis

`cli` alias → `diagnostics.<collection>.<tool>` entry → config in **both** `collections/jinja/` and `collections/legacy/` →
collection in a `run` group → `config.grouping.yaml` entry. New `<tool_key>`s in collections must also be added to
`KNOWN_DIAGNOSTIC_KEYS` in `tests/config/test_diagnostic_key_coverage.py`.

---

## Testing

- Data: `AQUA_tests/` at the repo root (DVC/HTTP download) plus the `ci` catalog (`aqua add ci`); test experiments are loaded via
  session fixtures in `tests/conftest.py` (IFS `test-tco79`, FESOM `test-pi`, ICON `test-healpix`, NEMO `test-e_orca1`, ...).
  Reuse them; never add new data files
- Two layers (details and template in `CONTRIBUTING.md`):
  - `tests/<tool>/`: tool logic on real `ci` data, no CLI
  - `tests/cli/test_cli_<tool>.py`: fully mocked; patch classes at the CLI module path; use the `build_config` / `mock_cluster` fixtures
- `tests/config/`: contract tests keeping analysis config, CLI paths, collection files and Jinja rendering consistent
- **Every test module needs a marker** (`pytestmark = [pytest.mark.aqua, pytest.mark.diagnostics]`): CI runs
  `-m "aqua or diagnostics or ensemble"`, so unmarked tests are silently skipped. `frontier` (deprecated tests) is not run in CI
- Use `DPI`, `LOGLEVEL`, `APPROX_REL` from `tests/shared_constants.py`
- Write outputs to `tmp_path`; CI runs `pytest -n 4 --dist loadgroup`, so modules sharing setup or output files need
  `pytest.mark.xdist_group(name=...)` in their `pytestmark` (ignored without `--dist loadgroup`)

```bash
pytest tests/cli tests/config --no-cov        # fast, no data needed (~20 s)
pytest tests/<tool> --no-cov                  # needs AQUA_tests + ci catalog
pytest -n 4 --dist loadgroup -m "aqua or diagnostics or ensemble"   # as in CI
```

PR tests run only with the `run tests` label; the other suites and labels (`cross-check` for changes depending on
unreleased aqua-core, `nightly`, `release`, `conda-lock`) are described in `CONTRIBUTING.md` → "Testing and CI".

---

## Conventions

- Logging: `self.logger = log_configure(log_level=loglevel, log_name="ClassName")`, `loglevel` passed through every
  constructor, `%`-style arguments (do not add new f-string log calls), no `print()`
- Raise `aqua.core.exceptions` (`NoDataError`, `NotEnoughDataError`, `NoObservationError`, ...) in classes; catch them in CLIs
- Keep data lazy through the computation; materialise once at the end (plot/save), never inside per-variable loops
- Never create a Dask cluster or client in tool classes, nor assume a scheduler: CLIs use `cli.open_dask_cluster()` /
  `cli.close_dask_cluster()`, which connect to the `--cluster` passed by `aqua analysis` or start a private one with `--nworkers`
- No hard-coded paths: use `ConfigLocator().configdir` or `get_diagnostic_configpath(tool, folder="tools")`
- Google-style docstrings; Ruff (line length 127, `E,F,I,W,N`): `pre-commit run --files <files>` with the `ruff` pinned in the `style` extra
- New dependencies only after discussion: Python packages in `pyproject.toml`; conda-only binaries in `environment.yml`,
  `environment-dev.yml` and `environment-pypi.yml` (built on the aqua-core conda-forge package: do not re-list its
  dependencies); then regenerate `conda-linux-64.lock` with the `conda-lock` PR label
- Every PR adds a line under `## [Unreleased]` → "Complete list" in `CHANGELOG.md`: `- <Tool>: <change> (#PR)`
- Behaviour or config changes also update the relevant `docs/sphinx/source/` page and notebook
