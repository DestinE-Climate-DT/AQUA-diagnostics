---
name: new-aqua-diagnostic
description: Create or review an AQUA diagnostic (tool) in aqua-diagnostics following the repository conventions - analysis class, plot class, CLI, YAML config, tests, docs, notebook. Use when the user wants to add a new diagnostic/tool, bring an existing one in line with the others, or asks how a diagnostic must be structured. Ships a working, tested skeleton to copy (`dummy`) and a script that renames it.
---

# Writing an AQUA diagnostic

Conventions of `aqua-diagnostics` (aqua-core >= 1.2.1). The full text, with the rationale and the list of what is
not homogeneous yet, is [DIAGNOSTIC_GUIDELINES.md](../../../DIAGNOSTIC_GUIDELINES.md) at the repository root:
read the section named below when a rule here is not enough. API tables (`Diagnostic`, `OutputSaver`, aqua-core
helpers, config folders) are in [reference.md](reference.md).

**Where the code disagrees with the docs (Sphinx pages), follow the code.**

## 0. Tool vs diagnostic

What lives in `aqua/diagnostics/<name>/` is a **tool**: generic, agnostic of variable and scientific question
(`timeseries`, `biases`, ...). A **diagnostic** is a *collection*: tools configured for a purpose (`atmosphere2d`
= `biases` + `timeseries` + `lat_lon_profiles` + `trends` on atmospheric fields vs ERA5). The code still says
"diagnostic" for both (`Diagnostic` base class, `diagnostics:` block, `diagnostic_name`).

Consequences, always:

- Variable, units, region, period and reference come **from the configuration**, never from the code.
- Scientific choices (variables, observations, periods, colour limits) live in the collection config.
- Output files are named after the **collection** (`diagnostic_name`), not the tool, so choose
  `diagnostic_product` values that say which tool produced a file.

## Workflow

1. **Clarify** with the user, only what is not already clear: tool name (snake_case, e.g. `lat_lon_profiles`),
   what is computed, inputs (variables, formulae, regions), whether a reference is needed, whether it must run in
   `aqua analysis` (and in which collection), minimum data for a scientific reason.
2. **Instantiate the skeleton** (never write the boilerplate by hand):
   ```bash
   .claude/skills/new-aqua-diagnostic/scripts/instantiate.sh <name> [ClassName]
   ```
   It copies `templates/` into the repo, renaming `dummy`/`Dummy`/`DUMMY` and refusing to overwrite anything.
   Then run `ruff check --fix` on the new files (the rename can change import order).
   Do **not** touch `aqua/diagnostics/dummy/cli_dummy.py`: that empty stub is used by the config tests.
3. **Replace the science** in `<name>.py` (`compute_result`), keep the structure. Adapt `plot_<name>.py`
   (figure type, title, caption), the CLI loops if the tool needs other axes, the config template.
4. **Wire it in** (checklist below): exports, `aqua analysis`, `KNOWN_DIAGNOSTIC_KEYS`, docs, notebook, CHANGELOG.
5. **Run the tests** of the new tool and the config tests, in an environment with aqua-core and the `ci` catalog:
   `pytest tests/<name> tests/cli/test_cli_<name>.py tests/config -q`.

For a review of an existing diagnostic, go through the checklist and the rules below, and report deviations
instead of fixing them silently.

## The three layers plus a config

```
YAML config ─► cli_<name>.py (DiagnosticCLI, no science) ─► analysis class (Diagnostic subclass, ONE dataset)
                                                              │ NetCDF via OutputSaver
                                                              ▼  load()
                                                           plot class (no data access, multi-model aware) ─► PNG/PDF/SVG
```

Files (snake_case folder and modules):

```
aqua/diagnostics/<name>/{__init__.py, <name>.py, plot_<name>.py, cli_<name>.py, [util.py, base.py, README.md]}
aqua/diagnostics/templates/collections/config-<name>.yaml        # standalone template
aqua/diagnostics/config/collections/jinja/<collection>/*.j2      # used by aqua analysis
tests/<name>/test_<name>.py   tests/cli/test_cli_<name>.py
docs/sphinx/source/stateoftheart_diagnostics/<name>.rst          # copy of dummy.rst
notebooks/diagnostics/<name>/<name>.ipynb
```

Naming: analysis class = noun in CamelCase (`Timeseries`); plot class = `Plot` + noun; workflow `run()`;
steps `compute_<what>()`; read back `load()`; write `save_netcdf()`; plot helpers `get_data_info()`,
`set_title()`, `set_description()`, `set_data_labels()`, `set_ref_label()`, `save_plot()`;
minimum data `MINIMUM_MONTHS_REQUIRED`. Export public classes in `aqua/diagnostics/__init__.py` (watch name
clashes, e.g. the ocean `Trends` are re-exported as `OceanTrends`).

## Analysis class rules (guidelines §3)

- Subclass `aqua.diagnostics.base.Diagnostic`; keep the common constructor arguments
  (`model, exp, source, catalog, regrid, startdate, enddate, loglevel`), add only what is really specific.
  Call `super().__init__`, then `log_configure(log_level=loglevel, log_name="<ClassName>")`.
  Initialise **every result attribute to `None`**.
- Accept `diagnostic_name` (default = tool name) and use it as the `diagnostic` of every output file.
- Data via `self.retrieve(var=..., reader_kwargs=..., months_required=...)` (or `self._retrieve` for complex
  cases). **Never build a `Reader` yourself.**
- `MINIMUM_MONTHS_REQUIRED` chosen for a scientific reason (12 for a seasonal cycle, years for ENSO), with a
  comment; add it to the minimum-data table in `stateoftheart_diagnostics/index.rst`.
- Use Reader operators on `self.reader` (`fldmean`, `timmean`, `fldstat`, `timstd`, `select_area`, `regrid`,
  `vertinterp`, `detrend`, `histogram`), do not reimplement them.
- Formulae (`formulae:` in YAML): retrieve all variables, evaluate with
  `aqua.core.fixer.EvaluateFormula(...).evaluate()`; see `BaseMixin.retrieve` in `timeseries/base.py`.
- Results have `long_name` (fall back to the variable), `short_name`, `units`, a name, and are in memory
  (`.load()`).
- **Provenance travels with the result** as `AQUA_*` attributes (`AQUA_catalog/model/exp/source/realization`
  from the Reader, `AQUA_startdate/enddate` from `retrieve`, `AQUA_region`, and `AQUA_<anything>` for whatever the
  title, plot or caption needs: season, level, reference period...). The plot class must not receive any of it
  through separate arguments. xarray drops attributes: copy them explicitly where you build new objects.
- Errors: `NoDataError`, `NotEnoughDataError` (`aqua.core.exceptions`) or `ValueError`/`KeyError` with a clear
  message. The CLI decides whether to skip.
- **Plot-only support (§3.3)**: one private `_extra_keys(...)` helper used by both `save_netcdf()` and `load()`
  (a mismatch only breaks plot-only runs, silently); `load()` takes the same identifying arguments as `run()`,
  sets `self.realization`, resolves the region with `_set_region`, passes `startdate/enddate`; a missing file
  leaves the attribute untouched.
- Gotcha: `load_netcdf()` compares `startdate/enddate` with the **dataset-level** attributes, but a saved bare
  `DataArray` keeps them on the variable, so loading with dates returns `None`. The skeleton saves
  `result.to_dataset()` with the attributes copied to `ds.attrs`; keep that, and keep `test_load_checks_the_period`.
- Mutable defaults: use `reader_kwargs: dict = None`, never `{}`.

## Plot class rules (guidelines §4)

- Plain class (no `Diagnostic` inheritance). The constructor receives **results** (DataArray/Dataset or lists,
  plus optional references), never a dataset to retrieve.
- `get_data_info()` reads catalog/model/exp/realization/region/dates from `AQUA_*`.
- Titles with `TitleBuilder(...).generate(max_chars=...)`; `set_description()` writes a full-sentence caption
  (what, variable, region, datasets and periods, reference, meaning of shading) embedded in the figure metadata
  under the key `"description"`.
- Draw with `aqua.core.graphics` (`plot_timeseries`, `plot_hovmoller`, `plot_single_map`, ... they take and return
  `fig, ax`). If the plot does not exist, write a **generic** function (data in, `fig, ax` out, no
  diagnostic-specific logic), say so in the PR and open an issue on AQUA to move it to the core.
- Save through one `OutputSaver` initialised with **all** plotted models and references:
  `save_figure(fig, diagnostic_product, extra_keys, metadata={"description": ...}, extension=save_format,
  dpi=dpi, rebuild=rebuild)`. Pass a single realization as a value, not a one-element list (the filename would
  contain `['r1']`). Honour `save_format`, `dpi`, `rebuild` from the CLI; `plt.close(fig)` unless `show=True`.
- Helpers: `unit_to_latex`, `cbar_get_label`, `evaluate_colorbar_limits(maps, sym=True)`, `get_projection`.

## CLI rules (guidelines §5)

- `cli_<name>.py` exposes `parse_arguments(args)` and `main(argv=None)`; only `if __name__ == "__main__": main()`
  at the bottom.
- Arguments start from `template_parse_arguments(parser)` of `aqua.diagnostics.base` (**not** the one in
  `aqua.core.util`); extra flags only if necessary, passed as `extra_parser_options`.
- `DiagnosticCLI(args, diagnostic_name=..., default_config=...).prepare()`, then `cli.open_dask_cluster()` and
  `cli.close_dask_cluster()` in a `finally`.
- Datasets with `cli.dataset_args(dataset)`, references with `cli.reference_args(reference)`; always pass
  `reader_kwargs=dataset.get("reader_kwargs") or {}`.
- Honour `diagnostics.<key>.run` (exit cleanly when false or missing) and `diagnostic_name`.
- Per-variable settings with `load_var_config(config_dict, var, diagnostic=<key>)`
  (`params.default` < `params.<var>` < inline; regions default to `[None]`).
- Failure policy: catch `NoDataError, NotEnoughDataError, KeyError, ValueError` **per variable/region**, warn and
  continue, so one missing variable does not kill a collection run.
- Plot-only: `if not cli.plot_only: run`, `if cli.plot_only or cli.save_netcdf: load`, plot only if results exist.
- No science in the CLI.
- Contract with `aqua analysis`: the CLI is launched with the whole `template_parse_arguments` set, must work with
  a shared Dask cluster (`--cluster`) and write only below `--outputdir`.

## Configuration rules (guidelines §6)

Top-level blocks `datasets`, `references` (optional), `setup`, `output`, `diagnostics.<key>`. Two keys are
**mandatory** in every tool block: `run` (bool) and `diagnostic_name` (str, the collection name in a collection,
the tool name in a standalone template). Computation options in `params` (`default` + per-variable overrides),
figure options in `plot_params`. Regions are names from `config/definitions/regions.yaml`: add new ones there,
never hard-code limits. Where files live: see [reference.md](reference.md).

Plugging into `aqua analysis` (`config/analysis/config.aqua-analysis.yaml`): add the CLI under `cli:`
(path relative to `aqua/diagnostics`), add it to a collection under `diagnostics:` with its `config:` (`.j2`),
make sure the collection is in a `run:` group. Add the key to `KNOWN_DIAGNOSTIC_KEYS` in
`tests/config/test_diagnostic_key_coverage.py`.

## Tests, docs, notebook (guidelines §7)

- `tests/<name>/test_<name>.py`: real `ci` catalog (pass `catalog="ci"`), `pytestmark = [pytest.mark.aqua,
  pytest.mark.diagnostics]`, constants from `tests/shared_constants.py`, `tmp_path` as `outputdir`, and the
  **save/load round trip** (also with dates). `tests/cli/test_cli_<name>.py`: everything mocked with the
  `build_config` and `mock_cluster` fixtures; patch classes at the CLI module path.
- Docs: copy `docs/sphinx/source/stateoftheart_diagnostics/dummy.rst` to `<name>.rst`, add it to the toctree, the
  minimum-data table and (if `load()` exists) the plot-only list of `index.rst`. CHANGELOG entry.
- Notebook `notebooks/diagnostics/<name>/<name>.ipynb` with the Python API (analysis -> plot) on data available
  on a common machine.

## Checklist (guidelines §9)

- [ ] `aqua/diagnostics/<name>/` with analysis class (subclass of `Diagnostic`), plot class, CLI, `__init__.py` with `__all__`
- [ ] Classes exported in `aqua/diagnostics/__init__.py`
- [ ] `MINIMUM_MONTHS_REQUIRED` defined and passed to `retrieve()`
- [ ] Results named, in memory, with `AQUA_*` provenance attributes
- [ ] `save_netcdf()` and `load()` share one `_extra_keys` helper; `load()` passes `startdate`/`enddate`
- [ ] Plot class reads metadata from attributes, uses `TitleBuilder`, writes a description, saves with `OutputSaver`
- [ ] CLI with `main(argv=None)`, `template_parse_arguments`, `DiagnosticCLI`, per-variable error handling, `plot_only`, cluster closed in `finally`
- [ ] Template `templates/collections/config-<name>.yaml` (+ `.j2` collection config if it runs in `aqua analysis`)
- [ ] Entry in `config.aqua-analysis.yaml` and in `KNOWN_DIAGNOSTIC_KEYS`
- [ ] Diagnostic tests (with save/load round trip) and CLI tests
- [ ] `.rst` page from `dummy.rst`, toctree, minimum-data table, plot-only list
- [ ] Notebook in `notebooks/diagnostics/<name>/`
- [ ] CHANGELOG entry; new dependencies added to `pyproject.toml` and environment files

## The skeleton

`templates/` mirrors the repository tree. It computes the annual mean of a variable averaged over a region, and
has been run against the `ci` catalog (ERA5 `era5-hpz3`, variable `tcc`): tests pass, the CLI runs end to end,
plot-only runs reproduce the figures. Keep its shape, replace the science.

| Template | Becomes |
|---|---|
| `aqua/diagnostics/dummy/dummy.py` | analysis class: `run`, `compute_result`, `_extra_keys`, `save_netcdf`, `load` |
| `aqua/diagnostics/dummy/plot_dummy.py` | plot class: attributes -> title/caption -> `plot_timeseries` -> `OutputSaver` |
| `aqua/diagnostics/dummy/cli_dummy.py` | CLI: variable -> region loop, failure policy, plot-only |
| `aqua/diagnostics/templates/collections/config-dummy.yaml` | standalone config template |
| `tests/dummy/test_dummy.py` | result, region, minimum data, round trip, period check, plot |
| `tests/cli/test_cli_dummy.py` | parsing, disabled, full pipeline, plot-only, failure skip |

Known limits of the skeleton: one dataset and no reference in the CLI (the plot class already accepts one);
no `formulae`; no `.j2` collection config, `aqua analysis` wiring, `.rst` page or notebook (steps 4 above).
