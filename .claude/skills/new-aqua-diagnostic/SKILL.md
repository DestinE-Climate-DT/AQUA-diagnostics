---
name: new-aqua-diagnostic
description: Create or review an AQUA diagnostic (tool) in aqua-diagnostics following the repository conventions - analysis class, plot class, CLI, YAML config, tests, docs, notebook. Use when the user wants to add a new diagnostic/tool, bring an existing one in line with the others, or asks how a diagnostic must be structured. Ships a working, tested skeleton to copy (`dummy`) and a script that renames it.
---

# Writing an AQUA diagnostic

Conventions of `aqua-diagnostics` (aqua-core >= 1.2.1). The full text, with the rationale and the list of what is
not homogeneous yet, is [DIAGNOSTIC_GUIDELINES.md](../../../DIAGNOSTIC_GUIDELINES.md) at the repository root:
read the section named below when a rule here is not enough. API tables (`Diagnostic`, `OutputSaver`, aqua-core
helpers, config folders) are in [reference.md](reference.md).

**Where the code disagrees with the docs (Sphinx pages), follow the code.** The repository's `AGENTS.md` applies as
well (logging, lazy data, Dask, tests markers, no new dependencies without discussion): this skill adds the procedure.

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

## Open choices: settle them before starting

Several decisions are the user's, and they change what is written. Before step 2 of the workflow, go through the
list below, work out what the request and the repository already answer, and **ask the user for the rest in one
message** (all questions together, with your proposed default for each). Do not ask what is already answered, and do
not decide in silence: when you infer an answer, state it in one line so the user can correct it. Start only when
the choices are settled. Come back to the user if one turns up later (a collection that turns out to need a
second reference, a name that already exists).

| Choice | What to settle | Default to propose |
|---|---|---|
| **Tool name** | snake_case folder/module name and the CamelCase class (`lat_lon_profiles` / `LatLonProfiles`); must not clash with an existing tool or exported class | derived from what the tool computes |
| **What it computes** | the quantity, the variables (or formulae), the regions, the period, the figure type | ask: this has no default |
| **References** | none, optional or required (see [References](#references)); which dataset; **whether it needs std dates** for an uncertainty estimate | optional, no std dates |
| **Collection** | does it join an existing collection (which one, which references), start a new one, or stay standalone with only the template config? Edit `.j2` and `legacy` only if the user wants it | standalone, collection later |
| **`aqua analysis`** | register it in `config.aqua-analysis.yaml`, the `run` group, `config.grouping.yaml` (dashboard section) | only if it joins a collection |
| **Minimum data** | `MINIMUM_MONTHS_REQUIRED` and the scientific reason for it | the shortest period with a meaningful result |
| **Plot-only** | support `load()` (yes, unless there is a reason not to) | yes |
| **Scope of the deliverable** | code only, or also docs page, notebook, CHANGELOG (PR number) | code, tests, config template, docs page, CHANGELOG |
| **Generic plot** | if the figure is not in `aqua.core.graphics`: write it in the tool as a generic function and flag it for the core | yes |

Review mode (an existing diagnostic) needs none of this: report deviations and ask before fixing.

## Workflow

1. **Settle the open choices** with the user before touching any file (see the next section).
2. **Instantiate the skeleton** (never write the boilerplate by hand), from the root of the repository:
   ```bash
   <this skill's folder>/scripts/instantiate.sh <name> [ClassName]
   ```
   It copies `templates/` into the repo (code, config template, tests and the `.rst` page), renaming
   `dummy`/`Dummy`/`DUMMY` and refusing to overwrite anything.
   Then run `ruff check --fix` on the new files (the rename can change import order).
   Do **not** touch `aqua/diagnostics/dummy/cli_dummy.py`: that empty stub is used by the config tests.
3. **Apply the reference decision** to the skeleton, then **replace the science** in `<name>.py`
   (`compute_result`), keeping the structure. Adapt `plot_<name>.py` (figure type, title, caption), the CLI loops if
   the tool needs other axes, the config template.
4. **Wire it in** (checklist below): exports, `aqua analysis` and [collections](#adding-the-tool-to-a-collection),
   `config.grouping.yaml`, `KNOWN_DIAGNOSTIC_KEYS`, docs (fill the generated `.rst`, toctree, tables), notebook,
   CHANGELOG.
5. **Run the tests** of the new tool and the config tests, in an environment with aqua-core and the `ci` catalog:
   `pytest tests/<name> tests/cli/test_cli_<name>.py tests/config -q`.

For a review of an existing diagnostic, go through the checklist and the rules below, and report deviations
instead of fixing them silently.

## References

A **reference** is an observational or reanalysis dataset (or another experiment) the dataset is compared with.
Whether a tool needs one changes the plot class, the CLI and the config, so **settle it before writing them**:
ask the user, unless the context already answers it (the request says "against ERA5", the method is a bias or an
evaluation metric, a similar tool in the repo is described, the collection already has references).
Do not guess in silence: when you infer it, say so in one line.

| Case | Typical tools | What to do |
|---|---|---|
| **No reference** | indices, drift, stratification | Delete the lines marked `# REFERENCE` in the CLI and the plot class, the `references` block of the config template, the reference tests and the "reference" sentences of the `.rst`. |
| **Optional reference** (skeleton default) | timeseries, lat_lon_profiles, histogram | Keep everything as it is: the reference is drawn when `references` is configured, the figure is made without it otherwise. |
| **Required reference** | biases | As optional, but when `references` is empty or the reference has no data, log a warning and skip the variable instead of plotting; the plot class takes `ref_data` as mandatory and computes the difference. |

How the skeleton handles a reference, which holds in every case:

- A reference is **a second instance of the analysis class**, with its own `catalog/model/exp/source`; it is run and
  saved exactly like the dataset, and loaded in plot-only runs. The analysis class does not know it is a reference.
- The CLI builds it with `cli.reference_args(reference)` (the reference keeps its own `regrid`, `startdate`,
  `enddate`: the CLI options do not override them) and its own `reader_kwargs`, and uses only the first entry of
  `references` (warn if there are more).
- A failing reference (no data, not enough data, or any other error) must **not** cost the figure of the dataset:
  log it with `cli.log_failure` and plot without it, unless the reference is required.
- The plot class takes `ref_data` and reads the reference from its `AQUA_*` attributes, like the data. It names the
  reference (and its period) in the caption; `OutputSaver` receives `catalog_ref/model_ref/exp_ref`, so the figure
  filename carries the reference. For a comparison figure (bias, difference) pass the reference to `TitleBuilder`
  (`ref_catalog/ref_model/ref_exp`, `comparison`) as well.
- Dataset and reference must differ, otherwise their NetCDF files have the same name and overwrite each other.
- A reference may also need **std dates** (`std_startdate`/`std_enddate`): when the tool computes a standard
  deviation over a reference period to draw an uncertainty (shading, bands), the reference carries its own period,
  not the dataset's. See [reference.md](reference.md#references-and-std-dates).
- `references` is a top-level block of the config, never inside the tool block.

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
- Do not create a new instance or `AQUA_*` attribute unless strictly needed: reuse an existing one first
  (`grep -rhoE "AQUA_[a-zA-Z_]+" aqua/diagnostics | sort -u`), as `AGENTS.md` asks.
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
- Failure policy: wrap **each unit that can fail on its own** (variable, region, dataset, reference, sub-diagnostic)
  in `try/except Exception` and report it with `cli.log_failure(e, "variable %s in region %s", var, region)`, then
  continue, so one missing variable does not kill a collection run. `log_failure` logs `NoDataError` and
  `NotEnoughDataError` as a warning, anything else as an error with its traceback (collected by the `aqua analysis`
  report). A failing dataset is skipped and the other datasets are plotted.
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
`tests/config/test_diagnostic_key_coverage.py`, and map the new output prefixes in
`config/analysis/config.grouping.yaml`: the dashboard sections are built from the `diagnostic_name`/`diagnostic_product`
prefixes of the files, so a new tool in a collection needs an entry there (a section maps to patterns like
`atmosphere2d.timeseries`, i.e. `<diagnostic_name>.<diagnostic_product>`).

### Adding the tool to a collection

The standalone template (`templates/collections/config-<name>.yaml`) is only the starting point for a user.
The tool becomes a *diagnostic* when a **collection** configures it, and a collection exists in two parallel sets
of files that must stay aligned:

- `config/collections/jinja/<collection>/config-<collection>-<ref>.j2`: what `aqua analysis` runs. Dates are Jinja
  variables (`'{{ default.startdate }}'`, `'{{ <tool>.std_startdate }}'`) rendered per experiment kind.
- `config/collections/legacy/<collection>/config-<collection>-<ref>.yaml`: the same file in plain YAML with fixed
  dates, usable with `--config` outside `aqua analysis`.

Do it only when the user asks for it, or when the tool is meant for a collection from the start (ask which one).
Then: add a `<name>:` block to the files of that collection with the **same keys** as the template, `run: true`,
`diagnostic_name` set to the **collection name**, and the variables, regions and plot options of the scientific
purpose; keep `.j2` and `legacy` in step; register the tool in `config.aqua-analysis.yaml` with the list of `.j2`
files (one per reference, as `biases` does in `atmosphere2d`). A `.j2` file configures one reference: a collection
with two references (ERA5 and Berkeley Earth) has two files, and the tool lists both. The tests in `tests/config`
check that every `.j2` renders for every experiment kind and that the keys are known. Details in
[reference.md](reference.md).

## Tests, docs, notebook (guidelines §7)

- `tests/<name>/test_<name>.py`: real `ci` catalog (pass `catalog="ci"`), `pytestmark = [pytest.mark.aqua,
  pytest.mark.diagnostics]`, constants from `tests/shared_constants.py`, `tmp_path` as `outputdir`, and the
  **save/load round trip** (also with dates). `tests/cli/test_cli_<name>.py`: everything mocked with the
  `build_config` and `mock_cluster` fixtures; patch classes at the CLI module path.
- Docs: `instantiate.sh` already writes `docs/sphinx/source/stateoftheart_diagnostics/<name>.rst` from
  `templates/docs/.../dummy.rst`, laid out like every page of that folder (description, classes, files, inputs,
  Python and CLI usage, config, collections, output, observations, notebooks, references, authors, API). Fill it in
  (the `..` comments guide the author; delete them), add it to the toctree, the minimum-data table and, if `load()`
  exists, the plot-only list of `index.rst`. The page keeps the same section order on purpose, so the pages can be
  updated and homogenised together later. CHANGELOG entry: a line under `Unreleased` -> "Complete list" of `CHANGELOG.md`, `- <Tool>: <change> (#PR)`.
- Notebook `notebooks/diagnostics/<name>/<name>.ipynb` with the Python API (analysis -> plot) on data available
  on a common machine.

## Checklist (guidelines §9)

- [ ] `aqua/diagnostics/<name>/` with analysis class (subclass of `Diagnostic`), plot class, CLI, `__init__.py` with `__all__`
- [ ] Classes exported in `aqua/diagnostics/__init__.py`
- [ ] `MINIMUM_MONTHS_REQUIRED` defined and passed to `retrieve()`
- [ ] Results named, in memory, with `AQUA_*` provenance attributes
- [ ] `save_netcdf()` and `load()` share one `_extra_keys` helper; `load()` passes `startdate`/`enddate`
- [ ] Plot class reads metadata from attributes, uses `TitleBuilder`, writes a description, saves with `OutputSaver`
- [ ] CLI with `main(argv=None)`, `template_parse_arguments`, `DiagnosticCLI`, `cli.log_failure` per variable/region/dataset, `plot_only`, cluster closed in `finally`
- [ ] Template `templates/collections/config-<name>.yaml` (+ `.j2` collection config if it runs in `aqua analysis`)
- [ ] Entry in `config.aqua-analysis.yaml`, in `config.grouping.yaml` and in `KNOWN_DIAGNOSTIC_KEYS`
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
| `docs/sphinx/source/stateoftheart_diagnostics/dummy.rst` | the documentation page, to fill in |
| `tests/dummy/test_dummy.py` | result, region, minimum data, round trip, period check, plot, plot with reference |
| `tests/cli/test_cli_dummy.py` | parsing, disabled, full pipeline, with/without/missing reference, plot-only, failure skip |

Lines that exist only for the reference carry the marker `# REFERENCE`. Known limits of the skeleton: one dataset
and at most one reference in the CLI; no `formulae`; no `.j2`/legacy collection config, `aqua analysis` wiring or
notebook (step 4).

## Portability

The content is plain Markdown plus a bash script: nothing in it depends on Claude. Only the *discovery* is
tool-specific: `.claude/skills/<name>/SKILL.md` is read by Claude Code. The `SKILL.md` format (a `name` and a
`description` in the frontmatter) follows the Agent Skills convention that other agents read from their own
folders, and agents that do not read skills can be pointed at this file from their instruction file
(`AGENTS.md`, `GEMINI.md`, Copilot or Cursor rules). Keep the body free of tool-specific syntax so it stays valid
for all of them.
