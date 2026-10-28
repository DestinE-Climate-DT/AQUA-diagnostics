# Contributing guide

In this guide you will get an overview of the contribution workflow in AQUA-diagnostics, from opening an *issue*, creating a *Pull Request* (PR), reviewing, and merging the PR.

We welcome contributions to the AQUA-diagnostics project in many forms, and there's always plenty to do!

## Reporting issues

Before opening an issue, please [search if the issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments). If it does, please add a comment to the existing issue instead of opening a new one.

As a general rule, if you are unsure, please open an issue anyway and we will help you.
There is no automatic assignment of issues to anyone. If it is a bug, please fill all the requested fields and verify that reproducible code is reported.
If you open a pull request to close some issues, please reference the issue it solves in the description.

### Reporting bugs

If you find a bug in the source code, you can help us by opening an issue in the AQUA-diagnostics repository.
If you have a solution to it, you can submit a Pull Request with a fix.
Please use the **Bug report** issue template, which sets the `Bug` issue type, and the `fix :hammer_and_wrench:` label for Pull Requests with a fix.

When describing the bug, please include as much information as possible. This includes:
- short description of the bug
- steps to reproduce the bug
- catalog you are using
- machine you are using

## Pull Requests

The contribution workflow is based on Pull Requests (PR).
A Pull Request is a request to merge a set of changes into the `main` branch of the repository.
It is the main way to contribute to AQUA-diagnostics.

### Creating a Pull Request

Pull requests can be created directly in the AQUA-diagnostics repository, creating your own fork of the repository is not mandatory.

When creating a Pull Request, please make sure to:
- add a meaningful title and description
- reference the issue it solves in the description, if any
- start from the `main` branch if your Pull Request wants to be merged in the `main` branch
- point to the correct branch

If your Pull Request is adding some new dependencies, please make sure to state it in the description.

### Finalizing a Pull Request

Adding the `run tests` label to the Pull Request will activate the CI tests at the next push (see [Testing and CI](#testing-and-ci) for the other labels).
Adding the `ready to merge` label to the Pull Request will indicate that it is ready to be reviewed and hopefully merged in the opinion of the author.

Before asking for a review, please make sure to:
- be up to date with the `main` branch
- run the tests successfully
- if a new dependency has been added to the framework, please make sure to update the relevant environment and packaging files (`environment.yml`, `environment-dev.yml`, and/or `pyproject.toml`)
- if the environment has been modified, please make sure to update `environment_lumi.yml` and `pip_lumi.txt` accordingly.
- if a new feature has been added, please make sure to update the documentation accordingly
- add docstrings to your code

Do not merge your Pull Request yourself, it will be merged by the AQUA-diagnostics team.

### Testing and CI

CI is organised in **suites**, as in [aqua-core](https://github.com/DestinE-Climate-DT/AQUA/blob/main/CONTRIBUTING.md#testing):
each suite tests a different way of installing AQUA-diagnostics, so that a broken dependency, an aqua-core change or a
packaging problem is caught by the suite that owns it, not by the PR tests. Tests are run with `pytest -m "aqua or diagnostics or ensemble"`.

| Workflow | Suite | When it runs | Platforms | Python | Install mode | Coverage |
|---|---|---|---|---|---|---|
| `aqua-diagnostics.yml` | PR tests | PR with a test label, Dependabot PRs, push to `main`, manual | `ubuntu-latest` in FDB container | 3.14 | `local` | yes (not on Dependabot) |
| `aqua-diagnostics-nightly.yml` | Nightly | every night (04:00 UTC), PR with `nightly` label, Dependabot pip/conda PRs, manual (with aqua-core ref) | `ubuntu-latest` in FDB container, `xcode-27` | 3.14 | `environment` and `cross-check` | no |
| `aqua-diagnostics-cross-check.yml` | Cross-check | PR with `cross-check` label, manual (with aqua-core ref) | `ubuntu-latest` in FDB container | 3.14 | `cross-check` | no |
| `aqua-diagnostics-release.yml` | Release | every Monday (06:00 UTC), PR with `release` label, manual | `ubuntu-22.04`, `ubuntu-26.04`, `macos-15`, `xcode-27` | 3.12, 3.13, 3.14 | `release` | no |
| `aqua-diagnostics-pypi.yml` | PyPI | every Tuesday (04:00 UTC), push to `pypi-test`, manual | `ubuntu-latest`, `macos-latest` | 3.12, 3.13, 3.14 | `pypi` | no |
| `aqua-diagnostics-conda.yml` | Conda | manual only (until aqua-diagnostics is on conda-forge) | `ubuntu-latest`, `macos-latest` | 3.12, 3.13, 3.14 | `conda` | no |

The FDB container is only the build environment on Linux: AQUA-diagnostics has no FDB tests.

The install modes are:

- `local`: checkout + `conda-linux-64.lock`, i.e. the exact locked environment with the pinned aqua-core release. Linux only.
- `environment`: checkout + `environment.yml`, unpinned, with the latest aqua-core release: catches breakages from new upstream releases.
- `cross-check`: as `environment`, then aqua-core is installed from the [AQUA](https://github.com/DestinE-Climate-DT/AQUA)
  repository at a chosen reference (default `main`): catches integration breaks with the current core codebase before the next aqua-core release.
- `release`: `environment-pypi.yml` (aqua-core and conda dependencies from conda-forge) + `pip install -e .[core,tests]` of the checkout: what a user gets when installing from source.
- `pypi`: `environment-pypi.yml` + `aqua-diagnostics[core,tests]` from PyPI, tested against the matching release tag.
- `conda`: `aqua-diagnostics` and the test dependencies from conda-forge, tested against the matching release tag.

Comparing the two nightly modes separates aqua-core regressions (`cross-check` fails, `environment` passes) from dependency or diagnostics issues (both fail).

**Labels**

| Label | Effect |
|---|---|
| `run tests` 🏃, `ready for review` 🏂, `ready to merge` 🌈 | run `aqua-diagnostics.yml` |
| `nightly` 🌙 | run `aqua-diagnostics-nightly.yml` (use it when changing dependencies or platform-specific code) |
| `cross-check` | run `aqua-diagnostics-cross-check.yml` against aqua-core `main` (use it when your change depends on aqua-core features not yet released) |
| `release` ⛵ | run the full `aqua-diagnostics-release.yml` matrix (use it on release PRs) |
| `conda-lock` 🐍 | regenerate the lock file against the PR branch |

Once a label is set, the corresponding suite runs again at every push. Adding a label triggers a run only for the label
just added: adding an unrelated label does not restart running tests. Labels are matched by prefix.

> **Note:** every label added to a PR starts a run of each suite, skipped when the label is not the suite's own.
> The PR checks panel shows only the most recent run of each workflow, so when several labels are added at once,
> or a label is added after the tests ran, a suite may appear as *skipped* even though its real run is running or
> has finished. The real results are in the **Actions** tab and still count for the PR status and the required check.
> To refresh the panel, push a new commit: a push starts one run per suite, so the panel shows the real ones. To test against another aqua-core branch, tag or
commit, run **Actions** → **AQUA-diagnostics cross-check** → **Run workflow** and set the aqua-core ref.

**How the CI is built**

All suites call `aqua-diagnostics-test-reusable.yml`, which runs one job (one platform, one Python, one install mode);
callers only define the matrix and the inputs. Test data, AQUA configuration and reports use the composite actions of
aqua-core (`setup-test-data`, `setup-aqua`, `pytest-reports`), checked out at `aqua-actions-ref` (default `main`).
Each test has an 800 s timeout (`pytest-timeout`, input `pytest-timeout-seconds`) and the test step a 40 min timeout. Every job uploads an artifact `aqua-test-artifacts-<install-mode>-<platform>-py<version>`
with `junit.xml`, `aqua_pytest.log`, `micromamba_list.txt` and `pip_freeze.txt`.
`AQUA_GITHUB_PAT` is used for all GitHub access (fallback: `github.token`). To test changes to `aqua-diagnostics-pypi.yml`,
push to a branch named `pypi-test`.

When a scheduled or manual run of the Nightly, Release or PyPI suite fails, the aqua-core `report-failure-reusable.yml`
opens an issue labelled `ci-failure:<suite>` 🤡 listing the broken tests, or comments on the open one.

### Suggesting enhancements

Enhancements of existing features or new features may be suggested by opening an issue in the AQUA-diagnostics repository.
Please use the **Feature request** issue template, which sets the `Feature` issue type.

### Writing tests

Tests live under `tests/` and use `pytest` (+ `pytest-mock` for the `mocker` fixture). Run the full suite from the repo root with:

```bash
pytest tests/
```

Two kinds of tests cover a diagnostic:

1. **Diagnostic-logic tests** (`tests/<diagnostic>/test_<diagnostic>.py`) — exercise the diagnostic classes (`Biases`, `SeaIce`, …) against the real `ci` catalog. These validate the *scientific* behaviour and data handling.
2. **CLI tests** (`tests/cli/test_cli_<diagnostic>.py`) — validate the CLI script itself: argument parsing, configuration loading, and that the right diagnostic/plot classes are invoked for each branch of the config. These are fast (seconds) because every external dependency is mocked.

Keep the two layers separate: CLI tests should **not** re-test diagnostic logic, and diagnostic-logic tests should **not** go through the CLI.

#### Adding tests for a new CLI

The CLI test framework relies on one small refactor and a shared set of fixtures.

1. **Expose a callable entry point.** In `aqua/diagnostics/<name>/cli_<name>.py`, wrap the body of `if __name__ == "__main__":` in a `main(argv=None)` function, leaving only a two-line stub at the bottom:

   ```python
   def main(argv=None):
       args = parse_arguments(argv if argv is not None else sys.argv[1:])
       cli = DiagnosticCLI(args, ...).prepare()
       # ... orchestration ...
       cli.close_dask_cluster()


   if __name__ == "__main__":
       main()
   ```

   This is a mechanical, behaviour-preserving change that makes the CLI importable and testable.

2. **Reuse the shared fixtures** in `tests/cli/conftest.py`:
   - `build_config(diagnostics, **kwargs)` — writes a minimal valid YAML config to a temp dir and returns its path. Pass a mapping of diagnostic blocks, e.g. `{"biases": {"run": True, ...}}` or `{"seaice_timeseries": {...}, "seaice_2d_bias": {...}}`.
   - `mock_cluster` — no-ops `open_cluster`/`close_cluster` so tests don't spawn a Dask cluster.

3. **Patch diagnostic classes at the CLI module path**, not where they are defined. In a per-file fixture, e.g.:

   ```python
   @pytest.fixture
   def mock_gb(self, mocker):
       mock_gb_cls = mocker.patch(f"{CLI_MODULE}.Climatology")
       mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotBias")
       return mock_gb_cls, mock_plot_cls
   ```

4. **Use this minimal template** for each CLI:
   - `test_parse_arguments_cli_options`: verify that parsed arguments include the expected values and any diagnostic-specific flags (e.g. seaice's `--proj` flag in seaice).
   - `test_<diagnostic>_disabled_skips_processing`: when `run: False`, verify that the plot class is not instantiated.
   - `test_<feature>_full_pipeline`: cover each main CLI execution branch (one test for each distinct orchestration block or plot type).

Do not test things already covered by `tests/diagnostic_base/` (argument merging, dataset overrides). The shared `DiagnosticCLI` is tested there.

Look at examples such as: `tests/cli/test_cli_biases.py` and `tests/cli/test_cli_seaice.py` as reference implementations.

### Coding style checks with ruff and pre-commit in a Pull Request

This project uses pre-commit hooks and ruff as linter and formatter to enforce the coding style.
The coding style is defined by the `pyproject.toml` file and the `pre-commit` configuration file in `.pre-commit-config.yaml`.

The pre-commit and ruff dependencies are installed automatically when setting up
the dev environment through the `environment-dev.yml` file.
The pre-commit configuration enforces two groups of checks:

- **General file cleaning hooks** from `pre-commit-hooks`: large-file check, trailing whitespace trimming (except `.md` and `Makefile`), end-of-file newline fix, YAML validation, and overall Python syntax validation.
- **Ruff hooks**: `ruff-check` and `ruff-format`.

Ruff has two complementary roles in this workflow:

- `ruff check` runs linting rules and reports code-quality/style violations; with `--fix` (and optionally `--unsafe-fixes`) it can also auto-fix part of them.
- `ruff format` only applies formatting (layout/style normalization), similar to a code formatter, without running lint-rule diagnostics.

To check and align code changes introduced in a PR with the coding style, developers can choose between the following options:

1. [Recommended option] Use `pre-commit` hooks to check and align code changes at commit time.


This requires installing the `pre-commit` hooks first. From the root folder of the repository, run:

```bash
pre-commit install
```

From this moment on, every time a commit is made, the `pre-commit` hooks will be run automatically to check and align
the code changes with the coding style. If the code changes are not aligned, the commit will be rejected, and the
proposed changes will appear as unstaged changes in the developer’s local repository.
The developer can then review the changes, stage them again, and commit successfully.

If, for any reason, the developer wants to disable the `pre-commit` hooks, they can run:
```bash
pre-commit uninstall
```

2. Alternatively, developers can manually run the `pre-commit` hooks to check the coding style of the code changes.
To trigger all the pre-commit hooks manually, from the root folder of the repository, run:

```bash
pre-commit run -a
```

If the developer wants to run the pre-commit hooks only for specific file(s) or folder(s), they can run:
```bash
pre-commit run --files <file_or_folder_to_target>
```

Side note:
During the manual run of the pre-commit hooks, some errors can be fixed automatically
by ruff. However, in some cases, ruff may require the extra flag `--unsafe-fixes`.
This flag allows ruff to apply fixes that might change the behavior of your code, even when it is not safe to do so.  
Use it with caution and review the diff!
To run manually ruff linter with the `--unsafe-fixes` flag:
```bash
ruff check --fix <file_or_folder_to_target> --no-cache --unsafe-fixes
```
(to run over all files, set "." as the target, from the root folder of the repository)

Then, to run the formatter manually:

```bash
ruff format <file_or_folder_to_target> --no-cache
```

This manual run will also format the code according to the formatting rules defined by the `pyproject.toml` file.
