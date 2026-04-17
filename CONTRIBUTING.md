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
Please use the `bug` label for bug issues and the `fix` label for Pull Requests with a fix.

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

Adding the `run tests` label to the Pull Request will activate the CI tests at the next push.
Adding the `ready to merge` label to the Pull Request will indicate that it is ready to be reviewed and hopefully merged in the opinion of the author.

Before asking for a review, please make sure to:
- be up to date with the `main` branch
- run the tests successfully
- if a new dependency has been added to the framework, please make sure to update the relevant environment and packaging files (`environment.yml`, `environment-dev.yml`, and/or `pyproject.toml`)
- if the environment has been modified, please make sure to update `environment_lumi.yml` and `pip_lumi.txt` accordingly.
- if a new feature has been added, please make sure to update the documentation accordingly
- add docstrings to your code

Do not merge your Pull Request yourself, it will be merged by the AQUA-diagnostics team.

### Cross-check tests

Regular CI runs **aqua-diagnostics** against the local branch, building the Micromamba environment which resolves aqua-core from PyPI. **Cross-check** uses `environment-dev.yml` and runs the same diagnostics test suite, but installs **aqua-core** from the [AQUA](https://github.com/DestinE-Climate-DT/AQUA) GitHub repository at a chosen reference (branch, tag, or commit SHA) instead of relying on the PyPI release. That helps to catch integration breaks between diagnostics and the current core codebase early.

**When to use cross-check**

- Your change depends on aqua-core behaviour or APIs that are not yet published to PyPI.
- You want to confirm compatibility against `main` or another aqua-core branch before merge.

**How to run cross-check on a pull request**

1. Add the `cross-check` label to the PR. After the `cross-check` label is added, the workflow runs automatically the cross-check tests when new commits are pushed.

**Manual run (maintainers)**

In the GitHub UI: **Actions** → **AQUA-diagnostics Cross-Check** → **Run workflow**. You can set **aqua-core branch/tag/commit**; it defaults to `main`.

### Suggesting enhancements

Enhancements of existing features or new features may be suggested by opening an issue in the AQUA-diagnostics repository.
Please use the `improvements` label for existing features.

### Writing tests

Tests live under `tests/` and use `pytest` (+ `pytest-mock` for the `mocker` fixture). Run the full suite from the repo root with:

```bash
pytest tests/
```

Two kinds of tests cover a diagnostic:

1. **Diagnostic-logic tests** (`tests/<diagnostic>/test_<diagnostic>.py`) — exercise the diagnostic classes (`GlobalBiases`, `SeaIce`, …) against the real `ci` catalog. These validate the *scientific* behaviour and data handling.
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
   - `build_config(diagnostics, **kwargs)` — writes a minimal valid YAML config to a temp dir and returns its path. Pass a mapping of diagnostic blocks, e.g. `{"globalbiases": {"run": True, ...}}` or `{"seaice_timeseries": {...}, "seaice_2d_bias": {...}}`.
   - `mock_cluster` — no-ops `open_cluster`/`close_cluster` so tests don't spawn a Dask cluster.

3. **Patch diagnostic classes at the CLI module path**, not where they are defined. In a per-file fixture, e.g.:

   ```python
   @pytest.fixture
   def mock_gb(self, mocker):
       mock_gb_cls   = mocker.patch(f"{CLI_MODULE}.GlobalBiases")
       mock_plot_cls = mocker.patch(f"{CLI_MODULE}.PlotGlobalBiases")
       return mock_gb_cls, mock_plot_cls
   ```

4. **Aim for this minimal template** per CLI (3–6 tests total):
   - `test_parse_arguments_cli_options` — smoke-check the namespace and any diagnostic-specific flags (e.g. seaice's `--proj`).
   - `test_<diagnostic>_disabled_skips_processing` — with `run: False`, assert the plot class is never instantiated.
   - `test_<feature>_full_pipeline` — the path for each major branch of the CLI (one per distinct orchestration block or plot type).
   - Optional: one test per distinct error path or non-trivial config branch (e.g. `NoDataError` handling) only if the behaviour is CLI-specific.

   Do **not** test things already covered by `tests/diagnostic_base/` (argument merging, dataset overrides) — the shared `DiagnosticCLI` is tested there.

Look at `tests/cli/test_cli_global_biases.py` and `tests/cli/test_cli_seaice.py` as reference implementations.

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
