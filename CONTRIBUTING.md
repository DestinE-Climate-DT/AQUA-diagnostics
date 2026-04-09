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

### Suggesting enhancements

Enhancements of existing features or new features may be suggested by opening an issue in the AQUA-diagnostics repository. Please use the `improvements` label for existing features.

### Coding style

To enforce the coding style, we leverage on `pre-commit` hooks and `ruff` as a linter and formatter.
We set the length limit of 127 characters per line. More information about the coding style chosen
can be found in the `pyproject.toml` file.

### Manual trigger of lint and formatting checks for specific files in a Pull Request

To manually trigger and align the code changes to the reference coding style, do the following steps:

1. Install the pre-commit hooks (if not already installed):
```bash
pre-commit install
```

2. If does not exist yet, create a file `.pre-commit-config.yaml` in the root of the repository with the following content:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
```

3. From AQUA-diagnostics root folder, manually run the pre-commit hooks:
```bash
pre-commit run --all-files
```

4. If not installed already, install `ruff` (check the version in the `.pre-commit-config.yaml` file):
```bash
pip install ruff==<version_number>
```

5. Run the linter fixer from the path to the file or folder that have been modified in the PR and need to be aligned to the reference coding style (it will use the rules set in the `pyproject.toml` file):

```bash
ruff check --fix <file_or_folder_to_target> --no-cache
```

Note:
The extra flag `--unsafe-fix` allows Ruff to apply fixes that might change the behavior of your code, even if it is not safe to do so.  
Use it with caution and review the diff!

6. Run the formatter from AQUA-diagnostics root folder:
```bash
ruff format <file_or_folder_to_target> --no-cache
```

This will format the code according to the formatting rules set in the `pyproject.toml` file.
