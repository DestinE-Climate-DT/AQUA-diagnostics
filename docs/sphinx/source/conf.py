# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from importlib.metadata import PackageNotFoundError, version as package_version

try:
    project_version = package_version("aqua-diagnostics")
except PackageNotFoundError:
    # Fallback for local/source builds where package metadata is unavailable.
    project_version = "unknown"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AQUA-diagnostics"
copyright = "2026, Climate DT Team"
author = "Climate DT Team"
version = str(project_version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.todo"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
autoclass_content = "both"
exclude_patterns = []

# Mock imports for modules that are not available during docs build, can be expanded
autodoc_mock_imports = ["dummy"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
