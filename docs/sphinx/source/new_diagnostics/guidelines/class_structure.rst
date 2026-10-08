.. _diagnostic-class_structure:

Core Diagnostic: ``Diagnostic``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Diagnostic`` class serves as the foundation for all diagnostics.

It provides essential functionalities such as:

- A unified ``__init__`` method for consistent initialization. Extra argument for the ``__init__`` should be added only if strictly necessary.
- Initialization of the ``Reader`` class for data access as ``self.reader`` attribute.
- A standardized data retrieval method: ``retrieve()``. This method stores the retrieved data in the ``self.data`` attribute.
  When ``std_startdate`` and ``std_enddate`` are provided, it also populates ``self.std_data`` with the corresponding slice.
  Both windows are covered by a single ``Reader`` call, and requested dates outside the catalog's effective bounds are
  automatically clipped (with a warning). It also populates the ``self.catalog`` and ``self.realization`` attributes if empty by deducing them.
- Built-in saving function ``save_netcdf()`` for NetCDF output. This includes the possibility to generate a catalog entry to be used in further analyses.
- A complementary reading function ``load_netcdf()``, which rebuilds the same filename as ``save_netcdf()`` and opens the file
  if it exists, returning ``None`` otherwise. It does not require a ``retrieve()``, deducing ``self.catalog`` from the
  model/exp/source triplet when needed, so that a run can produce the plots from the NetCDF files of a previous run.

Diagnostic Classes
^^^^^^^^^^^^^^^^^^

Each specific diagnostic inherits from ``Diagnostic`` and extends its capabilities.

This is done with the class inheritance structure, which allows for the creation of new diagnostics with minimal code duplication.

.. code-block:: python

    class MyDiagnostic(Diagnostic):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Additional initialization code

        def run(self):
            # Diagnostic-specific evaluation code

The purpose of this first class is to perform the data retrieval and the evaluations necessary on a single model.
At the end of the class execution, the results should be saved using the `save_netcdf()` method.
Additional metadata needed for plot or documentation of the results should be added to the xarray attributes,
using ``AQUA_`` as prefix of the metadata. Some metadata are added automatically when using the ``Reader``, such as:

- ``AQUA_catalog``
- ``AQUA_model``
- ``AQUA_exp``
- ``AQUA_source``
- ``AQUA_region`` when a region is selected
- ``AQUA_startdate`` and ``AQUA_enddate``
- ``AQUA_std_startdate`` and ``AQUA_std_enddate`` when the std window is set

If multiple models (e.g. model and observational dataset) are needed, two different instances of the diagnostic should be created.

Each diagnostic class must:

- Implement an ``__init__`` method that includes diagnostic-specific parameters.
- Use the ``retrieve()`` from ``Diagnostic`` for acquiring necessary data.
- If an operation is implemented in the ``Reader`` class, that method should be used (``self.reader.method()``).
- Implement a ``run()`` method or a clear order of methods to be called for the diagnostic evaluation.
- Specific substep should be called ``evaluate_<substep>()``.
- The computed results should be stored as class attributes.
- Implement a ``save_netcdf()`` method to save the results in NetCDF format, if an expansion of the ``Diagnostic.save_netcdf()`` method is needed.
- Implement a ``load()`` method that populates the same result attributes from the NetCDF files, mirroring ``save_netcdf()``.
  The filename keys must be built by a single helper shared with ``save_netcdf()``, so that the two can never address different files.
  Results with no file on disk should be left untouched, so that ``load()`` can be called both before and after ``run()``.

Comparison and Plot Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each diagnostic module should also include a dedicated class for eventually comparing results between different models and plot the final figure.

.. code-block:: python

    class MyDiagnosticPlot():
        def __init__(self, *args, **kwargs):

In this case, it may not fit the usage of the ``Diagnostic`` class, as it does not support multiple models.
It should provide methods for dataset comparison and plotting.
It should as much as possible rely on the available AQUA plotting functions.
Details about the plot should be deduced from the xarray attributes, if available.

Command-Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A CLI is available to streamline the execution of diagnostics and comparisons.
It should have a minimal mandatory set of arguments and be able to parse additional arguments if necessary (See :ref:`diagnostics-cli-arguments`).

.. _configuration-file-guidelines:

Configuration file
^^^^^^^^^^^^^^^^^^

When developing a new diagnostic, the configuration file is a mandatory component needed to expose the settings and parameters that the diagnostic requires and which can be modified by the user.
In order to ensure consistency and ease of use, some guidelines for the structure of the configuration files are provided.
The generic blocks, which should be consistent among diagnostics, are described in :ref:`diagnostics-configuration-files`,
while an example of the specific block for a diagnostic is shown below.

.. code-block:: YAML

    diagnostics:
        diagnostic_name:
            run: true # mandatory, if false the diagnostic will not run
            diagnostic_name: diagnostic_name # mandatory, may override the diagnostic name
            variables: ['variable1', 'variable2'] # example for diagnostics running on multiple variables
            regions: ['region1', 'region2'] # example for diagnostics running on multiple regions
            parameter1: default_value1
            plot_params: # example for diagnostics with specific plot parameters
                param1: value1
                param2: value2
            # Other diagnostic specific parameters here

The block may vary depending on the diagnostic, but it should always include the ``run`` parameter
to indicate whether the diagnostic should be executed or not. This allows users to enable or disable
specific diagnostics without modifying the code.

The ``diagnostic_name`` is present to override the diagnostic name if needed.
Imagine for example to run the timeseries diagnostic in an analysis about precipitation.
This will allow the files to be named ``precipitation.timeseries.png`` instead of ``timeseries.timeseries.png``,
which would be less informative.

.. _aqua-console:

Configuration Files and AQUA console
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration files shipped with AQUA-diagnostics are copied in the AQUA configuration folder
(by default ``$HOME/.aqua``) by the ``aqua install`` command of the AQUA console (see :ref:`getting_started`).
The console finds them through the ``aqua.plugins`` entry point declared in the ``pyproject.toml`` of the package,
which points to the ``get_install_dirs()`` function of ``aqua.diagnostics``.
This function returns the folders to be installed (``DIAGNOSTIC_CONFIG_DIRECTORIES`` and
``DIAGNOSTIC_TEMPLATE_DIRECTORIES`` in ``aqua/diagnostics/__init__.py``).
The mechanism is described in the AQUA-core
`documentation <https://aqua.readthedocs.io/en/latest/aqua_console.html#how-aqua-discovers-installable-components>`_.

To expose the configuration files of a new diagnostic it is therefore enough to add them to the existing folders of the
package, without any change to the code:

- ``aqua/diagnostics/templates/collections/config-<diagnostic>.yaml``: the template configuration file of the diagnostic;
- ``aqua/diagnostics/config/collections``: the configuration files used by ``aqua analysis``
  (jinja templates in ``jinja`` and legacy YAML files in ``legacy``);
- ``aqua/diagnostics/config/analysis/config.aqua-analysis.yaml``: to run the diagnostic with ``aqua analysis``,
  add its command line interface to the ``cli:`` block (e.g. ``biases: "biases/cli_biases.py"``);
- ``aqua/diagnostics/config/tools/<diagnostic>``: optional, settings and lookup files specific to the diagnostic only;
- ``aqua/diagnostics/config/definitions``: lookup files shared by all the diagnostics.

After the installation the folder structure is:

.. code-block:: text

    $HOME/.aqua/
        ├── analysis/
        ├── collections/
        ├── definitions/
        │   └── regions.yaml
        ├── templates/
        │   └── collections/
        │       └── config-<diagnostic>.yaml
        └── tools/
            └── <diagnostic>/

The ``definitions/`` folder contains shared lookup files used by all diagnostics.
``regions.yaml`` is the centralized registry of all available geographic regions:
any diagnostic can refer to a region by name (e.g. ``nh``, ``arctic``, ``io``) and the
``Diagnostic`` base class resolves it against this single file.

The ``tools/`` folder contains a subfolder only for the diagnostics that need settings or lookup files
*specific to that diagnostic* (e.g. custom index definitions for teleconnections).

.. note::
    If AQUA-diagnostics is installed in editable mode, with ``aqua install <machine> --diagnostics <path/to/AQUA-diagnostics>``,
    the folders are linked and new files are available immediately. Otherwise run ``aqua update`` to copy them.
