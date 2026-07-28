.. _stateoftheart_diagnostics:

State of the art diagnostics
============================

AQUA provides a collection of built-in diagnostics to analyze climate model outputs.
The family of diagnostics named **state-of-the-art** lists diagnostics which can be used for the simulation monitoring and
make use of low resolution data as input (1 degree in both latitude and longitude and monthly frequency).
Most of these diagnostics can be compared with observations to produce metrics of evaluation and aim at providing an assessment
of the model against observational datasets and, in some selected occasions, pre-existing climate simulations.

List of diagnostics
+++++++++++++++++++

This list includes such diagnostics whose goal is to monitor and diagnose possible model drifts, imbalances and biases.

Currently implemented diagnostics are:

.. toctree::
   :maxdepth: 1

   boxplots
   global_biases
   ecmean
   timeseries
   ocean_stratification
   ocean_trends
   ocean_drift
   seaice
   teleconnections
   lat_lon_profiles
   histogram

.. _diagnostics-configuration-files:

Diagnostics configuration files
+++++++++++++++++++++++++++++++

Each diagnostic has a corresponding YAML configuration file that specifies the options and parameters for the diagnostic.
These configuration files are located in the ``config/diagnostics/<diagnostic-name>`` directory of the AQUA package and copied
to the ``AQUA_CONFIG`` folder during installation (by default ``$HOME/.aqua/``).

Each diagnostic has its own configuration file, with a block devoted to the individual diagnostic settings.
However, general settings common to all the diagnostics have a common structure here described.
Please refer to the individual diagnostic documentation for the specific settings.
See :ref:`configuration-file-guidelines` for an example of diagnostic specific block.

* ``datasets``: a list of models to analyse (defined by the catalog, model, exp, source arguments).
  If the diagnostic can handle multiple datasets, all the models in the list will be processed, otherwise only the first one will be used.
  For simplicity the default in the repository should refer to only one model.

.. code-block:: yaml

    datasets:
      - catalog: climatedt-phase1
        model: IFS-NEMO
        exp: historical-1990
        source: lra-r100-monthly
        regrid: null
        reader_kwargs: null # it can be a dictionary with reader kwargs

      - catalog: climatedt-phase1
        model: ICON
        exp: historical-1990
        source: lra-r100-monthly
        regrid: null
        reader_kwargs: null # it can be a dictionary with reader kwargs

* ``references``: a list of reference datasets to use for the analysis.
  Some diagnostics may not work with multiple references, it is better to specify it in the documentation and in the configuration file.

.. code-block:: yaml

    references:
      - catalog: obs
        model: ERA5
        exp: era5
        source: monthly
        regrid: null
        reader_kwargs: null # it can be a dictionary with reader kwargs

* ``output``: a block describing the details of the output. It contains:

    * ``outputdir``: the output directory for the plots.
    * ``rebuild``: a boolean that enables the rebuilding of the plots.
    * ``save_format``: a list (or single string) that selects the image formats to save plots. Default is SAVE_FORMAT.
    * ``dpi``: the resolution of the plots.
    * ``create_catalog_entry``: a boolean that enables the creation of a catalog entry.

.. code-block:: yaml

    output:
      outputdir: "/path/to/output"
      rebuild: true
      save_format: ['png', 'svg'] # default is SAVE_FORMAT (['png', 'pdf', 'svg'])
      dpi: 300
      create_catalog_entry: true

.. note::

  Not all the diagnostics support yet the ``create_catalog_entry`` keyword.

.. _diagnostics-cli-arguments:

Diagnostics CLI arguments
+++++++++++++++++++++++++

The following command line arguments are available for all the diagnostics:

- ``--config``, ``-c``: Path to the configuration file.
- ``--nworkers``, ``-n``: Number of workers to use for parallel processing.
- ``--cluster``: Cluster to use for parallel processing. By default a local cluster is used.
- ``--loglevel``, ``-l``: Logging level. Default is ``WARNING``.
- ``--catalog``: Catalog to use for the analysis. It can be defined in the config file.
- ``--model``: Model to analyse. It can be defined in the config file.
- ``--exp``: Experiment to analyse. It can be defined in the config file.
- ``--source``: Source to analyse. It can be defined in the config file.
- ``--outputdir``: Output directory for the plots.

If a diagnostic has extra arguments, these will be described in the individual diagnostic documentation.

Running the monitoring diagnostics
++++++++++++++++++++++++++++++++++

Each state-of-the-art diagnostic is implemented as a Python class and can be run independently.
All the diagnostic have a command line interface that can be used to run them.
A YAML configuration file is provided to set the options for the diagnostics.

Together with the individual diagnostics command line interfaces, AQUA provides an entry point to run all the diagnostics
in a single command, with a shared Dask cluster, shared output directory and with parallelization.
The entry point is called `aqua analysis` and all the details can be found in :ref:`aqua_analysis`.

.. warning::
   The analysis has to be performed preferrably on Low Resolution Archive (LRA) data, meaning
   that data should be aggregated to a resolution of 1 degree in both latitude and longitude and
   to a monthly frequency.
   It is available the option to regrid the data on the fly, but the memory usage may be highly
   increased and it may be preferrable to run the diagnostics individually.

Minimum Data Requirements
-------------------------

In order to obtain meaningful results, the diagnostics require a minimum amount of data.
Here you can find the minimum requirements for each diagnostic.

.. list-table::
   :header-rows: 1

   * - Diagnostic
     - Minimum Data Required
   * - Global Biases
     - 12 months (1 year)
   * - ecmean
     - 12 months (1 year)
   * - Timeseries
     - 2 months
   * - Seasonal Cycles
     - 2 months
   * - Gregory Plot
     - 2 months
   * - Lat-Lon Profiles
     - 12 months (1 year)
   * - Histogram
     - 12 months (1 year)
   * - Ocean Stratification
     - 12 months (1 year)
   * - Ocean Trends
     - 12 months (1 year)
   * - Ocean Drift (Hovmoller)
     - 2 months
   * - Radiation
     - 2 months
   * - Seaice
     - 1 month (12 months when computing seasonal cycle)
   * - Teleconnections
     - 24 months (2 years)

.. note::
   All diagnostics enforce the minimum data requirement at retrieval time via a ``NotEnoughDataError``.
   If the available data falls below the threshold, the diagnostic will not run and the error will be
   reported in the log. Some diagnostics (e.g. Seasonal Cycles) may produce less meaningful results
   at their minimum threshold — the value reflects the technical lower bound, not the recommended input size.

.. note::
  If you are a developer you can enforce the minimum data requirements by using the ``months_required`` argument in the ``retrieve`` and ``_retrieve`` methods
  available in the diagnostic core. The conventional way is to define a class-level constant ``MINIMUM_MONTHS_REQUIRED``
  and pass it to the ``retrieve`` call.
