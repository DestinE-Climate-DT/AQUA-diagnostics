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
   biases
   ecmean
   timeseries
   ocean_stratification
   trends
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
        model: ECMWF
        exp: era5
        source: monthly
        regrid: null
        reader_kwargs: null # it can be a dictionary with reader kwargs

* ``output``: a block describing the details of the output. It contains:

    * ``outputdir``: the output directory for the plots.
    * ``rebuild``: a boolean that enables the rebuilding of the plots.
    * ``save_format``: a list (or single string) that selects the image formats to save plots. Default is SAVE_FORMAT.
    * ``save_netcdf``: a boolean that enables the writing of the NetCDF files of the diagnostic.
    * ``plot_only``: a boolean that skips the evaluation of the diagnostic, producing the plots from the NetCDF files
      already present in ``outputdir``. Default is ``false``. Supported only by some diagnostics, see :ref:`diagnostics-plot-only`.
    * ``dpi``: the resolution of the plots.
    .. * ``create_catalog_entry``: a boolean that enables the creation of a catalog entry.

.. code-block:: yaml

    output:
      outputdir: "/path/to/output"
      rebuild: true
      save_format: ['png', 'svg'] # default is SAVE_FORMAT (['png', 'pdf', 'svg'])
      save_netcdf: true
      plot_only: false
      dpi: 300

.. _diagnostics-regions:

Regions
+++++++

The diagnostics that can work on a region refer to it by its key in the centralized regions file,
``aqua/diagnostics/config/definitions/regions.yaml``, copied by ``aqua install`` in the ``definitions`` folder
of the AQUA configuration folder (by default ``$HOME/.aqua/``).
For example, ``regions: ['io', 'sss']`` in the configuration file of the ocean drift, or ``region: 'europe'`` in the one of the biases.

Regions are polygons whenever possible: the `IPCC AR6 reference regions <https://doi.org/10.5194/essd-12-2959-2020>`_
for land and atmosphere (e.g. ``europe``) and the `IHO sea areas <https://doi.org/10.14284/323>`_ for oceans and seas
(e.g. ``io``, ``ao``, ``so``, ``arc_o``, ``ls``, ``ms``).
The regions with no equivalent in either collection are lon/lat boxes.

A lon/lat box gives the limits of the region, while a polygon region names a polygon collection and the polygons to merge:

.. code-block:: yaml

    regions:
      nh:
        longname: North Midlatitudes
        lon_limits: [-180, 180]
        lat_limits: [30, 90]
      gs:
        regionmask: iho
        regionmask_names: [Greenland Sea, Norwegian Sea]
        longname: Gin Sea

* ``longname``: the name of the region in titles, logs and filenames (default: the key).
* ``lon_limits``, ``lat_limits``: the limits of a lon/lat box.
* ``regionmask``: the polygon collection of a polygon region, either one of the ``sources`` of the regions file (``ar6``, ``iho``)
  or a collection shipped with `regionmask <https://regionmask.readthedocs.io>`_ (e.g. ``srex`` or ``ar6.ocean``).
* ``regionmask_names``: the polygons of the collection merged into the region, with their names in the collection
  (default: the ``longname``).
* ``frac_threshold``: a grid cell is selected if the region covers more than this fraction of it.
  The default, ``0.0``, keeps every cell the region touches: on coarse grids (e.g. the 1° LRA) many coastal ocean cells
  have their centre on land, and they would be lost otherwise. ``null`` keeps only the cells whose centre is inside the region.
  Any value other than ``null`` needs data on a regular lon/lat grid, such as the LRA or regridded data.

The polygon collections are listed in the ``sources`` section of the regions file, with these keys:

* ``url``: the official link to the zipped shapefile.
* ``names``: the column of the shapefile with the polygon names.
* ``simplify``: the tolerance, in degrees, used to simplify the polygons (``0`` for none).

Each collection is read from ``<name>_<simplify>.gpkg`` (e.g. ``iho_0.02.gpkg``) in the ``regions`` folder next to the regions file,
where the container build stores it.
If it is not there, it is downloaded from ``url`` the first time it is used (for the IHO sea areas, about 100 MB in one or two minutes),
simplified and cached in the ``aqua`` folder of the regionmask cache (by default ``$HOME/.cache/regionmask/aqua``).

.. note::
    On machines without internet access (e.g. MareNostrum 5) the container already has the polygons.
    Otherwise, copy ``$HOME/.cache/regionmask/aqua`` from a machine where they were already used.

The diagnostics with a ``regions_file_path`` option (e.g. the biases) can use another regions file with the same structure.
A custom regions file with polygon regions needs its own ``sources`` section,
and its polygons are looked for in the ``regions`` folder next to it.

The AR6 regions (Iturbide et al., 2020) are distributed under the CC BY 4.0 licence
and the IHO sea areas (Flanders Marine Institute, 2018) under the CC BY-NC-SA 4.0 licence.

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
- ``--realization``: Realization name. Merged into the first dataset's ``reader_kwargs``.
- ``--regrid``: Target regrid resolution.
- ``--startdate``: Start date (``YYYY-MM-DD``).
- ``--enddate``: End date (``YYYY-MM-DD``).
- ``--outputdir``: Output directory for the plots.
- ``--reader_kwargs``: Additional Reader kwargs as a JSON object, merged into the first dataset's
  ``reader_kwargs`` (CLI values take precedence over the config file on matching keys). For example:

  .. code-block:: bash

     --reader_kwargs '{"engine": "polytope", "chunks": {"time": 12}}'

  Quote the entire object with single quotes in the shell and use double quotes for JSON keys and
  strings. JSON preserves numbers, booleans (``true`` and ``false``), lists, nested objects and
  ``null`` (Python ``None``). The selected Reader/backend determines which additional parameters
  it supports.

  ``catalog``, ``model``, ``exp``, ``source``, ``regrid``, ``startdate``, ``enddate``, ``loglevel``
  and ``realization`` must be set with their dedicated CLI flags: they are rejected inside
  ``--reader_kwargs`` to avoid conflicting settings. Invalid JSON or a value that is not an object
  also produces a CLI error before retrieval. Individual diagnostics may reject additional keys
  when they have their own dedicated flags for them (see their documentation).

Only the first dataset in the ``datasets`` list is affected by these CLI overrides.
If a diagnostic has extra arguments, these will be described in the individual diagnostic documentation.

.. _diagnostics-plot-only:

Plotting from previous results
++++++++++++++++++++++++++++++

Some diagnostics can produce their plots from the NetCDF files written by a previous run, without retrieving or computing anything.
From the CLI, this is enabled by ``plot_only: true`` in the ``output`` block of the configuration file.
From Python, the diagnostic class provides a ``load()`` method, which populates from the NetCDF files the same attributes as ``run()``,
so that the plot classes are used in the same way. Results with no file on disk are left untouched.
``load()`` must be given the same arguments used to produce the files, such as the variable and the ``reader_kwargs``,
because they take part in the filenames.

This is currently supported by the following diagnostics, whose pages describe the arguments their ``load()`` needs:
See the individual diagnostic documentation for usage details.

.. Keep the list in alphabetical order

* :doc:`lat_lon_profiles`
* :doc:`ocean_drift`
* :doc:`ocean_stratification`
* :doc:`ocean_trends`
* :doc:`trends`

The other diagnostics ignore ``plot_only`` and always run the full evaluation.

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
   * - Trends
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
