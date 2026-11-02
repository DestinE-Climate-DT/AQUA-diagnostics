.. _trends:

Trends Diagnostic
=================

Description
-----------

The **Trends** diagnostic computes linear trends of climate variables over a selected
analysis period. It can be used for atmospheric and ocean variables and produces one
horizontal map per variable.

Trend coefficients are obtained from a linear polynomial fit along the time coordinate.
They are expressed in the original variable units per year, where a year corresponds
to 365.25 elapsed days. The diagnostic can select geographic regions and compute
area-weighted spatial means before fitting.

For multi-level ocean maps and zonal mean vertical sections, see :doc:`ocean_trends`.

Classes
-------

There is one class for the analysis and one for the plotting:

* **Trends**: retrieves data through the AQUA Reader and computes linear trend
  coefficients. Results remain lazy until computed, plotted or saved. The ``run()``
  method retrieves, fits and saves the coefficients, and stores them in ``trend_coef``.
* **PlotTrends**: produces one map per variable using the AQUA core graphics functions.
  Plotting preserves the supplied data and saves figures through ``OutputSaver``.

File structure
--------------

* The diagnostic and its CLI script, ``cli_trends.py``, are located in
  ``aqua/diagnostics/trends``.
* A horizontal-map example is available at
  ``aqua/diagnostics/templates/collections/config-trends-2d.yaml``.
* The shared template at ``aqua/diagnostics/templates/collections/config-trends.yaml``
  contains separate generic and ocean trend settings. Its generic block saves
  multi-level coefficients with plotting disabled.
* Collection configurations are available in
  ``aqua/diagnostics/config/collections/jinja`` and
  ``aqua/diagnostics/config/collections/legacy``.

Input variables and datasets
----------------------------

The diagnostic accepts variables with a datetime time coordinate. At least 12 months
of data are required at retrieval time.

The analysis collections include:

* ``atmosphere2d``: ``2t`` (2 metre temperature) and ``msl`` (mean sea level pressure).
* ``water_cycle``: ``tprate`` (total precipitation rate).
* ``ocean2d``: ``tos`` (sea surface temperature).

Each dataset is analysed independently; a reference dataset is not required.
Monthly data from the Low Resolution Archive (LRA) can be used directly. Regridding
can be requested through the ``regrid`` argument when needed for maps.

The computation supports both surface and multi-level fields. Plotting supports
horizontal fields with latitude and longitude coordinates. Set ``plot: false`` in
the CLI configuration when coefficients retain a vertical dimension or spatial
averaging produces a non-horizontal result.

.. note::

    The ``config-trends-2d.yaml`` example uses CI FESOM test data for ``siconc`` and
    ``sithick``. These files must be available at the locations configured in the
    CI catalog. The default ``./AQUA_tests`` paths are relative to the working
    directory. For a model analysis, configure the corresponding catalog, model,
    experiment, source and variables instead.

Basic usage
-----------

The basic structure of the analysis is the following:

.. code-block:: python

    from aqua.diagnostics.trends import Trends, PlotTrends

    trend = Trends(
        catalog='climatedt-gen2',
        model='IFS-NEMO-5km',
        exp='baseline-hist',
        source='lra-r100-monthly',
        startdate='2001-01-01',
        enddate='2005-12-31'
    )
    trend.run(var=['2t', 'msl'], outputdir='./output')

    plot = PlotTrends(data=trend.trend_coef, outputdir='./output')
    plot.plot_trend(save_format=['png', 'pdf'])

To compute coefficients without saving them, use ``retrieve()`` followed by
``compute_trend()``:

.. code-block:: python

    trend.retrieve(var=['2t'])
    coefficients = trend.compute_trend(region='tropics')

    # Average over the selected region before fitting a zonal trend.
    zonal_coefficients = trend.compute_trend(region='tropics', dim_mean='lon')

The retrieved data can be reused for multiple regions. Without ``dim_mean``, the
CLI computes the pointwise trend once and selects the requested regions afterwards.
With ``dim_mean``, the area-weighted mean is computed before the fit for each region.

CLI usage
---------

The diagnostic can be run from the command line:

.. code-block:: bash

    python -m aqua.diagnostics.trends.cli_trends --config config-trends.yaml

The standard dataset, date, output and cluster arguments are described in
:ref:`diagnostics-cli-arguments`. The CLI processes only the ``diagnostics.trends``
block, even when the configuration also contains ``diagnostics.ocean_trends``.

Without ``--config``, the CLI uses the shared
``templates/collections/config-trends.yaml`` template.

Configuration file structure
----------------------------

Common dataset and output settings are described in
:ref:`diagnostics-configuration-files`. The generic trend settings are defined in
the ``trends`` block nested under ``diagnostics``:

* ``run``: enable or disable the diagnostic.
* ``diagnostic_name``: output filename prefix. Default is ``trends``.
* ``var``: variable name or list of variables to analyse.
* ``regions``: list of regions applied to all variables. Default is ``[null]``
  for the global domain.
* ``dim_mean``: dimension or list of dimensions over which to compute an
  area-weighted mean before fitting. Default is ``null`` for pointwise trends.
* ``plot``: enable maps. Default is ``true``.
* ``plot_params``: optional plotting arguments, such as ``cmap``, ``vmin``, ``vmax``
  and ``sym``, applied to all variables. By default, automatic colour limits are
  symmetric around zero. Supplying explicit limits disables this automatic
  symmetry unless ``sym`` is also set.

For example, save the following configuration as ``config-trends.yaml``:

.. code-block:: yaml

    datasets:
      - catalog: climatedt-gen2
        model: IFS-NEMO-5km
        exp: baseline-hist
        source: lra-r100-monthly
        reader_kwargs: {chunks: {time: 12}}
        startdate: '2001-01-01'
        enddate: '2005-12-31'

    output:
      outputdir: './output'
      save_netcdf: true
      save_format: ['png', 'pdf', 'svg']

    diagnostics:
      trends:
        run: true
        var: ['2t', 'msl']
        regions: [null, 'tropics']

To plot previously saved coefficients, set ``plot_only: true`` in ``output`` and
use the same output directory, dataset identity, realization, analysis dates,
regions and ``dim_mean`` settings as the original run. The CLI loads the saved
coefficients without retrieving the source data or fitting again. A missing
coefficient file raises ``NoDataError``.

Output
------

The diagnostic saves coefficients as NetCDF files under ``netcdf`` when
``save_netcdf`` is enabled. Each file contains the requested variables for one
dataset and region. Units and provenance are preserved in the output metadata.

Maps are saved separately for each variable in the selected image-format folders.
PNG, PDF and SVG are the default formats. NetCDF coefficients use the product
name ``trend``; maps use ``map_trend``. The ``diagnostic_name`` setting determines
the output prefix, including the collection name when run through
:ref:`aqua_analysis`.

Detailed API
------------

This section provides a detailed reference for the API of the ``trends`` diagnostic,
generated from the class docstrings.

.. automodule:: aqua.diagnostics.trends
    :members:
    :undoc-members:
    :show-inheritance:
