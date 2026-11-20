.. _dummy:

Dummy Diagnostic
==================================================

.. Template page of the new-aqua-diagnostic skill. The sections follow histogram.rst and the structure of
.. every page in this folder, so that they all read the same way. Comments like this one are for the author:
.. delete them, replace the example text, and keep the headings and their order.

Description
-----------

.. What the tool computes, which phenomenon or process it diagnoses and why it matters for the analysis of
.. high-resolution climate simulations. Cite the published method here and in the References section.
.. Remember that a *tool* is agnostic of variable and scientific question (see the guidelines): say what it
.. computes, and leave the scientific purpose to the collections that use it (see Collections below).

The **Dummy** tool computes the annual mean of a variable averaged over a region, for one dataset, and plots it as a
time series.
It can compare the dataset with a reference dataset (for example ERA5), shown as a reference line.

Classes
-------

There is one class for the analysis and one for the plotting:

* **Dummy**: retrieves the data of one dataset, computes the regional annual mean and saves it as a NetCDF file.
  Results are stored as class attributes with their ``AQUA_*`` provenance attributes.
  The results of a previous run can be read back with ``load()``, which makes plot-only runs possible.
  A reference is a second instance of the same class.

* **PlotDummy**: builds the title and the caption from the attributes of the results, draws the figure with
  ``aqua.core.graphics`` and saves it through ``OutputSaver``. It never accesses the data.

.. note::

    The minimum amount of data is ``MINIMUM_MONTHS_REQUIRED`` (12 months: one complete annual mean).
    Add the tool to the Minimum Data Requirements table of ``stateoftheart_diagnostics/index.rst``.

File structure
--------------

* The tool is located in the ``aqua/diagnostics/dummy/`` directory, which contains the source code and the
  command line interface (CLI) script.
* A template configuration file is available at ``aqua/diagnostics/templates/collections/config-dummy.yaml``.
* Region definitions are available in ``aqua/diagnostics/config/definitions/regions.yaml``.
* Notebooks are available in the ``notebooks/diagnostics/dummy/`` directory.

The source code directory contains the following files:

* ``dummy.py``: the analysis class ``Dummy``;
* ``plot_dummy.py``: the plot class ``PlotDummy``;
* ``cli_dummy.py``: the command line interface;
* ``util.py`` (optional): helper functions specific to this tool.

Input variables and datasets
----------------------------

.. List the variables the tool works on (the configuration decides, not the code) and the grids it needs.
.. If the tool needs a reference, or cannot work without one, say it here.

The tool works on any field that can be averaged over a region, for example:

* ``2t`` (2 metre temperature)
* ``tprate`` (total precipitation rate)

One dataset is analysed at a time. A reference dataset is optional: it is analysed with the same class, over its
own period and grid, and drawn together with the dataset.

Basic usage
-----------

The basic usage of this tool is explained with a working example in the notebook.
The basic structure of the analysis is the following:

.. code-block:: python

    from aqua.diagnostics import Dummy, PlotDummy

    dummy = Dummy(
        catalog='climatedt-phase1',
        model='ICON',
        exp='historical-1990',
        source='lra-r100-monthly',
        startdate='1990-01-01',
        enddate='1999-12-31',
        region='tropics',
        loglevel='INFO'
    )
    dummy.run(var='2t', outputdir='./')

    # Optional reference: a second instance of the same class
    ref = Dummy(catalog='obs', model='ECMWF', exp='era5', source='r100-monthly', regrid='r100', region='tropics')
    ref.run(var='2t', outputdir='./')

    PlotDummy(data=dummy.result, ref_data=ref.result, outputdir='./').plot()

.. note::

    Start/end dates and reference dataset can be customized.
    If not specified otherwise, plots are saved in PNG, PDF and SVG format in the output directory.

CLI usage
---------

The tool can be run from the command line interface (CLI) with the following command:

.. code-block:: bash

    cd $AQUA/aqua/diagnostics/dummy
    python cli_dummy.py --config <path_to_config_file>

The CLI accepts the arguments common to all the tools, see :ref:`diagnostics-cli-arguments`.

Configuration file structure
----------------------------

The configuration file is a YAML file that contains the details on the dataset to analyse and on the reference,
the output directory and the tool settings.
Most of the settings are common to all the tools (see :ref:`diagnostics-configuration-files`).
Here we describe only the specific settings of the Dummy tool.

* ``dummy``: a block (nested in the ``diagnostics`` block) containing the options of the tool.

    * ``run``: enable/disable the tool.
    * ``diagnostic_name``: prefix of every output file. ``dummy`` by default, the collection name when the tool
      runs inside a collection.
    * ``variables``: list of variables to analyse.
    * ``params``: computation options, with a ``default`` entry and per-variable overrides.
      ``regions`` is a list of region names (``null`` is the global domain).

* ``references``: optional block (top level) with the reference dataset. Only the first one is used. It keeps its own
  ``regrid``, ``startdate`` and ``enddate``. Remove it to run without a reference.

.. code-block:: yaml

    references:
      - catalog: 'obs'
        model: 'ECMWF'
        exp: 'era5'
        source: 'r100-monthly'
        regrid: 'r100'

    diagnostics:
      dummy:
        run: true
        diagnostic_name: 'dummy'
        variables: ['2t']
        params:
          default:
            regions: [null]

Collections
-----------

.. A collection is what the AQUA language calls a diagnostic: tools configured for a scientific purpose.
.. List here the collections that use this tool, or delete the section if there is none yet.

A tool becomes part of a diagnostic when a **collection** configures it for a scientific purpose. The collections
are defined in ``aqua/diagnostics/config/collections/`` and run by ``aqua analysis``
(see :ref:`aqua_analysis`):

* ``jinja/<collection>/config-<collection>-<quantity>.j2``: the configurations actually run by ``aqua analysis``.
  The reference datasets and the dates are written as Jinja variables (``{{ period.standard.startdate }}``)
  and rendered for each experiment kind.
* ``legacy/<collection>/config-<collection>-<quantity>.yaml``: the same file rendered with the ``default`` kind,
  that can be used with ``--config`` outside ``aqua analysis``.

See :ref:`analysis_collections` for the kind file and the naming of the files.

To add this tool to a collection, add a ``dummy`` block to the files of that collection (same keys as above,
with ``diagnostic_name`` set to the collection name) and register the tool under the collection in
``aqua/diagnostics/config/analysis/config.aqua-analysis.yaml``.

Output
------

The tool produces the following outputs, named ``<diagnostic_name>.annual_mean.<catalog>.<model>.<exp>.<realization>...``:

* ``netcdf/``: the annual mean of each variable and region, with the ``AQUA_*`` attributes;
* ``png/``, ``pdf/``, ``svg/``: the time series figure, with the caption stored in the metadata.

With ``plot_only: true`` in the ``output`` block the figures are produced again from the NetCDF files,
without recomputing (see :ref:`diagnostics-plot-only`).

Observations
------------

.. Reference datasets used by default and other suitable ones. Delete the section if the tool uses none.

The default reference dataset is the ERA5 reanalysis, provided by ECMWF.
Other reference datasets can be configured in the ``references`` block.

.. Example Plots
.. -------------
..
.. .. figure:: figures/dummy.annual_mean.<catalog>.<model>.<exp>.<realization>.<var>.png
..    :align: center
..    :width: 100%
..
..    Annual mean of the variable over the region, with the reference dataset as a line.

Available demo notebooks
------------------------

Notebooks are stored in ``notebooks/diagnostics/dummy``:

- `dummy.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/tree/main/notebooks/diagnostics/dummy/dummy.ipynb>`_

References
----------

.. Published material the method is based on, one bullet each.

Authors and contributors
------------------------

This tool is maintained by [Name] (`@github_username <https://github.com/github_username>`_, `email@example.com <mailto:email@example.com>`_).
Contributions are welcome, please open an issue or a pull request.
For questions or suggestions, contact the AQUA team or the maintainer.

Detailed API
------------

This section provides a detailed reference for the Application Programming Interface (API) of the ``dummy`` tool,
generated from the function docstrings.

.. automodule:: aqua.diagnostics.dummy
    :members:
    :undoc-members:
    :show-inheritance:
