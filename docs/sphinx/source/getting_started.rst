.. _getting_started:

Getting Started
===============

You can follow this guide to produce and see your first diagnostic plot.
Data access, catalogs and the ``aqua`` console are provided by AQUA-core and are documented in the
`AQUA-core documentation <https://aqua.readthedocs.io/en/latest/>`_: here we only cover
what you need to know to run the diagnostics.

Before you start
----------------

1. **Install AQUA-diagnostics**, following the :ref:`installation` guide (which also installs AQUA-core).

2. **Set up AQUA** (once per machine). The ``aqua`` console copies the configuration files in ``$HOME/.aqua``
   and adds the catalogs describing where the data are:

   .. code-block:: bash

       aqua install <machine>
       aqua add <catalog>

   For example ``aqua install lumi`` and ``aqua add climatedt-gen2``.
   The details are described in the AQUA-core
   `Getting Started <https://aqua.readthedocs.io/en/latest/getting_started.html>`_
   and `console <https://aqua.readthedocs.io/en/latest/aqua_console.html>`_ pages.

   When AQUA-diagnostics is installed in the same environment, ``aqua install`` copies in ``$HOME/.aqua``
   also the files used by the diagnostics:

   - ``templates/collections``: one template configuration file for each diagnostic (the starting point to write your own);
   - ``collections`` and ``analysis``: the configuration files used by ``aqua analysis`` (see :ref:`first-diagnostic-analysis`);
   - ``definitions`` and ``tools``: shared definitions, such as the regions, and diagnostic-specific settings.

3. **Find your data.** Diagnostics read data through the AQUA ``Reader``, so you need the *catalog*,
   *model*, *experiment* and *source* of your dataset. You can list what is available with:

   .. code-block:: python

       from aqua import show_catalog_content
       show_catalog_content(model="IFS-NEMO-5km")

   The state-of-the-art diagnostics are designed for the low-resolution archive (``lra-r100-monthly`` source,
   see :ref:`stateoftheart_diagnostics`).

Three ways to run a diagnostic
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - How
     - When to use it
     - Described in
   * - Notebooks and Python classes
     - Exploring a diagnostic interactively: the best way to start
     - :ref:`first-diagnostic-python`
   * - Command line of a diagnostic
     - A reproducible run, driven by a YAML configuration file
     - :ref:`first-diagnostic-cli`
   * - ``aqua analysis``
     - Advanced: many diagnostics at once on one experiment, mostly for production runs
     - :ref:`first-diagnostic-analysis`

.. _first-diagnostic-python:

In a notebook
-------------

The easiest way to start is to run one of the example notebooks, available for each diagnostic in the
`notebooks/diagnostics <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/tree/main/notebooks/diagnostics>`_
folder of the repository. To use them in Jupyter, register the kernel of your environment as described in the AQUA-core
`Getting Started <https://aqua.readthedocs.io/en/latest/getting_started.html#set-up-jupyter-kernel>`_.

Every diagnostic is made of a class that computes the results (e.g. ``Timeseries``) and one that plots them
(``PlotTimeseries``). The example computes the global mean 2 metre temperature of an experiment and plots it:

.. code-block:: python

    from aqua.diagnostics import PlotTimeseries, Timeseries

    ts = Timeseries(catalog="climatedt-gen2", model="IFS-NEMO-5km", exp="baseline-hist",
                    source="lra-r100-monthly", startdate="1990-01-01", enddate="1991-12-31")
    ts.run(var="2t", units="degC", outputdir="./output")  # computes and saves the netCDF files

    plot = PlotTimeseries(monthly_data=ts.monthly)
    fig, _ = plot.plot_timeseries(title=plot.set_title())
    plot.save_plot(fig, description=plot.set_description(), outputdir="./output")

The figure is saved in ``./output/png`` and the computed time series in ``./output/netcdf``.
The other diagnostics work in the same way.

.. _first-diagnostic-cli:

Your first diagnostic from the command line
-------------------------------------------

Each diagnostic has a command line interface driven by a YAML configuration file.
Start from the template installed in ``$HOME/.aqua``, copy it and edit the variables and options you need:

.. code-block:: bash

    cp $HOME/.aqua/templates/collections/config-timeseries.yaml my-timeseries.yaml

Then run the diagnostic, selecting the dataset from the command line:

.. code-block:: bash

    python -m aqua.diagnostics.timeseries.cli_timeseries --config my-timeseries.yaml \
        --catalog climatedt-gen2 --model IFS-NEMO-5km --exp baseline-hist --source lra-r100-monthly \
        --startdate 1990-01-01 --enddate 1991-12-31 --outputdir ./output

The same pattern, ``python -m aqua.diagnostics.<diagnostic>.cli_<diagnostic>``, applies to all the diagnostics.
The options shared by all the command line interfaces are listed in :ref:`diagnostics-cli-arguments`,
while the structure of the configuration file is described in :ref:`diagnostics-configuration-files`
and in the page of each diagnostic.

The output directory contains one folder for each type of file (``netcdf`` and the image formats, by default ``png``, ``pdf`` and ``svg``).
File names are built automatically from the diagnostic, the dataset and the variable, see :doc:`new_diagnostics/guidelines/output_management`.

.. tip::

    To verify that a dataset can be read before starting a long run, use the setup checker described in :ref:`cli`.

.. _first-diagnostic-analysis:

Advanced: running a complete analysis
-------------------------------------

The ``aqua analysis`` command runs the whole set of state-of-the-art diagnostics on an experiment, in parallel and
with a shared output folder. It is meant for monitoring and production runs on HPC systems (in a batch job):

.. code-block:: bash

    aqua analysis --catalog climatedt-gen2 --model IFS-NEMO-5km --exp baseline-hist \
        --source lra-r100-monthly --realization r1 --outputdir /path/to/output

Options and configuration are described in the AQUA-core `documentation <https://aqua.readthedocs.io/en/latest/aqua-analysis.html>`_.

Where to go next
----------------

- The diagnostics: :ref:`stateoftheart_diagnostics`, :ref:`frontier_diagnostics` and :ref:`ensemble`.
- Running in a container: :ref:`container`.
- Something not working? See :ref:`faq`.
- Writing your own diagnostic: :ref:`new_diagnostics`.
- How AQUA reads, regrids and fixes the data: the AQUA-core `Reader <https://aqua.readthedocs.io/en/latest/reader.html>`_ documentation.
