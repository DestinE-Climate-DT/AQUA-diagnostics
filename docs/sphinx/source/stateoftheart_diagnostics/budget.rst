.. _budget:

Budget
======

Description
-----------

The Budget tool is a sanity and leak check: it compares the global means of two quantities that should balance
each other, and their difference with its expected value.
Each quantity is a formula (or a single variable) of the same source. The second one can be replaced by its time
derivative, to compare a flux with the change of the quantity it feeds. Typical uses are:

* **Ocean heat budget**: the net surface heat flux into the ocean against the tendency of the ocean heat content of a
  layer (e.g. the upper 700 m). Their difference is the heat exchanged through the layer boundaries, drift and numerical leaks;
* **Atmosphere energy budget**: the net radiation at the top of the atmosphere against the net heat flux at the surface,
  with no time derivative. Their difference is the energy leak of the atmosphere.

The expected value of the difference is a configuration option: zero for a leak check, another value when an
imbalance is expected (e.g. the radiation imbalance of a warming climate).

The time derivative is the forward difference between consecutive time steps divided by their distance in seconds:
the last time step of the period has no tendency and is dropped from both quantities.
An optional mask variable excludes its missing values (at the first time step) from both means: e.g. ``tos`` to average
over the ocean only.

Classes
-------

* ``Budget``: retrieves the variables of the two formulae for one dataset and computes their global means;
* ``PlotBudget``: plots the two quantities together, then their difference with its mean and its expected value,
  with the aqua-core ``plot_timeseries``.

.. note::

    The minimum amount of data is ``MINIMUM_MONTHS_REQUIRED`` (12 months: one full annual cycle of the budget).

File structure
--------------

* Source code and CLI: ``aqua/diagnostics/budget/``
* Template configuration file: ``aqua/diagnostics/templates/collections/config-budget.yaml``
* Notebook: ``notebooks/diagnostics/budget/``

The source code directory contains the following files:

* ``budget.py``: the analysis class ``Budget``;
* ``plot_budget.py``: the plot class ``PlotBudget``;
* ``cli_budget.py``: the command line interface.

Input variables and datasets
----------------------------

Any variable of the source can enter the formulae (``+``, ``-``, ``*``, ``/``, ``^``, parentheses and numbers).
All variables of a budget must be in the same source, as in the LRA datasets. The template configuration uses:

* ``slhtf``, ``snlwrf``, ``snswrf``, ``ishf``: surface net latent heat, long-wave, short-wave and sensible heat fluxes;
* ``tsrwe``: total snowfall rate water equivalent, multiplied by the latent heat of fusion
  (:math:`L_f = 3.34 \times 10^5` J kg\ :sup:`-1`);
* ``tnlwrf``, ``tnswrf``: top net long-wave and short-wave radiation;
* ``tos``: sea surface temperature, used only as a mask;
* ``hc700m`` (or ``hc300m``, ``hcbtm``): ocean heat content.

Basic usage
-----------

.. code-block:: python

    from aqua.diagnostics import Budget, PlotBudget

    diag = Budget(catalog='climatedt-gen2', model='IFS-NEMO-5km', exp='baseline-hist',
                  source='lra-r100-monthly-zarr')
    diag.run(name='ocean_heat', first='slhtf+snlwrf+snswrf+ishf-3.34e5*tsrwe', second='hc700m',
             tendency=True, mask='tos', units='W m-2', outputdir='.')

    PlotBudget(data=diag.result, name='ocean_heat', expected=0, outputdir='.').plot()

To plot again from the saved NetCDF file, call ``diag.load(name='ocean_heat', outputdir='.')`` instead of ``run()``.

CLI usage
---------

.. code-block:: bash

    python -m aqua.diagnostics.budget.cli_budget --config <path_to_config_file>

For the complete list of options, see :ref:`diagnostics-cli-arguments`.

Configuration file structure
----------------------------

Every budget listed in ``budgets`` is configured in ``params.<name>``, on top of ``params.default``:

.. code-block:: yaml

    diagnostics:
      budget:
        run: true
        diagnostic_name: 'budget'
        budgets: ['ocean_heat', 'atmosphere_energy']
        params:
          default:
            units: 'W m-2'      # both quantities are converted to these units
            expected: 0.0       # expected value of first minus second
          ocean_heat:
            first: 'slhtf+snlwrf+snswrf+ishf-3.34e5*tsrwe'
            second: 'hc700m'
            tendency: true      # compare with the time derivative of the second quantity
            mask: 'tos'
            long_names: ['Net surface heat flux', 'Heat content tendency (0-700 m)']
          atmosphere_energy:
            first: 'tnlwrf+tnswrf'
            second: 'slhtf+snlwrf+snswrf+ishf-3.34e5*tsrwe'
            long_names: ['Net TOA radiation', 'Net surface heat flux']

Collections
-----------

The tool runs in ``aqua analysis`` within the ``climate_metrics`` collection, configured by
``aqua/diagnostics/config/collections/jinja/climate_metrics/config-climate_metrics-budget.j2``
(``legacy/climate_metrics/config-climate_metrics-budget.yaml`` for standalone runs, see :ref:`analysis_collections`).
It checks the ocean heat budget with the full-depth heat content ``hcbtm`` and the atmosphere energy budget, over the
whole period of the experiment. The figures are in the "Climate metrics, Energy budgets" section of the dashboard.

Output
------

For each budget (its name is in the filenames):

* a NetCDF file with the monthly global means ``first`` and ``second``;
* two figures: the two quantities (``budget_timeseries``) and their difference with its mean and expected value
  (``budget_difference``).

Available demo notebooks
------------------------

* `budget.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/blob/main/notebooks/diagnostics/budget/budget.ipynb>`_

Authors and contributors
------------------------

This tool is maintained by Matteo Nurisso (`@mnurisso <https://github.com/mnurisso>`_).
Contributions are welcome, please open an issue or a pull request.

Detailed API
------------

This section provides a detailed reference for the Application Programming Interface (API) of the ``budget`` tool,
generated from the function docstrings.

.. automodule:: aqua.diagnostics.budget
    :members:
    :undoc-members:
    :show-inheritance:
