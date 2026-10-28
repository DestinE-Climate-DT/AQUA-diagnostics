Ocean Heat Budget
=================

Description
-----------

The Ocean Heat Budget tool checks the closure of the heat budget of the global ocean. It compares the global
ocean mean of the net surface heat flux with the time derivative of the ocean heat content of a layer
(by default the upper 700 m).
If the surface fluxes were the only heat source of the layer the two curves would coincide: their difference
shows how much heat is exchanged through the layer boundaries (advection, mixing) or lost to drift and numerical issues,
and its mean over the period is a measure of the budget closure.

The net surface heat flux is the sum of the surface net latent, sensible, long-wave and short-wave heat fluxes
(positive downward, i.e. into the ocean) minus the latent heat of fusion of the snowfall, which the ocean has to melt
(:math:`L_f = 3.34 \times 10^5` J kg\ :sup:`-1`). Land points are masked with the missing values of the sea surface
temperature, so that fluxes and heat content are both averaged over the ocean.
The heat content tendency is the forward difference between consecutive time steps divided by their distance in seconds:
the last time step of the period has no tendency and is dropped.

Classes
-------

* ``OceanHeatBudget``: retrieves the data of one dataset and computes the two budget terms;
* ``PlotOceanHeatBudget``: plots the two terms and their difference with the aqua-core ``plot_timeseries``.

File structure
--------------

* Source code and CLI: ``aqua/diagnostics/ocean_heat_budget/``
* Template configuration file: ``aqua/diagnostics/templates/collections/config-ocean_heat_budget.yaml``
* Notebook: ``notebooks/diagnostics/ocean_heat_budget/``

Input variables and datasets
----------------------------

All variables must be in the same source, as in the LRA datasets:

* ``slhtf``, ``snlwrf``, ``snswrf``, ``ishf``: surface net latent heat, long-wave, short-wave and sensible heat fluxes;
* ``tsrwe``: total snowfall rate water equivalent;
* ``tos``: sea surface temperature, used only to define the ocean;
* ``hc700m`` (or ``hc300m``, ``hcbtm``): ocean heat content.

Basic usage
-----------

.. code-block:: python

    from aqua.diagnostics import OceanHeatBudget, PlotOceanHeatBudget

    diag = OceanHeatBudget(catalog='climatedt-gen2', model='IFS-NEMO-5km', exp='baseline-hist',
                           source='lra-r100-monthly-zarr')
    diag.run(hc_var='hc700m', outputdir='.')

    PlotOceanHeatBudget(data=diag.result, outputdir='.').plot()

To plot again from the saved NetCDF file, call ``diag.load(hc_var='hc700m', outputdir='.')`` instead of ``run()``.

CLI usage
---------

.. code-block:: bash

    python -m aqua.diagnostics.ocean_heat_budget.cli_ocean_heat_budget --config <path_to_config_file>

For the complete list of options, see :ref:`diagnostics-cli-arguments`.

Configuration file structure
----------------------------

.. code-block:: yaml

    diagnostics:
      ocean_heat_budget:
        run: true
        diagnostic_name: 'ocean_heat_budget'
        variables: ['hc700m']   # one budget for each heat content variable

Output
------

* A NetCDF file with the monthly ``net_surface_flux`` and ``hc_tendency`` (W m\ :sup:`-2`);
* two figures: the two terms (``budget_timeseries``) and their difference with its mean (``budget_difference``).

Available demo notebooks
------------------------

* `ocean_heat_budget.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/blob/main/notebooks/diagnostics/ocean_heat_budget/ocean_heat_budget.ipynb>`_
