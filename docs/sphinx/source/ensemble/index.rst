.. _ensemble:

Ensemble diagnostics
--------------------

In many areas of climate research, ensemble analysis is a standard approach to assess the robustness of model results and to quantify the uncertainty linked to numerical simulations.
By combining multiple simulations, whether from different models, parameter choices, or perturbed initial conditions, ensemble methods provide more reliable estimates of the behaviour of the system being studied.

The **Ensemble** diagnostics offers a set of tools to compute and visualise basic ensemble statistics, such as the mean and standard deviation.
These metrics help users examine both the typical response of the ensemble and the variability across its members, giving an indication of model agreement and the confidence that can be placed in the simulated fields.

It supports ensemble analysis for 1D time series, 2D ``LatLon`` maps, and zonal sections ``LevLon``, with the option to use weighted statistics for multi-model ensembles.

This section documents the Ensemble diagnostics available in AQUA-diagnostics.

.. toctree::
   :maxdepth: 2

   ensemble_timeseries
   ensemble_latlon
   ensemble_zonal

Detailed API
------------

.. automodule:: aqua.diagnostics.ensemble
    :members:
    :undoc-members:
    :show-inheritance:

.. note::
   WORK IN PROGRESS