.. _ensemble_analysis:

Ensemble analysis
=================

Description
-----------

``aqua analysis ensemble`` computes the ensemble statistics of an experiment from the output that ``aqua analysis`` already produced for each realization.
It does not read the model data again, so the analysis of the single realizations has to be completed first.

.. code-block:: bash

    aqua analysis ensemble --catalog <catalog> --model <model> --exp <exp> --source <source> -o <outputdir>

The realizations are read from ``<outputdir>/<catalog>/<model>/<exp>/r<N>/`` and the results are written to ``<outputdir>/<catalog>/<model>/<exp>/ensemble/``.
The list of realizations is taken from the catalog.

Collections
-----------

The collections are set in ``config.aqua-analysis-ensemble.yaml``. Each of them has its own configuration file in ``config/collections/legacy``:

.. list-table::
   :header-rows: 1

   * - Collection
     - Configuration file
     - Products
   * - ``ensemble_atmosphere2d``
     - ``config-atmosphere2d-berkeley-ensemble.yaml``
     - Time series, annual and seasonal maps, seasonal cycles, latitude profiles
   * - ``ensemble_atmosphere3d``
     - ``config-atmosphere3d-ensemble.yaml``
     - Zonal means
   * - ``ensemble_ocean2d``
     - ``config-ocean2d-ensemble.yaml``
     - Latitude profiles
   * - ``ensemble_water_cycle``
     - ``config-water_cycle-ensemble.yaml``
     - Latitude profiles
   * - ``ensemble_seaice``
     - ``config-seaice-ensemble.yaml``
     - Sea ice maps

Products
--------

Each block of the ``diagnostics`` section of a configuration file enables one product when it contains ``run: true``:

.. list-table::
   :header-rows: 1

   * - Block
     - Realization output
     - Product
   * - ``timeseries``
     - ``timeseries``
     - :ref:`ensemble_timeseries`
   * - ``biases``
     - ``annual_climatology`` and, with ``seasons: true``, ``seasonal_climatology``
     - :ref:`ensemble_maps`
   * - ``seasonalcycles``
     - ``seasonalcycles``
     - :ref:`ensemble_profiles`
   * - ``lat_lon_profiles``
     - ``zonal_profile``
     - :ref:`ensemble_profiles`
   * - ``zonal``
     - ``annual_climatology`` of 3D variables
     - :ref:`ensemble_zonal`
   * - ``seaice_2d_bias``
     - ``bias`` of the sea ice diagnostic
     - :ref:`ensemble_seaice`

Missing realizations
--------------------

If the file of a realization is missing, a warning lists the realizations involved and the statistics are computed from the remaining ones.
A product is skipped if less than two realizations are available.
Check the logs of each collection: a warning does not make the analysis fail, and the statistics of a variable may then use fewer members than those of the others.

Reference data is read from the output of the realizations too, from the first folder that contains it.
If it is not available, only the ensemble products are plotted.
