.. _ensemble_maps:

Ensemble Maps diagnostic
==========================

Description
-----------

The **EnsembleMaps** diagnostic provides tools to compute and visualize ensemble statistics of 2D latitude-longitude spatial maps:

- Compute ensemble mean and standard deviation for 2D spatial maps.
- Generate separate maps for ensemble mean and standard deviation.
- Support multiple map projections and bias comparisons against reference datasets.

Classes
-------

There is one class for the analysis and one for plotting:

* **EnsembleMaps**: computes ensemble mean and standard deviation for 2D latitude-longitude spatial maps.
  Results are saved as class attributes and optionally as NetCDF files.

* **PlotEnsembleMaps**: provides methods for plotting spatial maps of ensemble mean, standard deviation, and biases.
  It generates separate maps for each statistic by passing the specific dataset to the `plot()` or `plot_ensemble_diff_bias()` methods.

File structure
--------------

* The diagnostic is located in the ``aqua/diagnostics/ensemble`` directory, which contains both the source code and the command-line interface (CLI) scripts.
* Notebooks are available in the ``notebooks/diagnostics/ensemble`` directory and contain examples of how to use the diagnostic.

Input variables and datasets
----------------------------

Before using the diagnostic, input data must be loaded and merged using the ``Reader`` class via
``aqua.diagnostics.ensemble.util.reader_retrieve_and_merge``. The final merged dataset will contain all the requested ensemble members with appropriate metadata.
Alternatively, data can be provided as a list of NetCDF file paths and merged with ``merge_from_data_files``.
The merged dataset must contain all ensemble members concatenated along a dimension named ``ensemble`` (by default, but customizable).

Variables typically used in this diagnostic include:

* ``2t`` (2 metre temperature)
* ``msl`` (mean sea level pressure)

Example: loading and merging a 2D map ensemble from files into an ``xarray.Dataset``:

.. code-block:: python

   import glob
   from aqua.diagnostics import merge_from_data_files

   file_list = glob.glob(
       '/path/to/maps/*.nc'
   )
   file_list.sort()

   ens_dataset = merge_from_data_files(
       variable='2t',
       model_names=['IFS-FESOM', 'IFS-NEMO'],
       data_path_list=file_list,
       loglevel="WARNING",
       ens_dim="ensemble",
   )

Example: loading via the AQUA Reader

.. code-block:: python

   from aqua.diagnostics import reader_retrieve_and_merge

   ens_dataset = reader_retrieve_and_merge(
       variable='2t',
       catalog_list=['nextgems4', 'climatedt-phase1'],
       model_list=['IFS-FESOM', 'IFS-NEMO'],
       exp_list=['historical-1990', 'historical-1990'],
       source_list=['aqua-atmglobalmean', 'aqua-atmglobalmean'],
       loglevel="WARNING",
       ens_dim="ensemble",
   )

Basic usage
-----------

The basic usage of this diagnostic is explained with a working example in the notebook.
The ensemble analysis is performed on a merged ``2D`` map by the ``EnsembleMaps`` class.
The basic structure is the following:

.. code-block:: python

    from aqua.diagnostics import EnsembleMaps, PlotEnsembleMaps

    atmglobalmean_ens = EnsembleMaps(
        var='2t',
        dataset=ens_dataset,
        ensemble_dimension_name='ensemble',
    )
    atmglobalmean_ens.run()

    ens_latlon_plot = PlotEnsembleMaps(
        model_list=['IFS-FESOM', 'IFS-NEMO'],
    )

    # Plot Mean Map
    ens_latlon_plot.plot(
        var='2t',
        dataset=atmglobalmean_ens.dataset_mean,
        data_name='ensemble_mean',
        save_format=['png', 'pdf'], 
        title='Map of 2t for Ensemble Multi-Model Mean',
        cbar_label='2 meter temperature in K',
    )

    # Plot STD Map
    ens_latlon_plot.plot(
        var='2t',
        dataset=atmglobalmean_ens.dataset_std,
        data_name='ensemble_std',
        title='Map of 2t for Ensemble Multi-Model Standard Deviation',
        cbar_label='2 meter temperature in K',
    )

.. note::

    If not specified otherwise, plots will be saved using ``SAVE_FORMAT`` (PNG, PDF, and SVG)
    in the current working directory.

CLI usage
---------

The diagnostic can be run from the command line interface (CLI) using the following commands:

For running unified diagnostics (Maps + Timeseries):
.. code-block:: bash

    cd $AQUA/aqua/diagnostics/ensemble
    python cli_ensemble.py --config <path_to_config_file>

For running exclusively multi-model ensemble maps:
.. code-block:: bash

    python cli_multi_model_maps_ensemble.py --config <path_to_config_file>

Additionally, the CLI can be run with the following optional arguments:

- ``--config``, ``-c``: Path to the configuration file.
- ``--nworkers``, ``-n``: Number of workers to use for parallel processing.
- ``--cluster``: Cluster to use for parallel processing. By default a local cluster is used.
- ``--loglevel``, ``-l``: Logging level. Default is ``WARNING``.
- ``--catalog``: Catalog to use for the analysis. Can be defined in the config file.
- ``--model``: Model to analyse. Can be defined in the config file.
- ``--exp``: Experiment to analyse. Can be defined in the config file.
- ``--source``: Source to analyse. Can be defined in the config file.
- ``--outputdir``: Output directory for the plots.

Configuration file structure
----------------------------

The configuration file is a YAML file that contains the details on the dataset to analyse or use as reference, the output directory, and the diagnostic settings.
Most of the settings are common to all the diagnostics (see :ref:`diagnostics-configuration-files`).
Here we describe only the specific settings for the ensemble maps diagnostic.

* ``ensemble``: a block (nested in the ``diagnostics`` block) containing options for the Ensemble Maps diagnostic.
  Variable-specific parameters override the defaults.

    * ``run``: enable/disable the diagnostic.
    * ``diagnostic_name``: name of the diagnostic.
    * ``variable``: list of variables to analyse.
    * ``projection``: map projection (e.g., ``robinson``).
    * ``vmin`` / ``vmax``: colorbar limits for the mean bias plots.
    * ``vmin_ensemble`` / ``vmax_ensemble``: colorbar limits for the ensemble mean map.
    * ``vmin_std_ensemble`` / ``vmax_std_ensemble``: colorbar limits for the ensemble standard deviation map.
    * ``cmap``: colormap to use.

Output
------

The diagnostic produces the following outputs:

* 2D spatial map of ensemble mean
* 2D spatial map of ensemble standard deviation
* 2D spatial bias maps (if reference data is provided)

Plots are saved in PDF, PNG, and SVG format by default. Data outputs are saved as NetCDF files.

Example Plots
-------------

All plots can be reproduced using the notebooks in the ``notebooks`` directory on LUMI HPC.

.. figure:: figures/ensemble_2t_LatLon_mean.png
    :align: center
    :width: 100%

    Ensemble mean of multi-model global 2-meter temperature. Models considered are IFS-NEMO and IFS-FESOM.

.. figure:: figures/ensemble_2t_LatLon_STD.png
    :align: center
    :width: 100%

    Ensemble standard deviation of multi-model global 2-meter temperature. Models considered are IFS-NEMO and IFS-FESOM.

Available demo notebooks
------------------------

Notebooks are stored in the ``notebooks/diagnostics/ensemble`` directory and contain usage examples.

* `ensemble_global_2D.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/tree/main/notebooks/diagnostics/ensemble/ensemble_global_2D.ipynb>`_

Authors and contributors
------------------------

This diagnostic is maintained by Maqsood Mubarak Rajput (`@maqsoodrajput <https://github.com/maqsoodrajput>`_, `maqsoodmubarak.rajput@awi.de <mailto:maqsoodmubarak.rajput@awi.de>`_).
Contributions are welcome — please open an issue or a pull request.
For questions or suggestions, contact the AQUA team or the maintainer.