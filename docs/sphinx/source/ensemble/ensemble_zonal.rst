.. _ensemble_zonal:

Ensemble Zonal diagnostic
=========================

Description
-----------

The **EnsembleZonal** diagnostic provides tools to compute and visualize ensemble statistics of zonal-mean level-latitude cross-sections:

- Compute ensemble mean and standard deviation for zonal-mean cross-sections.
- Generate contour plots showing ensemble statistics as functions of latitude and depth/level.

Classes
-------

There is one class for the analysis and one for plotting:

* **EnsembleZonal**: computes ensemble mean and standard deviation for zonal-mean level-latitude cross-sections.
  Results are saved as class attributes and as NetCDF files.

* **PlotEnsembleZonal**: provides methods for plotting zonal cross-sections of ensemble mean and standard deviation.

File structure
--------------

* The diagnostic is located in the ``aqua/diagnostics/ensemble`` directory, which contains both the source code and the command-line interface (CLI) scripts.
* Template configuration files are available in the ``aqua/diagnostics/templates/diagnostics/config-ensemble_zonalmean.yaml`` directory.
* Notebooks are available in the ``notebooks/diagnostics/ensemble`` directory and contain examples of how to use the diagnostic.

Input variables and datasets
----------------------------

Before using the diagnostic, input data must be loaded and merged using the ``Reader`` class via
``aqua.diagnostics.ensemble.util.reader_retrieve_and_merge``. The final merged dataset will contain all the requested ensemble members with appropriate metadata.
Alternatively, data can be provided as a list of NetCDF file paths and merged with ``merge_from_data_files``.
The merged dataset must contain all ensemble members concatenated along a dimension named ``ensemble``.

Typical variables used in this diagnostic include:

* ``so`` (sea water practical salinity)

Example: loading and merging a zonal ``Lev-Lat`` ensemble into an ``xarray.Dataset``:

.. code-block:: python

   import glob
   from aqua.diagnostics import merge_from_data_files

   file_list = glob.glob(
       '/path/to/LevLat/*.nc'
   )
   file_list.sort()

   ens_dataset = merge_from_data_files(
       variable='so',
       model_names=['IFS-FESOM', 'IFS-NEMO'],
       data_path_list=file_list,
       loglevel="WARNING",
       ens_dim="ensemble",
   )

Example: loading via the AQUA Reader

.. code-block:: python

   from aqua.diagnostics import reader_retrieve_and_merge

   ens_dataset = reader_retrieve_and_merge(
       variable='so',
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
The basic structure of the analysis is the following:

.. code-block:: python

    from aqua.diagnostics import EnsembleZonal, PlotEnsembleZonal

    zonal_ens = EnsembleZonal(
        var='so',
        dataset=ens_dataset,
    )
    zonal_ens.run()

    ens_zm_plot = PlotEnsembleZonal(
        model_list=['IFS-NEMO', 'IFS-FESOM'],
    )

    # Plot Mean Zonal Map
    ens_zm_plot.plot(
        var='so',
        dataset=zonal_ens.dataset_mean,
        data_name='ensemble_zonal_mean',
        save_format=['png', 'pdf'],
        title='Mean of Ensemble of Zonal-average of so',
        cbar_label='Time-mean sea water practical salinity g kg**-1/year',
    )

    # Plot STD Zonal Map
    ens_zm_plot.plot(
        var='so',
        dataset=zonal_ens.dataset_std,
        data_name='ensemble_zonal_std',
        title='Standard deviation of Ensemble of Zonal-average of so',
        cbar_label='Time-mean sea water practical salinity g kg**-1/year',
    )

.. note::

    If not specified otherwise, plots will be saved using ``SAVE_FORMAT`` (PNG, PDF, and SVG)
    in the current working directory. The plotting function automatically infers the vertical coordinate names.

CLI usage
---------

The diagnostic can be run from the command-line interface (CLI) by running the following command:

.. code-block:: bash

    cd $AQUA/aqua/diagnostics/ensemble
    python cli_zonal_ensemble.py --config <path_to_config_file>

Additionally, the CLI can be run with the following optional arguments:

- ``--config``, ``-c``: Path to the configuration file.
- ``--nworkers``, ``-n``: Number of workers to use for parallel processing.
- ``--cluster``: Cluster to use for parallel processing.
- ``--loglevel``, ``-l``: Logging level. Default is ``WARNING``.
- ``--catalog``: Catalog to use for the analysis.
- ``--model``: Model to analyse.
- ``--exp``: Experiment to analyse.
- ``--source``: Source to analyse.
- ``--outputdir``: Output directory for the plots.
- ``--startdate``: Start date for the analysis.
- ``--enddate``: End date for the analysis.

Output
------

The diagnostic produces the following outputs:

* Contour plot of ensemble mean as a function of latitude and depth/level
* Contour plot of ensemble standard deviation as a function of latitude and depth/level

Plots are saved in PDF, PNG, and SVG format by default (see ``SAVE_FORMAT``). Data outputs are saved as NetCDF files.

Configuration file structure
----------------------------

The configuration file is a YAML file that contains details on the dataset to analyse, the output directory, and the diagnostic settings.

* ``ensemble``: a block (nested in the ``diagnostics`` block) containing options for the Ensemble Zonal diagnostic.
  Variable-specific parameters override the defaults.

    * ``run``: enable/disable the diagnostic.
    * ``diagnostic_name``: name of the diagnostic (e.g., ``EnsembleZonal``).
    * ``variable``: list of variables to analyse.
    * ``region``: region to analyse (e.g., ``atlantic``).
    * ``plot_params``: Nested dictionaries defining aesthetics like ``figure_size``, ``title_mean``, ``title_std``, and ``cbar_label``.

Example Plots
-------------

All plots can be reproduced using the notebooks in the ``notebooks`` directory on LUMI HPC.

.. figure:: figures/avg_so_LevLon_mean.png
    :align: center
    :width: 100%

    Ensemble-Zonal mean for average Time-mean sea water practical salinity.

.. figure:: figures/avg_so_LevLon_STD.png
    :align: center
    :width: 100%

    Ensemble-Zonal standard deviation for average Time-mean sea water practical salinity.

Available demo notebooks
------------------------

Notebooks are stored in the ``notebooks/diagnostics/ensemble`` directory and contain usage examples.

* `ensemble_zonalaverage.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/tree/main/notebooks/diagnostics/ensemble/ensemble_zonalaverage.ipynb>`_

Authors and contributors
------------------------

This diagnostic is maintained by Maqsood Mubarak Rajput (`@maqsoodrajput <https://github.com/maqsoodrajput>`_, `maqsoodmubarak.rajput@awi.de <mailto:maqsoodmubarak.rajput@awi.de>`_).
Contributions are welcome — please open an issue or a pull request.
