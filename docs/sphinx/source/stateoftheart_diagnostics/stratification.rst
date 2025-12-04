.. _stratification:

Ocean Stratification Diagnostic
================================

Description
-----------

The **Stratification** diagnostic is a set of tools for the analysis and visualization of ocean stratification and mixed layer depth (MLD) in climate model outputs.
It analyzes vertical profiles of temperature and salinity to compute potential density and diagnose the ocean's vertical structure.
The diagnostic supports comparative analysis between a target dataset (typically a climate model) and a reference dataset, commonly an observational product such as EN4.

The Stratification diagnostic provides tools to:

- Compute potential density from temperature and salinity
- Calculate mixed layer depth (MLD) from density profiles
- Generate vertical stratification profiles
- Compare model outputs with observations
- Analyze regional ocean stratification patterns
- Produce climatological stratification maps

Classes
-------

There are three main classes for the analysis and plotting:

* **Stratification**: retrieves the data and performs the stratification analysis.
  It handles the computation of potential density from temperature and salinity, computes mixed layer depth when requested, and prepares climatologies.
  The class converts practical salinity to absolute salinity and potential temperature to conservative temperature using the TEOS-10 standard.
  Results are saved as class attributes and as NetCDF files.

* **PlotStratification**: provides methods for plotting vertical stratification profiles.
  It generates plots of temperature, salinity, and density as a function of depth for specified regions.
  Optionally compares model data against observational datasets.

* **PlotMLD**: provides methods for plotting mixed layer depth maps.
  It generates spatial maps of MLD for climatological periods and can compare model MLD against observational datasets.

File structure
--------------

* The diagnostic is located in the ``aqua/diagnostics/ocean_stratification`` directory, which contains both the source code and the command line interface (CLI) script.
* A template configuration file is available at ``aqua/diagnostics/templates/diagnostics/config-stratification.yaml``
* Additional configuration files are available in the ``aqua/diagnostics/config/diagnostics/ocean3d`` directory.
* Notebooks are available in the ``notebooks/diagnostics/ocean_stratification`` directory and contain examples of how to use the diagnostic.

Input variables and datasets
----------------------------

The diagnostic requires ocean temperature and salinity data:

* ``thetao`` (potential temperature) - in degrees Celsius
* ``so`` (salinity) - in PSU (Practical Salinity Units)

These variables must be 3D fields with vertical levels. The diagnostic automatically converts:

* Practical salinity to absolute salinity (using TEOS-10)
* Potential temperature to conservative temperature (using TEOS-10)
* Computes potential density anomaly (sigma_0) referenced to 0 dbar

By default, the diagnostic can compare against the EN4 observational dataset, but it can be configured to use any other dataset as a reference.

The diagnostic is designed to work with 3D ocean data at monthly frequency. Regional analysis is supported for specific ocean regions.

Basic usage
-----------

The basic usage of this diagnostic is explained with a working example in the notebook.
The basic structure of the analysis is the following:

.. code-block:: python

    from aqua.diagnostics.ocean_stratification import Stratification, PlotStratification, PlotMLD

    # Initialize stratification diagnostic for model data
    model_strat = Stratification(
        model='FESOM',
        exp='hpz3',
        source='monthly-3d',
        loglevel="DEBUG"
    )
    
    # Run the diagnostic for a specific region and climatology
    model_strat.run(
        region='ls',  # Labrador Sea
        var=['thetao', 'so'],
        dim_mean=['lat', 'lon'],
        mld=True,
        climatology='January',
        outputdir='./',
        rebuild=True
    )
    
    # Initialize stratification diagnostic for reference data (e.g., EN4)
    obs_strat = Stratification(
        model='EN4',
        exp='v4.2.2',
        source='monthly',
        regrid='r100',
        loglevel="DEBUG"
    )
    
    obs_strat.run(
        region='ls',
        var=['thetao', 'so'],
        dim_mean=['lat', 'lon'],
        mld=True,
        climatology='January',
        outputdir='./',
        rebuild=True
    )
    
    # Plot stratification profiles
    strat_plot = PlotStratification(
        data=model_strat.data[['thetao', 'so', 'rho']],
        obs=obs_strat.data[['thetao', 'so', 'rho']],
        diagnostic_name='ocean_stratification',
        outputdir='./'
    )
    strat_plot.plot_stratification(save_pdf=True, save_png=True)
    
    # Plot mixed layer depth maps
    mld_plot = PlotMLD(
        data=model_strat.data[['mld']],
        obs=obs_strat.data[['mld']],
        diagnostic_name='ocean_stratification',
        outputdir='./'
    )
    mld_plot.plot_mld(save_pdf=True, save_png=True)

.. note::

    Start/end dates and reference dataset can be customized.
    Regions can be specified using predefined region names (e.g., 'ls' for Labrador Sea, 'ws' for Weddell Sea).
    If not specified otherwise, plots will be saved in PNG and PDF format in the current working directory.

CLI usage
---------

The diagnostic can be run from the command line interface (CLI) by running the following command:

.. code-block:: bash

    cd $AQUA/aqua/diagnostics/ocean_stratification
    python cli_ocean_stratification.py --config <path_to_config_file>

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

The configuration file is a YAML file that contains the details on the dataset to analyse or use as reference, the output directory and the diagnostic settings.
Most of the settings are common to all the diagnostics (see :ref:`diagnostics-configuration-files`).
Here we describe only the specific settings for the stratification diagnostic.

* ``ocean_stratification``: a block (nested in the ``diagnostics`` block) containing options for the Ocean Stratification diagnostic.

    * ``stratification``: nested block containing stratification-specific parameters.
    
        * ``run``: enable/disable the diagnostic.
        * ``diagnostic_name``: name of the diagnostic. Typically ``'ocean_stratification'`` or ``'ocean3d'``.
        * ``var``: list of variables to retrieve, typically ``['thetao', 'so']``.
        * ``regions``: list of ocean regions to analyse (e.g., ``['ls', 'ws', 'gs']``).
        * ``climatology``: list of climatological periods to compute (e.g., ``['January', 'JJA', 'DJF']``).
        * ``dim_mean``: dimensions over which to compute spatial means (e.g., ``['lat', 'lon']``).

.. note::

    The ``regions`` and ``climatology`` lists are paired (zipped) together. If you want to analyze the same region with different climatologies, you need to repeat the region name.

.. code-block:: yaml

    diagnostics:
      ocean_stratification:
        stratification:
          diagnostic_name: 'ocean3d'
          run: true
          var: ['thetao', 'so']
          regions: ['ls', 'is', 'ws', 'gs']
          climatology: ['DJF', 'JJA', 'JJA', 'DJF']

Available regions
^^^^^^^^^^^^^^^^^

The diagnostic supports predefined ocean regions for analysis:

* ``ls``: Labrador Sea
* ``is``: Irminger Sea
* ``ws``: Weddell Sea
* ``gs``: Greenland Sea
* ``ros``: Ross Sea

Custom regions can be defined in the configuration or through the diagnostic core region selection functionality.

Output
------

The diagnostic produces three types of outputs:

* **Vertical stratification profiles**: plots showing temperature, salinity, and density as functions of depth for the specified region.
  When observational data is provided, the plots include both model and observational profiles for comparison.

* **Mixed layer depth maps**: spatial maps showing the distribution of MLD for the specified climatological period.
  Includes model data and optionally observational comparison.

* **NetCDF data files**: processed datasets containing computed potential density and mixed layer depth, along with the original temperature and salinity fields.

Plots are saved in both PDF and PNG format.
Data outputs are saved as NetCDF files.

Observations
------------

The default reference dataset is EN4 (Met Office Hadley Centre observation dataset), version 4.2.2.

The diagnostic uses EN4 monthly averages with vertical levels from the AQUA catalog (``model=EN4``, ``exp=v4.2.2``, ``source=monthly``).
EN4 provides quality-controlled ocean temperature and salinity profiles from various observational platforms.

Custom reference datasets can be configured in the configuration file.

Example Plots
-------------

All plots can be reproduced using the notebooks in the ``notebooks`` directory on LUMI HPC.

.. figure:: figures/IFS-NEMO-historical-1990-lra-r100-monthly_stratification_Feb_clim_labrador_sea-1.jpg
    :align: center
    :width: 100%

    Vertical stratification profiles of temperature, salinity, and density from IFS-NEMO historical-1990 for the Labrador Sea in February, compared with EN4 observations.

Available demo notebooks
------------------------

Notebooks are stored in ``notebooks/diagnostics/ocean_stratification``:

* `stratification.ipynb <https://github.com/DestinE-Climate-DT/AQUA-diagnostics/tree/main/notebooks/diagnostics/ocean_stratification/stratification.ipynb>`_

Authors and contributors
------------------------

This diagnostic is maintained by the AQUA team.
Contributions are welcome — please open an issue or a pull request.
For questions or suggestions, contact the AQUA team or the maintainers.

Detailed API
------------

This section provides a detailed reference for the Application Programming Interface (API) of the ``ocean_stratification`` diagnostic,
produced from the diagnostic function docstrings.

.. automodule:: aqua.diagnostics.ocean_stratification
    :members:
    :undoc-members:
    :show-inheritance:
