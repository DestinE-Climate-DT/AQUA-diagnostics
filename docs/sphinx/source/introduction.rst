Introduction
============

Overview of AQUA-diagnostics
----------------------------

AQUA-diagnostics is a dedicated module of AQUA (Climate DT Applications for QUality Assessment) 
for running and managing diagnostics on low-resolution and high-resolution climate model data.  
This module provides all the tools and interfaces needed to configure, launch, and organise 
diagnostic analyses.

.. note::
   For detailed information about AQUA's internal mechanisms (data reading, catalog management, 
   regridding, etc.), please refer to the documentation of `AQUA-core`, which contains all core 
   functions and base classes.

The main repository for AQUA is the `AQUA-core <https://github.com/DestinE-Climate-DT/AQUA>`_ repository.
The main repository for AQUA-diagnostics is the `AQUA-diagnostics <https://github.com/DestinE-Climate-DT/AQUA-diagnostics>`_ repository.

The AQUA-diagnostics repository contains the full set of diagnostic tools developed for the Destination Earth Adaptation Climate Digital Twin (ClimateDT).
It is designed to be used together with the AQUA core framework which provides data access and preprocessing functionalities.

- For more information on AQUA-core, see the AQUA-core documentation here: https://aqua.readthedocs.io/en/latest/.  

Purpose
-------

The purpose of AQUA-diagnostics is to offer users a centralized environment to:
- Select and configure available diagnostics
- Run analyses on climate model outputs
- Easily organize and access diagnostic results

Key Features
------------

- Wide selection of ready-to-use diagnostics for climate model evaluation
- Simple and customizable command-line interface
- Automated management of configuration and analysis parameters
- Support for parallel execution of diagnostics, including HPC cluster integration
- Organized output structure for easy result retrieval
- Integration with AQUA-core for data access and core functionalities

Contributing
------------

AQUA-diagnostics is developed within the European Union Contract 
`DE_340_CSC - Destination Earth Programme Climate Adaptation Digital Twin (Climate DT)`.  
Contributions are welcome via the GitHub repository.  
Please refer to the ``CONTRIBUTING.md`` file for guidelines and further information.