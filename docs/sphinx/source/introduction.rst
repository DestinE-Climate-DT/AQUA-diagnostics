Introduction
============

Overview
--------

`AQUA-diagnostics <https://github.com/DestinE-Climate-DT/AQUA-diagnostics>`_ (Climate DT Applications for QUality Assessment) is a specialized package designed for managing and running diagnostics on climate model datasets both at reduced and native high-resolutions.
It provides robust tools and interfaces for configuring, launching, and organizing diagnostic analyses of climate simulations.
The framework operates in conjunction with AQUA-core, which handles fundamental operations including data access, catalog management, and regridding. 
AQUA-diagnostics builds upon this foundation by implementing a complete suite of diagnostic tools specifically tailored for climate model evaluation. 
The architecture maintains a clear separation between computational analysis classes and visualization components, enabling independent management of data processing and graphical representation.

.. note::

   For detailed information about AQUA's internal mechanisms (data reading, catalog management,
   regridding, etc.), please refer to the documentation of `AQUA-core <https://aqua.readthedocs.io/en/latest/>`_ , which contains all core
   functions and base classes.

Purpose 
-------

AQUA-diagnostics serves as a centralized environment for climate model evaluation, enabling researchers to:

- Select and configure diagnostics from a collection of ready-to-use analysis tools
- Execute comprehensive analyses on climate model outputs with an intuitive command-line interface with extensive customization options
- Organize and retrieve results through a standardized output structure
- Exploit parallel execution capabilities including integration with HPC cluster environments
- Compare model performance against observational benchmarks and previous model versions
- Assess simulation quality through multiple evaluation metrics and performance indices

The Diagnostics:
----------------
The diagnostics within AQUA are organized into three principal categories based on their methodological approach and data requirements:

1. **State-of-the-Art Diagnostics**: These diagnostics employ data at resolutions comparable to observational datasets and traditional climate models (such as CMIP6) to monitor and evaluate climate simulations. 
They focus on detecting model drifts, energy imbalances, and systematic biases through direct comparison with observational data or established climate simulations. 
State-of-the-art diagnostics support the derivation of quantitative evaluation metrics that characterize model performance across multiple dimensions. This category includes:
- Time series analyses tracking key climate variables over extended periods
- Global bias assessments identifying systematic model errors across spatial domains
- Performance indices quantifying model skill through standardized metrics
- Boxplots
- Teleconnection indices assessing representation of large-scale climate patterns
- Ocean circulation diagnostics evaluating oceanic transport and water mass characteristics
- LatLon profiles examining spatial variations in climate variables
- Histograms for distribution analysis of climate variables
- Sea ice extent and concentration diagnostics monitoring polar climate dynamics

To enable efficient execution of these diagnostics, a Low-Resolution Archive (LRA) has been created by downscaling original high-resolution data to coarser spatial (1°x 1°) and temporal (monthly) resolutions. 
This dimensional reduction is essential because state-of-the-art diagnostics do not require high-frequency, fine-scale data and can effectively assess fundamental model behavior using daily or monthly data at coarser resolutions. 
The LRA serves as an intermediate processing layer, enabling faster and more manageable analyses while preserving the information content relevant for climate model assessment.

2. **Frontier Diagnostics**: These diagnostics leverage emerging techniques and high-resolution data to explore new dimensions of model evaluation. 
They aim to address unresolved questions in climate science and push the boundaries of traditional diagnostic approaches. 
Frontier diagnostics may include novel metrics, advanced statistical methods, and machine learning techniques to uncover complex patterns in climate model outputs.
Examples of frontier diagnostics include:
- Sea surface height variability analyses capturing fine-scale ocean dynamics
- Tropical rainfall diagnostics focusing on precipitation distributions and extremes
- Tropical cyclones tracking and zoom-in

3. **Ensemble Diagnostics**: This category focuses on the analysis of multi-model ensembles and large collections of climate simulations.


Design Principles:
------------------

The design of the package is based on a clear separation between Python classes handling
the execution of analyses and those dedicated to visualization and plotting, allowing
computational and graphical representation to be managed independently.

The diagnostic framework emphasizes homogeneity and modularity, ensuring structural consistency across all implemented diagnostics while facilitating extensibility.
At the foundation lies the **Base Diagnostic Class** (`Diagnostic`), which provides essential functionalities shared by all diagnostics:

- Unified initialization methods establishing common interfaces
- AQUA Reader initialization for standardized data access
- Core data retrieval functions interfacing with AQUA-core
- Standardized output saving procedures

Each specific diagnostic class inherits from `Diagnostic` and extends its capabilities by implementing diagnostic-specific parameters and methods.
For example, the Global Biases diagnostic in its own class `GlobalBiases` introduces specialized methods like `compute_climatology()` while leveraging the core retrieval and saving functionalities provided by the base class. 
This inheritance pattern ensures that all diagnostics follow consistent implementation conventions, reducing code redundancy and improving long-term maintainability.
Additionally, each diagnostic incorporates comparison classes, e.g `PlotGlobalBiases` that handle visualization tasks.
These classes provide methods for cross-dataset comparisons (such as `plot_bias()` and `plot_seasonal_bias()`) and integrate configurable options including multiple output formats (`save_pdf`, `save_netcdf`).
Where applicable, comparison classes utilize existing AQUA-core plotting functions to maintain visualization consistency across the entire diagnostic suite.
This structured, modular design enables efficient implementation of new diagnostics with minimal development effort while ensuring scalability and coherence throughout the framework.
Although this standardization has not yet been fully extended to frontier diagnostics, it establishes a robust foundation for future enhancements.


Running Diagnostics
--------------------

AQUA-diagnostics provides three flexible approaches for executing analyses, designed to accommodate different use cases and levels of complexity:

1. **Direct Module Import in Notebooks:**

  For users interested in performing targeted analyses—such as examining biases in a specific variable—individual diagnostic modules can be imported directly into Python notebooks or scripts. 
  This approach offers maximum flexibility, allowing users to call specific functions from particular diagnostics and generate customized plots. 
 
2. **Command-Line Interface with Configuration Files:**

   Each diagnostic in AQUA includes a dedicated command-line interface (CLI) that enables execution of the complete diagnostic workflow. 
   Users configure the analysis by providing a configuration file that specifies all necessary parameters, including dataset selection, temporal ranges, spatial domains, and output options.
   Configuration file templates for all available diagnostics are provided in `aqua/diagnostics/templates`, serving as starting points that users can customize according to their specific requirements.
   This approach ensures reproducibility, as the configuration file documents all analysis settings and can be version-controlled alongside results.Example workflow:

3. **Level 3: Diagnostic collections via AQUA Analysis Wrapper**

   For comprehensive model evaluation involving multiple diagnostics, AQUA provides the `aqua-analysis` wrapper, which orchestrates the execution of **diagnostic-collections**. 
   Users construct a master configuration file that combines multiple diagnostics, defining which analyses to run and how they should be coordinated.
   This approach enables:

   - Thematic evaluation campaigns addressing specific climate system components (e.g. atmosphere, ocean, radiative-balance)
   - Automated multi-diagnostic workflows with dependency management
   - Coordinated parallel execution optimizing computational resources
   - Integrated result organization consolidating outputs from multiple diagnostics

Examples of diagnostic-suite configuration files are provided in `aqua/diagnostics/config/diagnostics`.
The aqua-analysis functionality is fully documented at ref:`aqua_analysis`.
 

Contributing
------------

AQUA-diagnostics is developed within the European Union Contract 
`DE_340_CSC - Destination Earth Programme Climate Adaptation Digital Twin (Climate DT)`.  
Contributions are welcome via the GitHub repository.  
Please refer to the ``CONTRIBUTING.md`` file for guidelines and further information.