# Temporal Variability Map Diagnostics Application

Author:
- Maqsood Mubarak Rajput (AWI, maqsoodmubarak.rajput@awi.de) (Author and maintainer v0.19.1)

## Description

This application calculates the standard deviation along time dimension for models namely, FESOM, ICON and NEMO. It compares them against a given reference model. The results are saved in NetCDF file format. It also provides visualization of the variability for the models.

## Installation Instructions

To install this diagnostic you can use conda.

No more environments than the regular AQUA-diagnostics ones (located in `./environment.yaml`) are needed.
Refer to the AQUA-diagnostics documentation for more information.

## Configuration
The CLI application requires a YAML configuration file. The `ocean2d` collection one (monthly `zos`) is `aqua/diagnostics/config/collections/legacy/ocean2d/config-ocean2d-aviso.yaml`, standalone examples (daily data, `tos`) are in `aqua/diagnostics/config/tools/variability_map/`.

## Usage
1. Choose one of the configuration files above, or write your own.
2. Run the application via CLI or the notebook available in `notebooks/diagnostics`.

## Note
In order to comapre the plots between the model and the reference, both are regridded to a common grid. This setting can be changed via the configuration file from above. 
