![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)

# AQUA-diagnostics

AQUA-diagnostics contains the full set of diagnostic tools developed for the Destination Earth Adaptation Climate Digital Twin (ClimateDT).
It is designed to be used together with the [AQUA core framework](https://github.com/DestinE-Climate-DT/AQUA), which provides data access and preprocessing functionalities.

This repository includes only the scientific tools which are configured as diagnostics to run for online monitoring of climate simulations.

The diagnostics can be executed standalone (Python API), or through the AQUA analysis wrapper (aqua analysis) provided by AQUA-core.

## Installation

AQUA-diagnostics requires:
- Python >=3.10,<3.13
- A working installation of AQUA-core
- A conda/mamba environment using packages from conda-forge

### Install AQUA-core

Follow installation instructions in the AQUA-core repository:
📘 https://github.com/DestinE-Climate-DT/AQUA

### Install AQUA-diagnostics

git clone git@github.com:DestinE-Climate-DT/AQUA-diagnostics.git
cd AQUA-diagnostics
mamba env create -f environment.yml
mamba activate aqua-diagnostics
pip install -e .

The environment contains only the dependencies needed to run the diagnostics;
AQUA-core must be accessible in the environment (installed or in editable mode).

## Container usage

Diagnostics can also be executed inside the AQUA container.
Refer to the Container chapter in the AQUA documentation for details.

## Documentation

Documentation for the diagnostics is part of the AQUA main documentation:
📘 https://aqua.readthedocs.io/en/latest/

This includes:
	•	Usage examples for each diagnostic
	•	Configuration files (YAML)
	•	How to run the analysis wrapper (aqua analysis)
	•	Guidance for writing custom diagnostics

## Examples

Examples and notebooks demonstrating the diagnostics are available in the notebooks directory.

## Contributing

Contributions are welcome!
Please refer to the Contribution Guidelines in this repository.

## License

AQUA-diagnostics is open-source under the Apache 2.0 License.
Copyright belongs to the European Union, represented by the European Commission.
The work is funded by Contract DE_340_CSC — Destination Earth Programme Climate Adaptation Digital Twin (Climate DT).
