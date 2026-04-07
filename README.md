![maintenance-status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aqua-diagnostics?style=flat)](https://pypi.org/project/aqua-diagnostics/)
[![AQUA-diagnostics tests](https://github.com/DestinE-Climate-DT/AQUA-diagnostics/actions/workflows/aqua-diagnostics.yml/badge.svg)](https://github.com/DestinE-Climate-DT/AQUA-diagnostics/actions/workflows/aqua-diagnostics.yml)
[![Documentation Status](https://readthedocs.org/projects/aqua-diagnostics/badge/?version=latest)](https://aqua-diagnostics.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/DestinE-Climate-DT/AQUA-diagnostics/graph/badge.svg?token=UIJTBR9ID0)](https://codecov.io/gh/DestinE-Climate-DT/AQUA-diagnostics)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17776618.svg)](https://doi.org/10.5281/zenodo.17776618)

# AQUA-diagnostics

AQUA-diagnostics contains the full set of diagnostic tools developed for the Destination Earth Adaptation Climate Digital Twin (ClimateDT). It is designed to be used together with the [AQUA core framework](https://github.com/DestinE-Climate-DT/AQUA), which provides data access and preprocessing functionalities.

This repository includes only the scientific tools configured as diagnostics to run for online monitoring of climate simulations. Diagnostics can be executed standalone via the Python API, or through the `aqua analysis` wrapper provided by AQUA-core.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Container Usage](#container-usage)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

- Python `>=3.10,<3.13`
- A working installation of [AQUA-core](https://github.com/DestinE-Climate-DT/AQUA)
- A conda/mamba environment using packages from `conda-forge`

---

## Installation

### 1. Install AQUA-core

Follow the installation instructions in the AQUA-core repository:

> 📘 https://github.com/DestinE-Climate-DT/AQUA

### 2. Install AQUA-diagnostics
```bash
git clone git@github.com:DestinE-Climate-DT/AQUA-diagnostics.git
cd AQUA-diagnostics
mamba env create -f environment.yml
mamba activate aqua-diagnostics
pip install -e .
```

> **Note:** The environment includes only the dependencies needed to run the diagnostics. AQUA-core must be separately accessible in the environment (installed or in editable mode).

---

## Container Usage

Diagnostics can also be executed inside the AQUA container. Refer to the Container chapter in the AQUA documentation for details.

---

## Documentation

Full documentation is available on ReadTheDocs:
📖 [AQUA-diagnostics](https://aqua-diagnostics.readthedocs.io/en/latest/)

Topics covered include:

- Usage examples for each diagnostic
- Configuration files (YAML)
- How to run the analysis wrapper (`aqua analysis`)
- Guidance for writing custom diagnostics

---

## Examples

Notebooks and usage examples are available in the [`notebooks/`](./notebooks) directory.

---

## Contributing

Contributions are welcome! Please refer to the [Contribution Guidelines](./CONTRIBUTING.md) in this repository.

---

## License

AQUA-diagnostics is open-source software licensed under the **Apache 2.0 License**.

Copyright belongs to the European Union, represented by the European Commission.
Funded by Contract **DE_340_CSC** — Destination Earth Programme Climate Adaptation Digital Twin (Climate DT).
