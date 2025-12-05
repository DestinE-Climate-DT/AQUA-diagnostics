# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

Unreleased in the current development version (target v0.22.0):

- Fix of aqua_path for analysis console (#56)
- Fix argument passing by common CLI for `regrid`, `startdate` and `enddate` (#52)
- Introduce call to pd.Timestamp in Global_Bias to accomodate also non-nanoseconds time window (#52)
- Remove time selection from SeaIce, now centralised in CLI (#52)
- Rename `core` diagnostic folder to `base` to avoid mixing up with `aqua.core` (#52)
- Updated installation scripts for HPC2020 (#51)

## [v0.21.0]

Main changes:
1. Complete refactor of the repository to accomodate for the first release of AQUA-diagnostics, now depending on `aqua-core` package.
2. `aqua-diagnostics` released on pypi.


Complete list:
- Notebooks update with new plt.close() pattern (#50)
- Workflow structure refactore to accomodate for multiple pipelines as in AQUA-core (#47)
- Removed `cli_checker` diagnostic tool from diagnostic repository (#46)
- Timeseries: Timeseries and Gregory correctly work with less that one year of data (#42)
- Adapt to new folder structure of AQUA core (fixing imports and removing diagnostics/src folder) (#36, #37)
- Complete activation of tests for CI/CD and complete coupling with aqua-core (#17)
- Coverage with coveralls and Zenodo DOI generation included (#17)
- Biases: fix contour in vertical biases plots (#41)
- Porting of the AQUA diagnostics documentation (#18)
- Added GitHub Issues and PR templates (#1)
- CHANGELOG, LICENSE and README files added (#2)
- Add LUMI installation scripts for AQUA-diagnostics (#40)

## Previous versions
Please notice that before v0.21.0 (i.e. up to v0.20.0) aqua-core and aqua-diagnostics have been developed in the same repository. Please refer to AQUA main repo for past changelog

[unreleased]: https://github.com/DestinE-Climate-DT/AQUA-diagnostics/compare/v0.21.0...HEAD