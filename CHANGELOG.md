# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

Unreleased in the current development version (target v0.24.0):

## [v0.23.0]

Main changes:
1. Introduce common `TitleBuilder` class for all diagnostics
2. Rename tools/diagnostics with tools/collections 

Complete list:
- Introduce common `TitleBuilder` class for all diagnostics (#99)
- LatLonProfiles: dask fix (#63) and fixes for description and PDF generation (#82)
- Global Biases: add mean value and RMSE to global bias plot (#132)
- Teleconnections: fix timmean assignment in ENSO, NAO, MJO diagnostics (#129)
- Seaice: Add gridlines in maps (#125)
- Add cross-check workflow for testing against aqua-core main or tag/branch (#126)
- Ocean3D: adapt config files for new working datamodel (#122)
- Add gridlines in sea ice maps (#125)
- Ocean3D Drift: optimize load in memory (#120)
- Rename tools/diagnostics with tools/collections (#111, #121)
- Timeseries: CLI correctly works if `reference` is not provided (#109)
- Base diagnostic cli should not forget config file reader_kwargs (#115)
- Ocean3d trend: optimization of region computation in CLI (#105)
- Remove pandas from environment files (#117)
- Increase seaice tests approximation tolerance and fix base util tests (#112)
- Ensemble: improve the coverage of the tests (#88)

## [v0.22.0]

Main changes:
1. Remove intake-esm dependency
2. Refactoring of the documentation following the new repository structure

Complete list:
- Ocean3D: wrong import fixed (#107)
- Ocean3D: added realization key in plot MLD
- Add missing netcdf4 dependency in development environment (#104)
- Add LaTeX units formatting to diagnostic labels (#70)
- GlobalBiases: few small bugs fixed (#98) 
- Remove intake-esm dependency (#100) 
- Porting ensemble config files as in the issue #78 (#87)
- General refactoring of AQUA-diagnostics documentation (#34)
- Add troubleshooting information for Intel compatibility with gdal and rasterio modules (#84)
- Vertical coordinate for Ocean3d can be passed as a configuration arguemnt (#60)
- Fix of aqua_path for analysis console (#56)
- Boxplots: hotfix for diagnostic_name (#69)
- Histogram: CLI refactoring (#49)
- Fix argument passing by common CLI for `regrid`, `startdate` and `enddate` (#52)
- Introduce call to pd.Timestamp in Global_Bias to accomodate also non-nanoseconds time window (#52)
- Remove time selection from SeaIce, now centralised in CLI (#52)
- Rename `core` diagnostic folder to `base` to avoid mixing up with `aqua.core` (#52)
- Updated installation scripts for HPC2020 (#51)
- fixed as suggested in the issue #59 (#80)

## [v0.21.0]

Main changes:
1. Complete refactor of the repository to accomodate for the first release of AQUA-diagnostics, now depending on `aqua-core` package.
2. `aqua-diagnostics` released on pypi.

Complete list:
- Timeseries: Extra timestep in Timeseries analysis fix (#57)
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

[unreleased]: https://github.com/DestinE-Climate-DT/AQUA-diagnostics/compare/v0.23.0...HEAD
[v0.23.0]: https://github.com/DestinE-Climate-DT/AQUA-diagnostics/compare/v0.22.0...v0.23.0
[v0.22.0]: https://github.com/DestinE-Climate-DT/AQUA-diagnostics/compare/v0.21.0...v0.22.0