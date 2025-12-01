# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)


## [Unreleased]

ClimateDT workflow modifications:

Complete list:
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

Unreleased in the current development version:
- Add LUMI installation scripts for AQUA-diagnostics (#40)