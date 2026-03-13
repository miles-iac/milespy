# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **API clarifications**: Renamed ambiguous public APIs for clarity; old names
  are are still supported as aliases.
- **Validation**: Added Pydantic v2-based validation for index configs
  (`LineStrengthIndexConfig`), magnitude configs
  (`MagnitudeComputationConfig`), and SSP library configs where applicable;
- band edges and zeropoints are validated with clear error messages.
- **Refactors**: Extracted helper functions in `magnitudes.compute_mags` and in
  `ssp_library` `__init__` for readability; renamed `misc.py` to
  `interpolation_utils.py` with a deprecated shim.
- **Documentation**: Added module-level docstrings across core modules;
  expanded docstrings for repository, SFH, spectra, and SSP library; changelog
  added to documentation navigation.

### Added

- **Dependencies**: `pydantic>=2,<3` for lightweight configuration validation.

### Fixed

- Docstring and comment fixes (e.g. "Angstron" → "Angstrom"); clarified
  parameter/return descriptions; narrowed broad `except` in repository file
  checks to specific exceptions.
