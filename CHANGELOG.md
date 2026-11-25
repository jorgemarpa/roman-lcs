# Changelog

All notable changes to `roman-lcs` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-11-25

### Added
- Batch light curve extraction CLI command (`extract-batch-lightcurves`) and SLURM array runner.
- Single light curve extraction CLI command (`extract-lightcurve`) with AWS S3 direct upload support.
- Performance optimizations:
  - Vectorized aperture photometry routines with circular, SNR-optimal, and isophotal aperture options.
  - Accelerated Cholesky linear solver for PRF matrix inversion with singular fallback.
  - Coordinate memoization for repeated SCA pixel-to-sky evaluations.
  - Threshold-clipped sparse PRF evaluations.
- Pure NumPy / SciPy implementation of precision metrics (P2P scatter, RMS, MAD, and Savitzky-Golay CDPP).
- Robust background modeling: polynomial surface fitting, median background estimation, and neighbor source masking.
- Crowding (`crowdsap`) and flux fraction (`flfrcsap`) aperture metrics.
- PEP 621 packaging with `uv` and `hatchling` backend.
- MkDocs documentation suite with Material theme covering API reference.
- Standardized GitHub Actions CI workflows (`pytest`, `ruff`, `mypy`, `deploy-mkdocs`, `pypi-publish`).

### Changed
- Migrated dependency management and packaging from Poetry to `uv`.
- Decoupled `lightkurve` dependency; precision metrics and export now operate natively with standard Astropy and NumPy structures.
- Embedded runtime FITS header templates into package data (`src/roman_lcs/templates/`).
- Relocated heavy PRF models, photometry logs, and legacy development notebooks to `pipeline/`.

### Fixed
- Fixed centroid calculation and pixel-coordinate projection in multi-extension FITS outputs.
- Corrected NaN and saturated pixel mask dilation in aperture photometry.

## [0.4.1] - 2025-09-30

### Added
- Added documentation for PRF model visualization (`plot_prf_model`) and diagnostic plotting (#8).

## [0.4.0] - 2025-09-29

### Added
- Initial PyPI release of `roman-lcs`.
- PRF shape model improvements and source masking algorithms (#2).
- Dynamic scene background subtraction for crowded field photometry.
