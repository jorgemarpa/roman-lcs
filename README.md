# Roman-lcs

Tools to do PSF photometry on Roman simulated data from TRExs group.

The PSF toosl are based on [PSFMachine](https://github.com/SSDataLab/psfmachine).

## Simulated Images

The simulated images are produced by the `RimTimSim` package.
Here's an exmaple

## PSF Model

The PSF model is computed from the image itself, using the source catalog to fix the stars positions and fitting all sources at the same time to get the PRF model.
See the figure below for a PRF example:


## Light Curves

Light curves are computed by fitting PSF photometry at every frame and saved into FITS files similar to TESS light curve files.