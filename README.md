# Roman-lcs

Tools to do PSF photometry on Roman simulated data from TRExS group.

The PSF toosl are based on [PSFMachine](https://github.com/SSDataLab/psfmachine).

## Installation

This package can be intalled using the `pip` command from this repository.

```
pip install git+https://github.com/jorgemarpa/roman-lcs
```


## Tutorial

For a full tutorial on how to build a PRF model, evaluate it to compute the photometry, and build 
light curves, see the this [Jupyter notebook](notebooks/roman_psfmachine_tutorial.ipynb)


## Simulated Images

The simulated images are produced by the `RimTimSim` [package](https://github.com/robertfwilson/rimtimsim).
Here's an example image:

![sim_img](data/figures/roman_wfi_sca2_F146.png)

## PRF Model

The PRF model is computed from the image itself, using the source catalog to fix the stars positions and fitting all sources at the same time to get the PRF model.
See the figure below for a PRF example:

![prf_model](data/figures/prf_model_field03_sca02_F146_center.png)


## Light Curves

Light curves are computed by fitting the PRF at every frame and saved into FITS files.

![lc1](data/figures/lc_example_01.png)
![lc2](data/figures/lc_example_02.png)