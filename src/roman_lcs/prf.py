"""
Class to manage PSF loading and evaluation
"""
import os
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.convolution import Box2DKernel, convolve
from scipy import interpolate

from . import PACKAGEDIR

PATH = os.path.dirname(os.path.dirname(PACKAGEDIR))

class RomanPRF(object):
    """
    Class to manage PSF loading and evaluation for Roman WFI observations.
    
    Loads pre-computed PSF models from FITS files and provides interpolation
    for PSF evaluation at arbitrary positions.
    
    Parameters
    ----------
    sca : int, optional
        Sensor chip assembly number (default: 2)
    filter : str, optional
        Filter name for PSF model (default: "F146")
    sky_space : bool, optional
        If True, express coordinates in sky space using pixel scale.
        If False, use detector coordinates (default: False)
    conv : bool, optional
        If True, apply box convolution to PSF model (default: True)
    
    Attributes
    ----------
    prf : ndarray
        2D array containing the PSF model
    oversample_factor : int
        Oversampling factor of the PSF model
    pixel_scale : float
        Pixel scale in appropriate units
    detector_center : tuple
        (x, y) coordinates of detector center
    drow, dcol : ndarray
        Row and column coordinate grids relative to PSF center
    interp_func : scipy.interpolate.RectBivariateSpline
        Interpolation function for PSF evaluation
    
    Raises
    ------
    FileNotFoundError
        If PSF model file not found at expected path
    """
    def __init__(
        self,
        sca: int = 2,
        filter: str = "F146",
        sky_space: bool = False,
        conv: str = True,
        file_name: Optional[str] = None,
    ):

        if file_name is not None:
            self.fname = file_name
        else:
            self.fname = f"{PATH}/data/prf_models/rimtimsim_wfi_psfmodel_{filter}_SCA{sca:02}_spectype_M0V_jitter_12mas_nlambda_10.fits"
        
        if not os.path.isfile(self.fname):
            print(self.fname)
            raise FileNotFoundError
        
        self.filter = filter
        self.sca = sca
        self.hdul = fits.open(self.fname)

        self.prf = self.hdul[0].data
        self.oversample_factor = self.hdul[0].header["OVERSAMP"]
        self.naxis = (self.hdul[0].header["NAXIS1"], self.hdul[0].header["NAXIS2"])
        self.difraction_limit = self.hdul[0].header["DIFFLMT"]
        self.pixel_scale = self.hdul[0].header["PIXELSCL"] * self.oversample_factor
        self.detector_center = (
            self.hdul[0].header["DET_X"],
            self.hdul[0].header["DET_Y"],
        )
        self.detector = self.hdul[0].header["DETECTOR"]

        y = np.linspace(0, self.prf.shape[0], num=self.prf.shape[0])
        x = np.linspace(0, self.prf.shape[1], num=self.prf.shape[1])
        x, y = np.meshgrid(x, y, indexing="xy")

        rpos = y[:, 0].mean()
        cpos = x[0, :].mean()

        self.dy = (y - rpos) / self.oversample_factor
        self.dx = (x - cpos) / self.oversample_factor

        if sky_space:
            self.dy *= self.pixel_scale
            self.dx *= self.pixel_scale

        if conv:
            box_kernel = Box2DKernel(self.oversample_factor)
            self.prf = convolve(self.prf, box_kernel)
        
        self.prf_sum = self.prf.sum()

        self._build_interpolator()

    def _build_interpolator(self):
        """
        Build 2D interpolation function for PSF model.
        
        Creates a RectBivariateSpline interpolator from the loaded PSF data
        for efficient evaluation at arbitrary positions.
        """
        self.interp_func = interpolate.RectBivariateSpline(
            self.dx[0, :], self.dy[:, 0], self.prf
        )

    def evaluate_from_position(
        self,
        center: tuple[float, float] = (12.5, 12.5), 
        shape: tuple[int, int] = (25, 25), 
        corner: tuple[int, int] = (0, 0),
        transpose: bool = True,
        ):
        """
        Evaluate PSF at a specified position within a region.

        Parameters
        ----------
        center : tuple of float, optional
            (row, col) center position of PSF in pixel coordinates (default: (12.5, 12.5))
        shape : tuple of int, optional
            (height, width) of evaluation region in pixels (default: (25, 25))
        corner : tuple of int, optional
            (row, col) corner position of evaluation region (default: (0, 0))

        Returns
        -------
        x : ndarray
            X coordinates of evaluation grid
        y : ndarray
            Y coordinates of evaluation grid
        psf_values : ndarray
            2D array of interpolated PSF values at grid positions
        """
        dx = np.arange(corner[1], corner[1] + shape[1], dtype=float)
        dy = np.arange(corner[0], corner[0] + shape[0], dtype=float)
        dx, dy = np.meshgrid(dx, dy, indexing="xy")

        dx -= (center[1])
        dy -= (center[0])
        eval = self.interp_func(dy, dx, grid=False)
        # if transpose:
        #     eval = eval.T

        return dx + center[1], dy + center[0], eval

    def evaluate_from_array(self, dx: np.ndarray, dy: np.ndarray):
        """
        Evaluate PSF at specified offset coordinates.
        
        Parameters
        ----------
        dx : ndarray
            X-offset coordinates from PSF center
        dy : ndarray
            Y-offset coordinates from PSF center
        
        Returns
        -------
        psf_values : ndarray
            Interpolated PSF values at given offset coordinates
        """
        return self.interp_func(dx, dy, grid=False)



