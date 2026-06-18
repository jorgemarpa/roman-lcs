"""
Class to manage PSF loading and evaluation
"""
import os
from typing import Optional

import numpy as np
from astropy.io import fits

# from astropy.convolution import Box2DKernel, convolve
from scipy import interpolate
from scipy.ndimage import convolve, gaussian_filter, uniform_filter
from stpsf import roman
from tqdm import tqdm

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
        interp_method: str = "linear",
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
            # box_kernel = Box2DKernel(self.oversample_factor)
            # self.prf = convolve(self.prf, box_kernel)

            self.prf = uniform_filter(self.prf, size=self.oversample_factor)
        
        self.prf_sum = self.prf.sum()

        self._build_interpolator(method=interp_method)

    def _build_interpolator(self, method: str = "linear"):
        """
        Build 2D interpolation function for PSF model.
        
        Creates a RectBivariateSpline interpolator from the loaded PSF data
        for efficient evaluation at arbitrary positions.
        """
        self.interp_func = interpolate.RegularGridInterpolator(
            (self.dx[0, :], self.dy[:, 0]), self.prf, method=method
        )
        return

    def evaluate_from_position(
        self,
        center: tuple[float, float] = (12.5, 12.5), 
        shape: tuple[int, int] = (25, 25), 
        corner: tuple[int, int] = (0, 0),
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
        eval = self.interp_func((dy, dx))
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
        return self.interp_func((dy, dx))




def apply_wfi_detector_effects(
    psf_data, ipc_center=0.9119, symmetric_ipc=True, diffuse=False, diffusion_sigma=0.3
):
    """
    Applies Inter-Pixel Capacitance (IPC) and Charge Diffusion to a PSF image.
    
    Parameters:
    - psf_data: 2D numpy array (the monochromatic PSF)
    - ipc_alpha: Coupling coefficient for nearest neighbors (typical WFI ~0.008)
    - diffusion_sigma: Sigma for Gaussian charge diffusion in pixels.
    """
    # apply charge diffusion (Gaussian blur). this is already incorporated in latest
    # version of STPSF
    if diffuse:
        psf_data = gaussian_filter(psf_data, sigma=diffusion_sigma)
    
    # apply inter-pixel capacitance (IPC)
    # Standard 3x3 IPC kernel
    # See Roman WFI detector documentation for updated alpha values per SCA
    if symmetric_ipc:
        # using central value and symmetric matrix given in Roman Doc:
        # https://roman.ipac.caltech.edu/page/param-db
        ipc_res = (1 - ipc_center)/4
        ipc_kernel = np.array(
            [[0, ipc_res, 0], [ipc_res, ipc_center, ipc_res], [0, ipc_res, 0]]
        )
    else:
        # taken from:
        # https://roman.gsfc.nasa.gov/science/RRI/Roman_WFI_Reference_Information_20210125.pdf
        ipc_kernel = np.array(
            [[0.21, 1.90, 0.21], [1.84, 91.19, 1.82], [0.22, 1.98, 0.22]]
        ) / 100.0

    
    psf_data = convolve(psf_data, ipc_kernel)
    return psf_data

def generate_chromatic_psf_library(
    filter_name="F146",
    detector="WFI01",
    positions=[(1000, 1000)],
    step_um=0.05,
    oversample=7,
    dir_path=None,
):
    """
    Generates a library of monochromatic PRFs across the filter bandpass.
    """
    wfi = roman.WFI()
    wfi.filter = filter_name
    wfi.detector = detector
    wfi.options["parity"] = "odd"
    if isinstance(positions, tuple):
        positions = [positions]
    
    # define wavelength grid based on F146 (approx 0.93 - 2.00 um)
    # using microns as input, converting to meters for STPSF
    wavelens_um = np.arange(0.95, 2.05, step_um)
    
    psf_library = {}

    print(f"Generating PSF library for {filter_name} {detector}")
    print(f"Wavelength range: {wavelens_um[0]:.2f} - {wavelens_um[-1]:.2f} um, step: {step_um:.2f} um")
    print(f"Total Positions: {len(positions)}")
    for pos in tqdm(positions, total=len(positions), desc=f"Processing positions"):
        wfi.detector_position = pos
        pos_key = f"{pos[0]}_{pos[1]}"
        psf_library[pos_key] = {}
        for wl in tqdm(
            wavelens_um,
            total=len(wavelens_um),
            desc=f"Processing wavelength",
            leave=False,
        ):
            wl_meters = wl * 1e-6
            
            # calculate monochromatic PSF
            # 'fov_pixels' should be sized to capture the wings for high-precision photometry
            hdul = wfi.calc_psf(
                monochromatic=wl_meters, fov_pixels=101, oversample=oversample
            )
            
            # ext 0 is the oversampled PSF
            # ext 1 is the detector sampled PSF
            # ext 2 is the oversampled distorted PSF, including charge diffusion
            # ext 3 is the detector sampled distorted PSF, including charge diffusion
            
            # apply detector effects (IPC & diffusion) this should be applied to 
            # detector sampled PSF
            over_prf_dist = apply_wfi_detector_effects(hdul[2].data, diffuse=False)
            det_prf_dist = apply_wfi_detector_effects(hdul[3].data, diffuse=False)

            hdul[2].data = over_prf_dist
            hdul[2].header.set("IPCTYPE", "symmetric", "IPC kernel definition", after="CHDFSIGM")
            hdul[2].header.set("IPCENT", 0.9119, "IPC kernel definition", after="IPCTYPE")
            hdul[2].header["HISTORY"] = "Applied IPC"
            
            hdul[3].data = det_prf_dist
            hdul[3].header.set("IPCTYPE", "symmetric", "IPC kernel definition", after="CHDFSIGM")
            hdul[3].header.set("IPCENT", 0.9119, "IPC kernel definition", after="IPCTYPE")
            hdul[3].header["HISTORY"] = "Applied inter-pixel capacitance (IPC)"

            if dir_path is not None:
                output_file = os.path.join(dir_path, f"roman_psf_{filter_name}_{detector}_{pos_key}_{wl:.2f}um.fits")
                hdul.writeto(output_file, overwrite=True)
            
            psf_library[pos_key][wl] = over_prf_dist

    return psf_library