"""Subclass of `Machine` that Specifically work with FFIs"""

import logging
import os
from enum import IntFlag
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import astropy.units as u
import lightkurve as lk
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper
from roman_cuts import RomanCuts
from scipy import ndimage, sparse
from tqdm import tqdm

from .aperture import (
    compute_CROWDSAP,
    compute_FLFRCSAP,
    correct_aperture_flux,
    make_snr_optimal_aperture,
)
from .machine import Machine
from .prf import RomanSimplePSF
from .utils import _make_A_cartesian, solve_linear_model

log = logging.getLogger(__name__)


def _format_aperture_key(radius: Union[int, float]) -> str:
    """
    Format aperture radius into key string, e.g. 3 -> 'aperture_3pix'.
    """
    if isinstance(radius, (int, np.integer)) or (
        isinstance(radius, (float, np.floating)) and float(radius).is_integer()
    ):
        return f"aperture_{int(radius)}pix"
    return f"aperture_{radius}pix"


class PRFFitFlag(IntFlag):
    """
    Bitmask flags tracking PRF scene model fitting quality and convergence.

    Attributes
    ----------
    SUCCESS : int (0)
        Fit succeeded and converged normally.
    NOT_CONVERGED : int (1)
        Iterative centroid shift did not reach convergence threshold within max iterations.
    MAX_ITER_REACHED : int (2)
        Iterative fitting stopped because maximum iterations was reached.
    LINALG_FAILED : int (4)
        Linear solver failed (LinAlgError or singular design matrix).
    NON_FINITE_WEIGHTS : int (8)
        Linear solve produced non-finite (NaN or Inf) weights or flux errors.
    INSUFFICIENT_UNMASKED_PIXELS : int (16)
        Insufficient unmasked pixels in cutout to constrain model.
    DIVERGED_OFFSET : int (32)
        Centroid offset step diverged or exceeded physical bound (> 1.5 pix).
    NEGATIVE_FLUX : int (64)
        Fitted flux for one or more target stars is negative.
    HIGH_CHI2 : int (128)
        Model reduced chi-squared is unusually high (> 10.0).
    """
    SUCCESS = 0
    NOT_CONVERGED = 1 << 0
    MAX_ITER_REACHED = 1 << 1
    LINALG_FAILED = 1 << 2
    NON_FINITE_WEIGHTS = 1 << 3
    INSUFFICIENT_UNMASKED_PIXELS = 1 << 4
    DIVERGED_OFFSET = 1 << 5
    NEGATIVE_FLUX = 1 << 6
    HIGH_CHI2 = 1 << 7


class _SafeWCSWrapper(BaseWCSWrapper):
    """
    Wraps a LowLevelWCS to ensure longitude coordinates are normalized to [0, 360)
    during world_to_pixel transformations. This prevents GWCS from returning NaNs
    when Astropy's find_coordinate_range normalizes celestial RA coordinates into [-180, 180).
    """

    def pixel_to_world_values(self, *pixel_arrays):
        return self._wcs.pixel_to_world_values(*pixel_arrays)

    def world_to_pixel_values(self, *world_arrays):
        world_list = list(world_arrays)
        if len(world_list) > 0:
            world_list[0] = np.asarray(world_list[0]) % 360.0
        return self._wcs.world_to_pixel_values(*world_list)


class LCExtractor(Machine):
    """
    Subclass of Machine for working with Roman data.
    """

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        sources: pd.DataFrame,
        column: np.ndarray,
        row: np.ndarray,
        cadenceno: Optional[np.ndarray] = None,
        wcs: Optional[Any] = None,
        sparse_dist_lim: int = 4,
        quality_mask: Optional[np.ndarray] = None,
        sources_flux_column: str = "flux",
        sources_mag_column: str = "F146",
        meta: Optional[dict] = None,
        dithered: bool = True,
    ) -> None:
        """
        Repeated optional parameters are described in `Machine`.

        Parameters
        ----------
        time: numpy.ndarray
            Time values in JD
        flux: numpy.ndarray
            Flux values at each pixels and times in units of electrons / sec. Has shape
            [n_times, n_rows, n_columns]
        flux_err: numpy.ndarray
            Flux error values at each pixels and times in units of electrons / sec.
            Has shape [n_times, n_rows, n_columns]
        ra: numpy.ndarray
            Right Ascension coordinate of each pixel
        dec: numpy.ndarray
            Declination coordinate of each pixel
        sources: pandas.DataFrame
            DataFrame with source present in the images
        column: np.ndarray
            Data array containing the "columns" of the detector that each pixel is on.
        row: np.ndarray
            Data array containing the "columns" of the detector that each pixel is on.
        wcs : astropy.wcs
            World coordinates system solution for the FFI. Used for plotting.
        quality_mask : np.ndarray or booleans
            Boolean array of shape time indicating cadences with bad quality.
        meta : dictionary
            Meta data information related to the FFI

        Attributes
        ----------
        meta : dictionary
            Meta data information related to the FFI
        flux_3d : numpy.ndarray
            2D image representation of the FFI, used for plotting. Has shape [n_times,
            image_height, image_width]
        image_shape : tuple
            Shape of 2D image
        """

        self.ref_frame = 0
        if cadenceno is not None:
            self.cadenceno = cadenceno 
        else:
            self.cadenceno = np.arange(time.shape[0])

        self.meta = meta
        self.dithered = dithered

        # keep 2d image shape
        self.image_shape = flux.shape[1:]
        self.sources_mag_column = sources_mag_column

        flux = flux.reshape((-1, np.multiply(*self.image_shape)))
        flux_err = flux_err.reshape((-1, np.multiply(*self.image_shape)))
        ra = ra.reshape((-1, np.multiply(*self.image_shape)))
        dec = dec.reshape((-1, np.multiply(*self.image_shape)))
        row = row.reshape((-1, np.multiply(*self.image_shape)))
        column = column.reshape((-1, np.multiply(*self.image_shape)))

        # init `machine` object
        super().__init__(
            time,
            flux,
            flux_err,
            ra,
            dec,
            sources,
            column,
            row,
            wcs=wcs,
            sparse_dist_lim=sparse_dist_lim,
            sources_flux_column=sources_flux_column,
        )
        self._mask_pixels()
        self.median_bkg = None
        self.bkg_pixel_mask = None
        self._cached_pixel_coords = {}
        if quality_mask is None:
            self.quality_mask = np.zeros(len(time), dtype=int)
        else:
            self.quality_mask = quality_mask

    def __repr__(self) -> str:
        return f"LCExtractor (N sources, N times, N pixels): {self.shape}"

    @property
    def flux_3d(self) -> np.ndarray:
        return self.flux.reshape((self.nt, *self.image_shape))

    @property
    def flux_err_3d(self) -> np.ndarray:
        return self.flux_err.reshape((self.nt, *self.image_shape))

    @property
    def row_3d(self) -> np.ndarray:
        return self.row.reshape((-1, *self.image_shape))

    @property
    def column_3d(self) -> np.ndarray:
        return self.column.reshape((-1, *self.image_shape))

    @property
    def ra_3d(self) -> np.ndarray:
        return self.ra.reshape((-1, *self.image_shape))

    @property
    def dec_3d(self) -> np.ndarray:
        return self.dec.reshape((-1, *self.image_shape))

    @property
    def bkg_flux_3d(self) -> Optional[np.ndarray]:
        if hasattr(self, "bkg_flux") and self.bkg_flux is not None:
            return self.bkg_flux.reshape((-1, *self.image_shape))
        return None

    @property
    def bkg_flux_err_3d(self) -> Optional[np.ndarray]:
        if hasattr(self, "bkg_flux_err") and self.bkg_flux_err is not None:
            return self.bkg_flux_err.reshape((-1, *self.image_shape))
        return None

    @property
    def bkg_model_3d(self) -> Optional[np.ndarray]:
        if hasattr(self, "bkg_model") and self.bkg_model is not None:
            return self.bkg_model.reshape((-1, *self.image_shape))
        return None

    @property
    def bkg_model_err_3d(self) -> Optional[np.ndarray]:
        if hasattr(self, "bkg_model_err") and self.bkg_model_err is not None:
            return self.bkg_model_err.reshape((-1, *self.image_shape))
        return None

    @property
    def bkg_pixel_mask_3d(self) -> Optional[np.ndarray]:
        if hasattr(self, "bkg_pixel_mask") and self.bkg_pixel_mask is not None:
            return self.bkg_pixel_mask.reshape((-1, *self.image_shape))
        return None

    @staticmethod
    def from_file(
        fname: Union[str, List[str], np.ndarray],
        cutout_size: int = 32,
        cutout_center: Union[Tuple[float, float], Tuple[int, int]] = (0, 0),
        sources: Optional[pd.DataFrame] = None,
        file_version: str = "1.0",
        **kwargs,
    ) -> "LCExtractor":
        """
        Reads data from files and initiates a new object of LCExtractor class.
        Two options are available:
        1. If providing pixel coordinates with `cutout_origin`,
            the data will be fixed tothe pixel grid, no dithering
            correctin will be applied and the star field will move across the image.
        2. When providing `cutout_center` in RA, Dec coordinates,
            the data will be cetered in the target coordinate and account for
            dithering. The star field will be fixed, but the pixel grid will change.

        Parameters
        ----------
        fname : str or list of strings
            File name or list of file names of the FFI files.
        cutout_size : int, optional
            Size of the cutout in , assumed to be squared
        cutout_origin : tuple of ints
            Origin pixel coordinates where to start the cut out. The cutout will be
            centered in `cutout_origin + cutout_size / 2`. Follows matrix indexing.
        cutout_center : tuple of floats, optional
            Center of the cutout in RA, Dec coordinates. If provided, the cutout will be
            centered on this position and the pixel grid will be adjusted to account for
            dithering.
        sources : pandas.DataFrame
            Catalog with sources to be extracted by PSFMachine
        **kwargs : dictionary
            Keyword arguments that defines shape model in a `Machine` class object.
            See `psfmachine.Machine` for details.

        Returns
        -------
        LCExtractor : Machine object
            A Machine class object built from the FFI.
        """
        # check if source catalog is pandas DF
        if not isinstance(sources, pd.DataFrame):
            raise ValueError(
                "Source catalog has to be a Pandas DataFrame with columns "
                "['ra', 'dec', 'row', 'column', 'flux']"
            )

        # load FITS files and parse arrays
        (
            wcs,
            time,
            cadenceno,
            flux,
            flux_err,
            ra,
            dec,
            column,
            row,
            metadata,
            quality_mask,
        ) = _load_file(
            fname,
            cutout_size=cutout_size,
            cutout_center=cutout_center,
            file_version=file_version,
        )
        if ra.shape[0] > 1:
            dithered = True
        else:
            dithered = False

        #####
        # ra,dec and row,column are 3D arrays
        # with shape of [n_times, axis1, axis2]
        #####
        log.info("Initializing LCExtractor object...")
        return LCExtractor(
            time,
            flux,
            flux_err,
            ra,
            dec,
            sources,
            column,
            row,
            cadenceno=cadenceno,
            wcs=wcs,
            meta=metadata,
            quality_mask=quality_mask,
            dithered=dithered,
            **kwargs,
        )

    def _mask_pixels(
        self, pixel_saturation_limit: float = 6e4, magnitude_bright_limit: float = 13
    ) -> None:
        """
        Mask saturated pixels and halo/difraction pattern from bright sources.

        Parameters
        ----------
        pixel_saturation_limit: float
            Flux value at which pixels saturate.
        magnitude_bright_limit: float
            Magnitude limit for sources at which pixels are masked.
        """

        # mask saturated pixels.
        self.sat_pixel_mask = self._saturated_pixels_mask(
            saturation_limit=pixel_saturation_limit
        )
        # tolerance dependens on pixel scale
        # self.non_bright_source_mask = ~self._bright_sources_mask(
        #     magnitude_limit=magnitude_bright_limit, tolerance=10
        # )
        self.pixel_mask = ~self.sat_pixel_mask #& self.non_bright_source_mask

        # if not hasattr(self, "source_mask"):
        #     self._get_source_mask()
        #     # include saturated pixels in the source mask and uncontaminated mask
        #     self._remove_bad_pixels_from_source_mask()

        return

    def _saturated_pixels_mask(
        self, saturation_limit: float = 6e4, tolerance: int = 0
    ) -> np.ndarray:
        """
        Finds and removes saturated pixels, including bleed columns.

        Saturation limit defaul value set from:
        https://roman.gsfc.nasa.gov/science/docs/Roman_Cycle_1___Bright_Star_Saturation_and_Persistence_Effects.pdf

        Parameters
        ----------
        saturation_limit : foat
            Saturation limit at which pixels are removed.
        tolerance : int
            Number of pixels masked around the saturated pixel, remove bleeding.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
        # which pixels are saturated at every time
        sat_mask = self.flux_3d > saturation_limit

        # add nan values to the mask
        sat_mask |= ~np.isfinite(self.flux_3d)

        # dilate the mask with tolerance
        if tolerance > 0:
            struct = np.zeros((3, 3, 3), dtype=bool)
            struct[1, :, :] = True
            sat_mask = ndimage.binary_dilation(sat_mask, struct, iterations=tolerance)

        return sat_mask.reshape(self.nt, -1)

    def _bright_sources_mask(
        self, magnitude_limit: float = 13, tolerance: float = 30
    ) -> np.ndarray:
        """
        Finds and mask pixels with halos produced by bright stars (e.g. <8 mag).

        Parameters
        ----------
        magnitude_limit : foat
            Magnitude limit at which bright sources are identified.
        tolerance : float
            Radius limit (in pixels) at which pixels around bright sources are masked.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
        bright_mask = self.sources[self.sources_mag_column] <= magnitude_limit

        mask = np.array([
            np.hypot(self.ra[0] - s.ra, self.dec[0] - s.dec) < tolerance
            for _, s in self.sources[bright_mask].iterrows()
        ])
        mask = mask.sum(axis=0) > 0

        return mask

    def _pointing_offset(self) -> None:
        """
        Computes pointing offsets due to dithering
        """
        self.ra_offset = (self.ra - self.ra[0]).mean(axis=1)
        self.dec_offset = (self.dec - self.dec[0]).mean(axis=1)

    def _compute_median_background(
        self,
        sigma: float = 3.0,
        max_iters: int = 5,
    ) -> np.ndarray:
        """
        Computes the median background flux as a function of time using only
        background pixels selected via sigma clipping statistics.

        Parameters
        ----------
        sigma : float, optional
            Number of standard deviations for sigma clipping. Default is 3.0.
        max_iters : int, optional
            Maximum number of iterations for sigma clipping. Default is 5.

        Returns
        -------
        median_bkg : np.ndarray
            1D array of shape (nt,) with the median background flux for each cadence.
        """
        median_bkg = np.zeros(self.nt, dtype=np.float64)
        bkg_pixel_mask = np.zeros_like(self.flux, dtype=bool)

        for tdx in range(self.nt):
            flux_frame = self.flux[tdx]
            if hasattr(self, "pixel_mask") and self.pixel_mask is not None:
                valid = self.pixel_mask[tdx] & np.isfinite(flux_frame)
            else:
                valid = np.isfinite(flux_frame)

            if np.sum(valid) < 3:
                if np.sum(valid) > 0:
                    median_bkg[tdx] = float(np.nanmedian(flux_frame[valid]))
                else:
                    median_bkg[tdx] = 0.0
                continue

            clipped = sigma_clip(
                flux_frame[valid],
                sigma=sigma,
                maxiters=max_iters,
                cenfunc="median",
                stdfunc="mad_std",
            )
            unclipped = ~clipped.mask
            if np.sum(unclipped) > 0:
                median_bkg[tdx] = float(np.nanmedian(flux_frame[valid][unclipped]))
                valid_indices = np.where(valid)[0]
                bkg_pixel_mask[tdx, valid_indices[unclipped]] = True
            else:
                median_bkg[tdx] = float(np.nanmedian(flux_frame[valid]))
                bkg_pixel_mask[tdx, valid] = True

        self.median_bkg = median_bkg
        self.bkg_pixel_mask = bkg_pixel_mask
        return median_bkg

    def load_prf_model(
        self,
        input: Optional[str] = None,
        plot: bool = False,
    ) -> Optional[matplotlib.figure.Figure | matplotlib.axes.Axes]:
        """
        Load and process a prf model for the sources.

        Parameters
        ----------
        input : str, optional
            The path to the shape model file or other input source. If None, defaults to a predefined
            shape model location.
        plot : bool, optional, default=False
            Whether to display a diagnostic plot of the loaded shape model. If set to True, the plot
            will be shown upon loading the model.

        Returns
        -------
        None
            This function does not return any value. It modifies the internal state of the object
            by loading the shape model and potentially creating plots.
        """
        if not getattr(self, "meta", None):
            raise AttributeError(
                "self.meta is not set; populate meta before calling load_prf_model"
            )

        if not isinstance(self.meta, dict):
            raise TypeError(
                f"expected self.meta to be dict, got {type(self.meta).__name__}"
            )

        sca = self.meta.get("SCA")
        filt = self.meta.get("FILTER")
        if sca is None or filt is None:
            raise KeyError("self.meta must contain 'SCA' and 'FILTER' keys")

        # check if file exists and is the right format
        self.prf_model = RomanSimplePSF(file_name=input, sca=sca, filter=filt)

        log.info(f"Loading shape model from {input}")
        if plot:
            return self.plot_prf_model(hires=True, stretch="log")
        return None

    def pixel_coordinates(self, frame_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates pixel coordinates of sources for a given cadence.
        Supports both classic FITS WCS and APE 14 WCS (such as SlicedLowLevelWCS,
        HighLevelWCSWrapper, or GWCS). Results are memoized for performance.

        Parameters
        ----------
        frame_index : int
            Time index.

        Returns
        -------
        row, col : np.ndarray
            Row and column pixel coordinates of the sources in the catalog.
        """
        if hasattr(self, "_cached_pixel_coords") and self._cached_pixel_coords is not None:
            if frame_index in self._cached_pixel_coords:
                return self._cached_pixel_coords[frame_index]

        if self.WCSs is None:
            raise ValueError("WCS information is not available")
        wcs_obj = self.WCSs[frame_index]
        if hasattr(wcs_obj, "all_world2pix"):
            ROW, COL = wcs_obj.all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0.0).T
            res = (np.asarray(ROW), np.asarray(COL))
            if hasattr(self, "_cached_pixel_coords") and self._cached_pixel_coords is not None:
                self._cached_pixel_coords[frame_index] = res
            return res
        elif hasattr(wcs_obj, "_gwcs"):
            wcs_obj = wcs_obj._gwcs

        # APE 14 world_to_pixel interface
        coords = SkyCoord(
            ra=self.sources["ra"].values * u.deg,
            dec=self.sources["dec"].values * u.deg,
            frame="icrs",
        )
        col_vals, row_vals = wcs_obj.world_to_pixel(coords)
        res = (np.asarray(row_vals), np.asarray(col_vals))
        if hasattr(self, "_cached_pixel_coords") and self._cached_pixel_coords is not None:
            self._cached_pixel_coords[frame_index] = res
        return res

    def precompute_pixel_coordinates(self) -> None:
        """
        Precomputes and caches pixel coordinates for all cadences in the dataset.
        """
        if self.WCSs is None:
            return
        if not hasattr(self, "_cached_pixel_coords") or self._cached_pixel_coords is None:
            self._cached_pixel_coords = {}
        for tdx in range(self.nt):
            self.pixel_coordinates(frame_index=tdx)

    def draw_scene(
        self,
        frame_index: int = 0,
        offset: Tuple[float, float] = (0.0, 0.0),
        threshold: float = 1e-7,
    ) -> None:
        """
        Convenient function to make the scene model, optionally applying sub-pixel
        pointing offsets (offset_row, offset_col) in cutout pixel coordinates.
        Supports fast local bounding-box evaluation with threshold clipping.
        """
        # check of prf interpolation model exist
        if not hasattr(self, "prf_model"):
            raise AttributeError(
                "No PRF interpolation model loaded, run `self.load_prf_model` first"
            )

        # get the pixel grid for current frame
        col = self.column[frame_index]
        row = self.row[frame_index]

        # get source pixel locs for current frame
        srow, scol = self.pixel_coordinates(frame_index=frame_index)

        # Ensure source positions are in cutout pixel coordinates
        if np.nanmean(srow) >= row.min():
            srow_cutout = srow - row.min()
            scol_cutout = scol - col.min()
        else:
            srow_cutout = srow
            scol_cutout = scol

        # evaluate PRF at each source position with offset applied
        off_r, off_c = offset
        ny, nx = self.image_shape
        npix = ny * nx

        if hasattr(self.prf_model, "evaluate_local"):
            row_inds = []
            col_inds = []
            vals = []
            for sdx in range(self.nsources):
                c_row = srow_cutout[sdx] + off_r
                c_col = scol_cutout[sdx] + off_c
                r_pix, c_pix, patch_vals = self.prf_model.evaluate_local(
                    center=(c_row, c_col),
                    shape=self.image_shape,
                    corner=(0, 0),
                    threshold=threshold,
                )
                if len(patch_vals) > 0:
                    flat_pix = r_pix * nx + c_pix
                    row_inds.append(np.full(len(patch_vals), sdx, dtype=int))
                    col_inds.append(flat_pix)
                    vals.append(patch_vals)

            if len(vals) > 0:
                row_inds = np.concatenate(row_inds)
                col_inds = np.concatenate(col_inds)
                vals = np.concatenate(vals)
                self.scene_model_sparse = sparse.csr_matrix(
                    (vals, (row_inds, col_inds)),
                    shape=(self.nsources, npix),
                    dtype=float,
                )
            else:
                self.scene_model_sparse = sparse.csr_matrix(
                    (self.nsources, npix), dtype=float
                )
            self.scene_model = self.scene_model_sparse.toarray()
        else:
            scene_model = []
            for sdx in range(self.nsources):
                eval = self.prf_model.evaluate_from_position(
                    center=(srow_cutout[sdx] + off_r, scol_cutout[sdx] + off_c),
                    corner=(0, 0),
                    shape=self.image_shape,
                )[-1]
                eval = np.nan_to_num(eval, nan=0.0)
                if threshold > 0:
                    max_v = np.max(eval)
                    if max_v > 0:
                        eval[eval < threshold * max_v] = 0.0
                scene_model.append(eval.ravel())
            self.scene_model = np.array(scene_model)
            self.scene_model_sparse = sparse.csr_matrix(self.scene_model)
        return

    def plot_prf_model(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        hires: bool = False,
        stretch: str = "linear",
    ) -> Optional[matplotlib.figure.Figure | matplotlib.axes.Axes]:
        """
        Plot the Point Response Function (PRF) model for the current LCExtractor instance.

        This function visualizes the PRF either at high resolution (supersampled) or at the
        native pixel sampling, depending on the `hires` flag. The PRF is computed using the
        spline basis and current PSF weights, and displayed as a color mesh or scatter plot.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], default=None
            Matplotlib axis to plot on. If None, a new figure and axis will be created.
        hires : bool, default=False
            If True, plot a supersampled PRF model. If False, plot at the native pixel sampling.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axis containing the PRF plot.
        """

        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax_main = fig.add_subplot()
            # Create marginal Axes
            # ax_histx (top): positioned above the main axes
            ax_histx = ax_main.inset_axes(
                (0, 1.05, 1, 0.25), sharex=ax_main
            )  # [left, bottom, width, height]
            # ax_histy (right): positioned to the right of the main axes
            ax_histy = ax_main.inset_axes((1.05, 0, 0.25, 1), sharey=ax_main)

        # widen the annotated type so mypy accepts both QuadMesh and PathCollection
        cbar: Any = None

        if hires:
            dcol = np.linspace(0, self.image_shape[1], self.image_shape[1] * 6)
            drow = np.linspace(0, self.image_shape[0], self.image_shape[0] * 6)
            shape = (self.image_shape[0] * 6, self.image_shape[1] * 6)
        else:
            dcol = np.arange(0, self.image_shape[1], dtype=float)
            drow = np.arange(0, self.image_shape[0], dtype=float)
            shape = self.image_shape
        dcol, drow = np.meshgrid(dcol, drow)
        dcol -= self.image_shape[1] / 2
        drow -= self.image_shape[0] / 2

        model = self.prf_model.evaluate_from_array(dcol.ravel(), drow.ravel()).reshape(
            shape
        ).T

        cmap = simple_norm(model, stretch, percent=99.9)
        if isinstance(self.meta, dict) and "FILTER" in self.meta.keys():
            filt = self.meta["FILTER"]
        else:
            filt = ""
            
        if hires:
            ax_main.set_title(f"Super Sampled PRF | {filt}", y=1.1)
        else:
            ax_main.set_title(f"PRF in Cutout Pixel Grid | {filt}")

        cbar = ax_main.pcolormesh(
            dcol * self.pixel_scale.value,
            drow * self.pixel_scale.value,
            model,
            norm=cmap,
        )
        ax_histx.plot(dcol[0] * self.pixel_scale.value, model.mean(axis=0))
        ax_histy.plot(model.mean(axis=1), drow[:, 0] * self.pixel_scale.value)
        ax_histx.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax_histy.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        plt.colorbar(cbar, ax=ax_main, shrink=0.7, location="bottom", label="Normalized Flux")
        ax_main.set_xlabel(r"$\Delta$ Column [arcsec]")
        ax_main.set_ylabel(r"$\Delta$ Row [arcsec]")
        ax_main.set_aspect("equal")

        return ax

    def plot_image(
        self,
        data: str = "image",
        ax: Optional[matplotlib.axes.Axes] = None,
        sources: bool = False,
        frame_index: int = 0,
        coords: str = "sky",
        grid: Optional[bool] = None,
    ) -> matplotlib.axes.Axes:
        """
        Function to plot the image cutout in sky coordinates (with WCS projection)
        or in pixel space, optionally with Gaia sources.

        Parameters
        ----------
        data : str, optional
            Data plane to plot, "image" or "scene". Default is "image".
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis can be provided. If not, one will be created and returned.
        sources : boolean, optional
            Whether to overplot or not the source catalog. Default is False.
        frame_index : int, optional
            Time index used to plot the image data. Default is 0.
        coords : str, optional
            Coordinate space for plotting: "sky" (with WCS celestial projection) or "pixel".
            Default is "sky".
        grid : bool, optional
            Whether to display the coordinate grid. If None, defaults to True for "sky"
            and False for "pixel".

        Returns
        -------
        ax : matplotlib.axes.Axes
            Matplotlib axis with the figure.
        """
        coords = coords.lower()
        if coords not in ("sky", "pixel"):
            raise ValueError(f"Unknown coords '{coords}'. Options are 'sky' or 'pixel'.")

        if grid is None:
            grid = True if coords == "sky" else False

        wcs_obj = None
        if coords == "sky" and self.WCSs is not None:
            wcs_obj = self.WCSs[frame_index]
            if hasattr(wcs_obj, "_gwcs"):
                wcs_obj = wcs_obj._gwcs
            elif hasattr(wcs_obj, "wcs") and not hasattr(wcs_obj, "pixel_to_world"):
                wcs_obj = wcs_obj.wcs

            # Wrap low-level WCS to ensure longitudes are mapped to [0, 360) during WCSAxes gridline inversion
            if hasattr(wcs_obj, "low_level_wcs"):
                safe_low = _SafeWCSWrapper(wcs_obj.low_level_wcs)
                wcs_obj = HighLevelWCSWrapper(safe_low)
            elif hasattr(wcs_obj, "world_to_pixel_values"):
                wcs_obj = HighLevelWCSWrapper(_SafeWCSWrapper(wcs_obj))

        if ax is None:
            _ = plt.figure(figsize=(10, 10))
            if coords == "sky" and wcs_obj is not None:
                ax = plt.subplot(projection=wcs_obj)
            else:
                ax = plt.subplot()

        if data == "image":
            to_plot = self.flux_3d[frame_index]
            title = "Image"
        elif data == "scene":
            self.draw_scene(frame_index=frame_index)
            to_plot = self.scene_model.T.dot(self.source_flux_estimates).reshape(
                self.image_shape
            )
            title = "Scene Model"
        else:
            raise ValueError(f"Unknown data '{data}'. Options are 'image' or 'scene'.")
        norm = simple_norm(to_plot.ravel(), "asinh", percent=99)

        if coords == "sky":
            ny, nx = to_plot.shape
            row_grid, col_grid = np.meshgrid(
                np.arange(ny),
                np.arange(nx),
                indexing="ij",
            )
        else:
            col_grid = self.column_3d[frame_index]
            row_grid = self.row_3d[frame_index]

        bar = ax.pcolormesh(
            col_grid,
            row_grid,
            to_plot,
            norm=norm,
            cmap=plt.cm.viridis,
            rasterized=True,
        )
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(bar, ax=ax, shrink=0.7, label=r"Flux [DN/s]")

        if coords == "sky" and hasattr(ax, "coords"):
            if grid:
                try:
                    ax.coords.grid(True, color="w", ls="-", alpha=0.7)
                except Exception:
                    ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
            try:
                ax.coords[0].set_axislabel("Right Ascension")
                ax.coords[1].set_axislabel("Declination")
            except Exception:
                pass
        else:
            if grid:
                ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
            else:
                ax.grid(False)
            ax.set_xlabel("Column [pixel]")
            ax.set_ylabel("Row [pixel]")

        ax.set_title(
            f"Roman | SCA {self.meta['SCA']} | {self.meta['FILTER']}\n"  # type: ignore
            f"Frame {self.cadenceno[frame_index]} | JD {self.time[frame_index]} \n"  # type: ignore
            f"{title}"
        )

        if sources:
            srow, scol = self.pixel_coordinates(frame_index=frame_index)
            if coords == "pixel":
                srow = srow + self.row[frame_index].min()
                scol = scol + self.column[frame_index].min()
            # marker size correlates with source magnitude
            size = self.sources.loc[:, self.sources_mag_column].to_numpy(dtype=float)  # type: ignore
            size = np.clip(size, 15.0, 23.0)
            size = (23 - size) / (23 - 15)
            size = 10 + (50 - 10) * size
            ax.scatter(
                scol,
                srow,
                c="tab:red",
                marker="o",
                s=size,
                linewidths=0.1,
                alpha=0.8,
            )

        return ax

    def plot_pixel_masks(
        self, ax: Optional[matplotlib.axes.Axes] = None
    ) -> matplotlib.axes.Axes:
        """
        Function to plot the mask used to reject saturated and bright pixels.

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned.

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure.
        """

        if ax is None:
            _, ax = plt.subplots(1, figsize=(10, 10))
        if hasattr(self, "non_bright_source_mask"):
            ax.scatter(
                self.column_3d.ravel()[~self.non_bright_source_mask],
                self.row_3d.ravel()[~self.non_bright_source_mask],
                c="y",
                marker="s",
                s=1,
                label="bright mask",
            )
        if hasattr(self, "sat_pixel_mask"):
            ax.scatter(
                self.column_3d.ravel()[self.sat_pixel_mask],
                self.row_3d.ravel()[self.sat_pixel_mask],
                c="r",
                marker="s",
                s=1,
                label="saturated pixels",
                zorder=5000,
            )
        ax.legend(loc="best")

        ax.set_xlabel("Column Pixel Number")
        ax.set_ylabel("Row Pixel Number")
        ax.set_title("Pixel Mask")
        ax.set_xlim(self.column.min() - 5, self.column.max() + 5)
        ax.set_ylim(self.row.min() - 5, self.row.max() + 5)

        return ax

    def get_lightcurves(self, mode: str = "lk") -> dict:
        """
        Bundle light curves as `lightkurve` objects if `mode=="lk"`
        or as a DataFrame if `mode=="table"'.

        Parameters
        ----------
        mode : str
            What type of light curve will be created
        """
        if not hasattr(self, "lcs"):
            raise AttributeError("No light curve has been extracted yet, run `self.fit_prf_photometry` first")
        if mode == "lk":
            lcs = {}
            for idx, val in self.lcs.items():
                psf_lc = {f"psf_{k}": v for k, v in val["psf"].items()}
                if "psf_time" in psf_lc:
                    del psf_lc["psf_time"]

                ap_data = val.get("aperture_snr", val.get("aperture_iso", None))
                if ap_data is not None:
                    ap_meta_cols = [
                        'flfrcsap',
                        'crowdsap',
                        'centroid_x',
                        'centroid_y',
                        'bkg_flux',
                        'bkg_flux_err',
                        'flux_corr',
                        'flux_err_corr',
                        'quality',
                    ]
                    extra_cols = {f"ap_{k}": v for k, v in ap_data.items() if k in ap_meta_cols}
                    # Also include any circular apertures in data
                    for ap_name, ap_dict in val.items():
                        if ap_name.startswith("aperture_") and ap_name not in ("aperture_snr", "aperture_iso"):
                            for k, v in ap_dict.items():
                                if k not in ("time", "aperture_mask"):
                                    extra_cols[f"{ap_name}_{k}"] = v
                    time = ap_data["time"] * u.d
                    flux = ap_data["flux"] * u.electron / u.second
                    flux_err = ap_data["flux_err"] * u.electron / u.second
                else:
                    extra_cols = {}
                    time = val["psf"]["time"] * u.d
                    flux = val["psf"]["flux"] * u.electron / u.second
                    flux_err = val["psf"]["flux_err"] * u.electron / u.second

                lc = lk.LightCurve(
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    data=extra_cols | psf_lc,
                )
                lcs[idx] = lc

            return lcs
        elif mode == "table":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    def to_fits(
        self,
        target: Optional[Union[int, str]] = None,
        path: Optional[str] = None,
        overwrite: bool = True,
        extra_meta: Optional[Dict[str, Any]] = None,
        to_s3: bool = False,
        s3_fs: Optional[Any] = None,
        s3_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[fits.HDUList, Dict[Union[int, str], fits.HDUList]]:
        """
        Saves target light curve(s) to a standardized FITS file using `roman_lcs.io`.
        Supports saving to local disk or directly to an S3 bucket via `s3fs`.

        Parameters
        ----------
        target : int or str, optional
            Target index to save. If None, saves all extracted targets (or writes
            multiple files if path contains a formatting placeholder).
        path : str, optional
            Path or S3 key where the FITS file will be written.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is True.
        extra_meta : dict, optional
            Additional metadata cards to include in PRIMARY header.
        to_s3 : bool, optional
            Whether to write the output directly to an S3 bucket using `s3fs`. Default is False.
        s3_fs : s3fs.S3FileSystem or fsspec filesystem, optional
            Pre-configured S3/fsspec filesystem instance.
        s3_kwargs : dict, optional
            Keyword arguments passed to `s3fs.S3FileSystem(...)`.
        **kwargs : Any
            Additional metadata keyword arguments for PRIMARY header.
            Accepts `bucket_name` (str) as the mandatory AWS S3 bucket name when writing to S3.

        Returns
        -------
        hdul : astropy.io.fits.HDUList or dict of HDULists
        """
        if not hasattr(self, "lcs") or len(self.lcs) == 0:
            raise AttributeError("No light curves extracted yet. Run `fit_prf_photometry` first.")

        bucket_name = kwargs.pop("bucket_name", None)

        from .io import write_lightcurve_fits

        # Build metadata from self.meta and extractor attributes
        base_meta = dict(self.meta) if hasattr(self, "meta") and self.meta else {}
        if extra_meta is not None:
            base_meta.update(extra_meta)
        base_meta.update(kwargs)
        if hasattr(self, "cutout_size"):
            base_meta["CUTSIZE"] = self.cutout_size
        elif hasattr(self, "image_shape"):
            base_meta["CUTSIZE"] = self.image_shape[0]

        if target is None:
            if len(self.lcs) == 1:
                target = list(self.lcs.keys())[0]
            else:
                out = {}
                for t in self.lcs.keys():
                    t_meta = dict(base_meta)
                    if hasattr(self, "sources") and t in self.sources.index:
                        if "roman_source_id" in self.sources.columns:
                            t_meta["TARGETID"] = self.sources.loc[t, "roman_source_id"]
                        elif "TARGETID" not in t_meta:
                            t_meta["TARGETID"] = t
                        if "ra" in self.sources.columns:
                            t_meta["RA_OBJ"] = self.sources.loc[t, "ra"]
                        if "dec" in self.sources.columns:
                            t_meta["DEC_OBJ"] = self.sources.loc[t, "dec"]
                    p = path.format(target=t) if path and "{target}" in path else path
                    out[t] = write_lightcurve_fits(
                        self.lcs[t],
                        output_path=p,
                        meta=t_meta,
                        cadenceno=self.cadenceno if hasattr(self, "cadenceno") else None,
                        overwrite=overwrite,
                        to_s3=to_s3,
                        bucket_name=bucket_name,
                        s3_fs=s3_fs,
                        s3_kwargs=s3_kwargs,
                    )
                return out

        target_lcs = self.lcs[target]
        if hasattr(self, "sources") and target in self.sources.index:
            if "roman_source_id" in self.sources.columns:
                base_meta["TARGETID"] = self.sources.loc[target, "roman_source_id"]
            elif "TARGETID" not in base_meta:
                base_meta["TARGETID"] = target
            if "ra" in self.sources.columns:
                base_meta["RA_OBJ"] = self.sources.loc[target, "ra"]
            if "dec" in self.sources.columns:
                base_meta["DEC_OBJ"] = self.sources.loc[target, "dec"]

        return write_lightcurve_fits(
            target_lcs,
            output_path=path,
            meta=base_meta,
            cadenceno=self.cadenceno if hasattr(self, "cadenceno") else None,
            overwrite=overwrite,
            to_s3=to_s3,
            bucket_name=bucket_name,
            s3_fs=s3_fs,
            s3_kwargs=s3_kwargs,
        )

    def fit_prf_photometry(
        self,
        targets: List[int] = [],
        model_bkg: bool = True,
        bkg_nknots: int = 3,
        bkg_sigma: float = 3.0,
        do_aperture: bool = True,
        aperture_radius: Union[float, int, Sequence[Union[float, int]]] = 3,
        fit_offsets: bool = False,
        offset_max_iter: int = 5,
        offset_threshold: float = 0.01,
        solver_method: str = "cholesky",
    ) -> None:
        """
        Fits PRF photometry the given targets in the image accounting for backgronund
        stars and signal.

        Parameters
        ----------
        targets : list of int, optional
            List of target names to fit PRF photometry for. If None or empty, all sources in
            the image will be used.
        model_bkg : bool, optional
            Whether to model the background signal. Default is True.
        bkg_nknots : int, optional
            Number of spline knots for background modeling. Default is 3.
        bkg_sigma : float, optional
            Sigma clipping threshold for computing the median background. Default is 3.0.
        do_aperture : bool, optional
            Whether to compute aperture photometry. Default is True.
        aperture_radius : float, int, or sequence of numbers, optional
            Radius or list of radii for circular aperture photometry in pixels.
            Default is 3.
        fit_offsets : bool, optional
            Whether to iteratively fit sub-pixel centroid offsets (drow, dcol) using
            scene model spatial gradients. Default is False.
        offset_max_iter : int, optional
            Maximum number of Gauss-Newton iterations for offset fitting. Default is 5.
        offset_threshold : float, optional
            Convergence threshold in pixels for incremental sub-pixel shifts.
            Default is 0.01 pix (~1.1 mas for Roman WFI).
        """
        if targets is None or len(targets) == 0:
            targets = list(range(len(self.sources)))
        n_targets = len(targets)

        if isinstance(aperture_radius, (int, float, np.integer, np.floating)):
            radii = [aperture_radius]
        elif isinstance(aperture_radius, (list, tuple, np.ndarray)):
            radii = list(aperture_radius)
        else:
            raise TypeError(
                f"`aperture_radius` must be a float, int, or list of numbers, got {type(aperture_radius)}"
            )

        targets_prf_flux = np.zeros((self.nt, n_targets))
        targets_prf_flux_err = np.zeros((self.nt, n_targets))
        model_chi2 = np.zeros(self.nt)
        scene_model = np.zeros_like(self.flux)
        psf_quality = np.zeros(self.nt)

        self.prf_drow = np.zeros(self.nt, dtype=np.float64)
        self.prf_dcol = np.zeros(self.nt, dtype=np.float64)
        self.prf_fit_flags = np.zeros(self.nt, dtype=np.int32)
        targets_nominal_row = np.zeros((self.nt, n_targets), dtype=np.float64)
        targets_nominal_col = np.zeros((self.nt, n_targets), dtype=np.float64)

        if model_bkg:
            median_bkg = self._compute_median_background(sigma=bkg_sigma)
            bkg_terms = _make_A_cartesian(
                x=self.dcolumn[targets[0] if len(targets) > 0 else 0].ravel(), # type: ignore
                y=self.drow[targets[0] if len(targets) > 0 else 0].ravel(), # type: ignore
                n_knots=bkg_nknots,
            ).T
            n_bkg_terms = bkg_terms.shape[0]
            self.bkg_flux = np.zeros_like(self.flux)
            self.bkg_flux_err = np.zeros_like(self.flux)
        else:
            median_bkg = np.zeros(self.nt, dtype=np.float64)
            self.bkg_flux = None
            self.bkg_flux_err = None

        has_bkg_components = model_bkg or (len(targets) > 0 and len(self.sources) > len(targets))
        if has_bkg_components:
            self.bkg_model = np.zeros_like(self.flux)
            self.bkg_model_err = np.zeros_like(self.flux)
        else:
            self.bkg_model = None
            self.bkg_model_err = None

        if do_aperture:
            snr_ap_flux = np.zeros((self.nt, n_targets))
            snr_ap_flux_err = np.zeros((self.nt, n_targets))
            snr_ap_metrics = np.zeros((self.nt, n_targets, 2))
            snr_centroid = np.zeros((self.nt, n_targets, 2))
            snr_bkg_flux = np.zeros((self.nt, n_targets))
            snr_bkg_flux_err = np.zeros((self.nt, n_targets))
            snr_ap_quality = np.zeros((self.nt, n_targets))
            snr_masks = np.zeros((self.nt, n_targets, np.prod(self.image_shape)), dtype=bool)

            iso_ap_flux = np.zeros((self.nt, n_targets))
            iso_ap_flux_err = np.zeros((self.nt, n_targets))
            iso_ap_metrics = np.zeros((self.nt, n_targets, 2))
            iso_centroid = np.zeros((self.nt, n_targets, 2))
            iso_bkg_flux = np.zeros((self.nt, n_targets))
            iso_bkg_flux_err = np.zeros((self.nt, n_targets))
            iso_ap_quality = np.zeros((self.nt, n_targets))
            iso_masks = np.zeros((self.nt, n_targets, np.prod(self.image_shape)), dtype=bool)

            circ_data = {}
            for r in radii:
                r_key = _format_aperture_key(r)
                circ_data[r_key] = {
                    "radius": float(r),
                    "flux": np.zeros((self.nt, n_targets)),
                    "flux_err": np.zeros((self.nt, n_targets)),
                    "metrics": np.zeros((self.nt, n_targets, 2)),
                    "centroid": np.zeros((self.nt, n_targets, 2)),
                    "bkg_flux": np.zeros((self.nt, n_targets)),
                    "bkg_flux_err": np.zeros((self.nt, n_targets)),
                    "quality": np.zeros((self.nt, n_targets)),
                    "masks": np.zeros((self.nt, n_targets, np.prod(self.image_shape)), dtype=bool),
                }

        self.precompute_pixel_coordinates()
        self.draw_scene(frame_index=0)
        self._compute_aperture_cut()
        if do_aperture:
            self._compute_snr_optimal_pixel_count()

        for tdx in tqdm(
            range(self.nt),
            desc="Fitting PRF photometry",
            total=self.nt,
            disable=self.quiet,
        ):
            # Record nominal positions of targets for this frame in original image pixel grid
            srow_nom, scol_nom = self.pixel_coordinates(frame_index=tdx)
            corner_row = self.row[tdx].min()
            corner_col = self.column[tdx].min()
            if np.nanmean(srow_nom) < corner_row:
                row_orig = srow_nom + corner_row
                col_orig = scol_nom + corner_col
            else:
                row_orig = srow_nom
                col_orig = scol_nom
            targets_nominal_row[tdx, :] = row_orig[targets]
            targets_nominal_col[tdx, :] = col_orig[targets]

            epoch_flag = PRFFitFlag.SUCCESS
            if np.sum(self.pixel_mask[tdx]) < 10:
                epoch_flag |= PRFFitFlag.INSUFFICIENT_UNMASKED_PIXELS

            drow_acc = 0.0
            dcol_acc = 0.0

            if fit_offsets:
                converged = False
                last_w = None
                for it in range(offset_max_iter):
                    self.draw_scene(frame_index=tdx, offset=(drow_acc, dcol_acc))
                    mean_model = getattr(self, "scene_model_sparse", sparse.csr_matrix(self.scene_model))
                    if len(targets) > 0:
                        targets_models = mean_model[targets]
                        bkg_star_indices = ~np.isin(self.sources.index.values, targets)
                        if bkg_star_indices.any():
                            bkg_star_model = mean_model[bkg_star_indices]
                            bkg_star_flux = np.delete(self.source_flux_estimates, targets)
                            bkg_star_scene = bkg_star_model.T.dot(bkg_star_flux)[None, :]
                            stellar_models = sparse.vstack([targets_models, sparse.csr_matrix(bkg_star_scene)]).tocsr()
                        else:
                            stellar_models = targets_models
                    else:
                        stellar_models = mean_model

                    # Get current estimate of stellar fluxes
                    if last_w is None:
                        init_m = stellar_models
                        if model_bkg:
                            init_m = sparse.vstack([init_m, bkg_terms]).tocsr()
                        try:
                            w_init = solve_linear_model(
                                init_m.T,
                                y=self.flux[tdx] - median_bkg[tdx],
                                y_err=self.flux_err[tdx],
                                errors=False,
                                k=self.pixel_mask[tdx],
                                method=solver_method,
                            )
                            w_stellar = w_init[:stellar_models.shape[0]]
                        except Exception:
                            w_stellar = np.array([self.source_flux_estimates[t] for t in targets])
                            if len(targets) > 0 and bkg_star_indices.any():
                                w_stellar = np.append(w_stellar, 1.0)
                    else:
                        w_stellar = last_w[:stellar_models.shape[0]]

                    stellar_scene_1d = stellar_models.T.dot(w_stellar).ravel()
                    stellar_scene_2d = stellar_scene_1d.reshape(self.image_shape)
                    dM_dr, dM_dc = np.gradient(stellar_scene_2d)
                    G_row = -dM_dr.ravel()
                    G_col = -dM_dc.ravel()

                    grad_models = sparse.csr_matrix(np.vstack([G_row, G_col]))
                    iter_model = sparse.vstack([stellar_models, grad_models]).tocsr()
                    idx_Grow = stellar_models.shape[0]
                    idx_Gcol = idx_Grow + 1
                    if model_bkg:
                        iter_model = sparse.vstack([iter_model, bkg_terms]).tocsr()

                    try:
                        w, werr = solve_linear_model(
                            iter_model.T,
                            y=self.flux[tdx] - median_bkg[tdx],
                            y_err=self.flux_err[tdx],
                            errors=True,
                            k=self.pixel_mask[tdx],
                            method=solver_method,
                        )
                        if not np.all(np.isfinite(w)):
                            epoch_flag |= PRFFitFlag.NON_FINITE_WEIGHTS
                            break
                    except np.linalg.LinAlgError as e:
                        log.error(f"Error solving linear model with offsets: {e}")
                        epoch_flag |= PRFFitFlag.LINALG_FAILED
                        break

                    delta_r = float(w[idx_Grow])
                    delta_c = float(w[idx_Gcol])

                    if max(abs(delta_r), abs(delta_c)) > 1.5 or max(abs(drow_acc + delta_r), abs(dcol_acc + delta_c)) > 2.0:
                        epoch_flag |= PRFFitFlag.DIVERGED_OFFSET

                    step_r = np.clip(delta_r, -0.5, 0.5)
                    step_c = np.clip(delta_c, -0.5, 0.5)
                    drow_acc += step_r
                    dcol_acc += step_c

                    last_w = w
                    last_werr = werr

                    if max(abs(delta_r), abs(delta_c)) < offset_threshold:
                        converged = True
                        break

                if not converged:
                    epoch_flag |= PRFFitFlag.NOT_CONVERGED
                    if it == offset_max_iter - 1:
                        epoch_flag |= PRFFitFlag.MAX_ITER_REACHED

                # Draw scene at final converged offset
                self.draw_scene(frame_index=tdx, offset=(drow_acc, dcol_acc))
            else:
                # update scene model due to pixel grid offsets
                self.draw_scene(frame_index=tdx)

            mean_model = getattr(self, "scene_model_sparse", sparse.csr_matrix(self.scene_model))
            if len(targets) > 0:
                targets_models = mean_model[targets]
                # get background stars PSF model
                bkg_star_indices = ~np.isin(self.sources.index.values, targets)
                if bkg_star_indices.any():
                    bkg_star_model = mean_model[bkg_star_indices]
                    bkg_star_flux = np.delete(self.source_flux_estimates, targets)
                    # compute background star scene
                    bkg_star_model = bkg_star_model.T.dot(bkg_star_flux)[None, :]
                    bkg_star_model = sparse.csr_matrix(bkg_star_model)
                    # stack linear model with target, bkg stars and bkg signal
                    model = sparse.vstack([targets_models, bkg_star_model]).tocsr()
                else:
                    model = targets_models
            else:
                model = mean_model
            if model_bkg:
                model = sparse.vstack([model, bkg_terms]).tocsr()

            # solve linear model with current flux (with median background subtracted)
            try:
                w, werr = solve_linear_model(
                    model.T,
                    y=self.flux[tdx] - median_bkg[tdx],
                    y_err=self.flux_err[tdx],
                    errors=True,
                    k=self.pixel_mask[tdx],
                    method=solver_method,
                )
                if not np.all(np.isfinite(w)):
                    epoch_flag |= PRFFitFlag.NON_FINITE_WEIGHTS
                targets_prf_flux[tdx, :] = w[:n_targets]
                targets_prf_flux_err[tdx, :] = werr[:n_targets]
                scene_model[tdx] = model.T.dot(w).ravel() + median_bkg[tdx]
                if model_bkg:
                    w_bkg = w[-n_bkg_terms:]
                    werr_bkg = werr[-n_bkg_terms:]
                    # Strictly evaluate bkg_terms basis (excluding background stars):
                    bkg_f = bkg_terms.T.dot(w_bkg).ravel()
                    bkg_var = (bkg_terms.power(2)).T.dot(werr_bkg**2).ravel()
                    total_bkg_f = bkg_f + median_bkg[tdx]
                    poisson_var = np.clip(total_bkg_f, 0, None)
                    bkg_var = np.where(
                        np.isfinite(bkg_var) & (bkg_var >= 0), bkg_var, poisson_var
                    )
                    bkg_err = np.sqrt(bkg_var)

                    self.bkg_flux[tdx] = total_bkg_f
                    self.bkg_flux_err[tdx] = bkg_err

                if self.bkg_model is not None and model.shape[0] > n_targets:
                    # Account for diffuse background signal plus background stars (everything but targets)
                    non_target_model = model[n_targets:]
                    w_non_target = w[n_targets:]
                    werr_non_target = werr[n_targets:]
                    bkg_mod_f = non_target_model.T.dot(w_non_target).ravel()
                    total_bkg_mod_f = bkg_mod_f + median_bkg[tdx]
                    bkg_mod_var = (non_target_model.power(2)).T.dot(werr_non_target**2).ravel()
                    poisson_mod_var = np.clip(total_bkg_mod_f, 0, None)
                    bkg_mod_var = np.where(
                        np.isfinite(bkg_mod_var) & (bkg_mod_var >= 0),
                        bkg_mod_var,
                        poisson_mod_var,
                    )
                    bkg_mod_err = np.sqrt(bkg_mod_var)

                    self.bkg_model[tdx] = total_bkg_mod_f
                    self.bkg_model_err[tdx] = bkg_mod_err
            except np.linalg.LinAlgError as e:
                log.error(f"Error solving linear model: {e}")
                log.error("Skipping PRF fit for this cadence.")
                epoch_flag |= PRFFitFlag.LINALG_FAILED
                targets_prf_flux[tdx, :] = -1e6
                targets_prf_flux_err[tdx, :] = -1e6

            # Check for negative flux in targets
            if np.any(targets_prf_flux[tdx, :] < 0):
                epoch_flag |= PRFFitFlag.NEGATIVE_FLUX

            # reduced chi2
            if tdx == 0:
                model_rank = np.linalg.matrix_rank(model.toarray())
            model_chi2[tdx] = ((self.flux[tdx] - scene_model[tdx]) ** 2).sum() / (
                np.prod(self.image_shape) - model_rank
            )
            if model_chi2[tdx] > 10.0:
                epoch_flag |= PRFFitFlag.HIGH_CHI2

            # quality mask with sat and bad pixels
            psf_quality[tdx] = (~self.pixel_mask[tdx]).sum()

            self.prf_drow[tdx] = drow_acc
            self.prf_dcol[tdx] = dcol_acc
            self.prf_fit_flags[tdx] = epoch_flag

            # do aperture photometry for targets
            if do_aperture:
                # 1. SNR optimal aperture (default)
                ap_flux, snr_mask = self._single_frame_aperture_photometry(
                    frame_index=tdx,
                    targets=targets,
                    aperture_type="snr",
                )
                snr_ap_flux[tdx] = ap_flux[:, 0]
                snr_ap_flux_err[tdx] = ap_flux[:, 1]
                snr_ap_metrics[tdx] = ap_flux[:, 2:4]
                snr_centroid[tdx] = ap_flux[:, 4:6]
                snr_bkg_flux[tdx] = ap_flux[:, 6]
                snr_bkg_flux_err[tdx] = ap_flux[:, 7]
                snr_ap_quality[tdx] = ap_flux[:, 8]
                snr_masks[tdx] = snr_mask

                # 2. Isophote aperture (uses self.prf_cut)
                iso_flux, iso_mask = self._single_frame_aperture_photometry(
                    frame_index=tdx,
                    targets=targets,
                    aperture_type="iso",
                )
                iso_ap_flux[tdx] = iso_flux[:, 0]
                iso_ap_flux_err[tdx] = iso_flux[:, 1]
                iso_ap_metrics[tdx] = iso_flux[:, 2:4]
                iso_centroid[tdx] = iso_flux[:, 4:6]
                iso_bkg_flux[tdx] = iso_flux[:, 6]
                iso_bkg_flux_err[tdx] = iso_flux[:, 7]
                iso_ap_quality[tdx] = iso_flux[:, 8]
                iso_masks[tdx] = iso_mask

                # 3. Circular apertures
                for r_key, cdict in circ_data.items():
                    c_flux, c_mask = self._single_frame_aperture_photometry(
                        frame_index=tdx,
                        targets=targets,
                        aperture_type="circular",
                        radius=cdict["radius"],
                    )
                    cdict["flux"][tdx] = c_flux[:, 0]
                    cdict["flux_err"][tdx] = c_flux[:, 1]
                    cdict["metrics"][tdx] = c_flux[:, 2:4]
                    cdict["centroid"][tdx] = c_flux[:, 4:6]
                    cdict["bkg_flux"][tdx] = c_flux[:, 6]
                    cdict["bkg_flux_err"][tdx] = c_flux[:, 7]
                    cdict["quality"][tdx] = c_flux[:, 8]
                    cdict["masks"][tdx] = c_mask

        self.scene_model_cube = scene_model

        if not hasattr(self, "lcs"):
            self.lcs = {}
        for sdx, snum in enumerate(targets):
            if snum not in self.lcs:
                self.lcs[snum] = {}
            self.lcs[snum]["psf"] = {
                "time": self.time,
                "flux": targets_prf_flux[:, sdx],
                "flux_err": targets_prf_flux_err[:, sdx],
                "chi2": model_chi2,
                "quality": psf_quality,
                "drow": self.prf_drow,
                "dcol": self.prf_dcol,
                "row": targets_nominal_row[:, sdx],
                "column": targets_nominal_col[:, sdx],
                "fit_flag": self.prf_fit_flags,
            }

        if do_aperture:
            snr_masks_arr = snr_masks
            iso_masks_arr = iso_masks
            self.aperture_masks = snr_masks_arr
            for sdx, snum in enumerate(targets):
                snr_flux_corr, snr_flux_err_corr = correct_aperture_flux(
                    flux=snr_ap_flux[:, sdx],
                    flux_err=snr_ap_flux_err[:, sdx],
                    flfrcsap=snr_ap_metrics[:, sdx, 0],
                    crowdsap=snr_ap_metrics[:, sdx, 1],
                )
                snr_dict = {
                    "time": self.time,
                    "flux": snr_ap_flux[:, sdx],
                    "flux_err": snr_ap_flux_err[:, sdx],
                    "flux_corr": snr_flux_corr,
                    "flux_err_corr": snr_flux_err_corr,
                    "flfrcsap": snr_ap_metrics[:, sdx, 0],
                    "crowdsap": snr_ap_metrics[:, sdx, 1],
                    "centroid_x": targets_nominal_col[:, sdx] + snr_centroid[:, sdx, 0],
                    "centroid_y": targets_nominal_row[:, sdx] + snr_centroid[:, sdx, 1],
                    "bkg_flux": snr_bkg_flux[:, sdx],
                    "bkg_flux_err": snr_bkg_flux_err[:, sdx],
                    "aperture_mask": snr_masks_arr[:, sdx, :],
                    "quality": snr_ap_quality[:, sdx],
                }
                iso_flux_corr, iso_flux_err_corr = correct_aperture_flux(
                    flux=iso_ap_flux[:, sdx],
                    flux_err=iso_ap_flux_err[:, sdx],
                    flfrcsap=iso_ap_metrics[:, sdx, 0],
                    crowdsap=iso_ap_metrics[:, sdx, 1],
                )
                iso_dict = {
                    "time": self.time,
                    "flux": iso_ap_flux[:, sdx],
                    "flux_err": iso_ap_flux_err[:, sdx],
                    "flux_corr": iso_flux_corr,
                    "flux_err_corr": iso_flux_err_corr,
                    "flfrcsap": iso_ap_metrics[:, sdx, 0],
                    "crowdsap": iso_ap_metrics[:, sdx, 1],
                    "centroid_x": targets_nominal_col[:, sdx] + iso_centroid[:, sdx, 0],
                    "centroid_y": targets_nominal_row[:, sdx] + iso_centroid[:, sdx, 1],
                    "bkg_flux": iso_bkg_flux[:, sdx],
                    "bkg_flux_err": iso_bkg_flux_err[:, sdx],
                    "aperture_mask": iso_masks_arr[:, sdx, :],
                    "quality": iso_ap_quality[:, sdx],
                }
                self.lcs[snum]["aperture_snr"] = snr_dict
                self.lcs[snum]["aperture_iso"] = iso_dict

                for r_key, cdict in circ_data.items():
                    cmasks_arr = cdict["masks"]
                    c_flux_corr, c_flux_err_corr = correct_aperture_flux(
                        flux=cdict["flux"][:, sdx],
                        flux_err=cdict["flux_err"][:, sdx],
                        flfrcsap=cdict["metrics"][:, sdx, 0],
                        crowdsap=cdict["metrics"][:, sdx, 1],
                    )
                    self.lcs[snum][r_key] = {
                        "time": self.time,
                        "flux": cdict["flux"][:, sdx],
                        "flux_err": cdict["flux_err"][:, sdx],
                        "flux_corr": c_flux_corr,
                        "flux_err_corr": c_flux_err_corr,
                        "flfrcsap": cdict["metrics"][:, sdx, 0],
                        "crowdsap": cdict["metrics"][:, sdx, 1],
                        "centroid_x": targets_nominal_col[:, sdx] + cdict["centroid"][:, sdx, 0],
                        "centroid_y": targets_nominal_row[:, sdx] + cdict["centroid"][:, sdx, 1],
                        "bkg_flux": cdict["bkg_flux"][:, sdx],
                        "bkg_flux_err": cdict["bkg_flux_err"][:, sdx],
                        "aperture_mask": cmasks_arr[:, sdx, :],
                        "quality": cdict["quality"][:, sdx],
                    }

        return

    def _compute_aperture_cut(self, percentile_cut: Optional[np.ndarray] = None):
        """
        Computes the cut for aperture photometry based on the enclosed flux
        percentile (default 90%) of the PRF scene model.
        """
        if percentile_cut is None:
            percentile_cut = np.array([90] * self.nsources)
        elif isinstance(percentile_cut, (int, float, np.integer, np.floating)):
            percentile_cut = np.array([percentile_cut] * self.nsources)

        cuts = np.zeros(self.nsources, dtype=float)
        for s_idx, (obj, per) in enumerate(zip(self.scene_model, percentile_cut)):
            if hasattr(obj, "toarray"):
                flat = obj.toarray().ravel()
            else:
                flat = np.asarray(obj).ravel()
            valid = np.isfinite(flat) & (flat > 0)
            if not np.any(valid):
                cuts[s_idx] = 0.0
                continue
            sorted_vals = np.sort(flat[valid])[::-1]
            cumsum = np.cumsum(sorted_vals)
            frac = per / 100.0 if per > 1.0 else per
            target = frac * cumsum[-1]
            idx = min(np.searchsorted(cumsum, target), len(sorted_vals) - 1)
            cuts[s_idx] = sorted_vals[idx]
        self.prf_cut = cuts
        return

    def _compute_snr_optimal_pixel_count(self) -> np.ndarray:
        """
        Precomputes the optimal pixel count N_opt for each source based on the
        median noise across all epochs and the reference frame scene model.
        """
        median_error = np.nanmedian(self.flux_err, axis=0)
        sat_limit = getattr(self, "pixel_saturation_limit", 6e4)
        pixel_counts = np.zeros(self.nsources, dtype=int)
        for sdx in range(self.nsources):
            flux_est = (
                float(self.source_flux_estimates[sdx])
                if hasattr(self, "source_flux_estimates")
                and np.isfinite(self.source_flux_estimates[sdx])
                and self.source_flux_estimates[sdx] > 0
                else 1.0
            )
            signal = self.scene_model[sdx] * flux_est
            mask = make_snr_optimal_aperture(
                signal=signal,
                error=median_error,
                sat_flux=sat_limit,
                plot=False,
            )
            pixel_counts[sdx] = int(mask.sum())
        self.snr_optimal_pixel_count = pixel_counts
        return self.snr_optimal_pixel_count

    def _compute_circular_aperture_mask(
        self, frame_index: int = 0, radius: float = 3.0
    ) -> np.ndarray:
        """
        Computes circular aperture boolean mask of given radius for all sources.

        Parameters
        ----------
        frame_index : int
            Time index.
        radius : float
            Aperture radius in pixels.

        Returns
        -------
        mask : np.ndarray
            Boolean array of shape [nsources, npixels].
        """
        col = self.column[frame_index]
        row = self.row[frame_index]
        srow, scol = self.pixel_coordinates(frame_index=frame_index)

        if np.nanmean(srow) >= row.min():
            srow_cutout = srow - row.min()
            scol_cutout = scol - col.min()
        else:
            srow_cutout = srow
            scol_cutout = scol

        col_cutout = col - col.min()
        row_cutout = row - row.min()

        # Distance from each pixel to each source: shape (nsources, npixels)
        dx = col_cutout[None, :] - scol_cutout[:, None]
        dy = row_cutout[None, :] - srow_cutout[:, None]
        dist = np.hypot(dx, dy)

        return dist <= radius

    def _compute_snr_optimal_aperture_mask(
        self, frame_index: int = 0
    ) -> np.ndarray:
        """
        Computes SNR optimal aperture boolean mask for all sources at frame_index
        using a precomputed constant pixel count per source (tracking the top-N
        pixels of each source's scene model).

        Parameters
        ----------
        frame_index : int
            Time index.

        Returns
        -------
        mask : np.ndarray
            Boolean array of shape [nsources, npixels].
        """
        if not hasattr(self, "snr_optimal_pixel_count"):
            self._compute_snr_optimal_pixel_count()

        masks = np.zeros((self.nsources, self.scene_model.shape[1]), dtype=bool)
        for sdx in range(self.nsources):
            n_opt = int(self.snr_optimal_pixel_count[sdx])
            flat_sig = np.asarray(self.scene_model[sdx]).ravel()
            n_opt = max(1, min(len(flat_sig), n_opt))
            top_indices = np.argpartition(-flat_sig, n_opt - 1)[:n_opt]
            masks[sdx, top_indices] = True
        return masks

    def _measure_aperture_photometry(
        self,
        frame_index: int,
        targets: List[int],
        aperture_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measures flux and aperture diagnostics for targets using the given aperture mask.
        Vectorized with BLAS matrix products for speed.
        """
        if len(targets) == 0:
            return np.empty((0, 9)), np.empty((0, aperture_mask.shape[1]), dtype=bool)

        # combine aperture mask with pixel mask to remove bad pixels from aperture photometry
        if hasattr(self, "pixel_mask"):
            _aperture_mask = aperture_mask & self.pixel_mask[frame_index][None, :]
        else:
            _aperture_mask = aperture_mask

        # check for background model on self
        if hasattr(self, "bkg_flux") and self.bkg_flux is not None:
            bkg = self.bkg_flux[frame_index]
            bkg_err = (
                self.bkg_flux_err[frame_index]
                if hasattr(self, "bkg_flux_err") and self.bkg_flux_err is not None
                else None
            )
            flux_sub = self.flux[frame_index] - bkg
        else:
            bkg = None
            bkg_err = None
            flux_sub = self.flux[frame_index]

        _aperture_mask_tar = _aperture_mask[targets].astype(float)

        # sum flux, flux error, and background in the aperture (vectorized)
        f_vals = _aperture_mask_tar.dot(flux_sub)
        if bkg_err is not None:
            var_pix = (self.flux_err[frame_index] ** 2) + (bkg_err ** 2)
            err_vals = np.sqrt(_aperture_mask_tar.dot(var_pix))
        else:
            err_vals = np.sqrt(_aperture_mask_tar.dot(self.flux_err[frame_index] ** 2))
        targets_ap_flux = np.column_stack([f_vals, err_vals])

        if bkg is not None:
            b_vals = _aperture_mask_tar.dot(bkg)
            b_err_vals = (
                np.sqrt(_aperture_mask_tar.dot(bkg_err ** 2))
                if bkg_err is not None
                else np.zeros(len(targets))
            )
        else:
            b_vals = np.zeros(len(targets))
            b_err_vals = np.zeros(len(targets))
        targets_bkg = np.column_stack([b_vals, b_err_vals])

        # compute aperture metrics
        flux_weights = (
            np.asarray(self.source_flux_estimates, dtype=float)
            if hasattr(self, "source_flux_estimates") and self.source_flux_estimates is not None
            else None
        )
        scene_mod = getattr(self, "scene_model_sparse", self.scene_model)
        flux_metrics = np.column_stack(
            [
                compute_FLFRCSAP(scene_mod, _aperture_mask)[targets],
                compute_CROWDSAP(
                    scene_mod,
                    _aperture_mask,
                    flux_weights=flux_weights,
                )[targets],
            ]
        )

        # compute centroids using flux weighted average of pixel coordinates in the aperture (vectorized)
        col_frame = self.column[frame_index]
        row_frame = self.row[frame_index]
        srow_nom, scol_nom = self.pixel_coordinates(frame_index=frame_index)
        corner_row = row_frame.min()
        corner_col = col_frame.min()
        if np.nanmean(srow_nom) < corner_row:
            row_orig = srow_nom[targets] + corner_row
            col_orig = scol_nom[targets] + corner_col
        else:
            row_orig = srow_nom[targets]
            col_orig = scol_nom[targets]

        w_pix = np.clip(flux_sub, 0, None)
        W = _aperture_mask_tar * w_pix[None, :]
        w_sum = W.sum(axis=1)
        mask_sum = _aperture_mask_tar.sum(axis=1)

        dc = np.full(len(targets), np.nan)
        dr = np.full(len(targets), np.nan)

        valid_w = w_sum > 0
        if np.any(valid_w):
            dc[valid_w] = (W[valid_w].dot(col_frame) / w_sum[valid_w]) - col_orig[valid_w]
            dr[valid_w] = (W[valid_w].dot(row_frame) / w_sum[valid_w]) - row_orig[valid_w]

        unweighted = (~valid_w) & (mask_sum > 0)
        if np.any(unweighted):
            dc[unweighted] = (_aperture_mask_tar[unweighted].dot(col_frame) / mask_sum[unweighted]) - col_orig[unweighted]
            dr[unweighted] = (_aperture_mask_tar[unweighted].dot(row_frame) / mask_sum[unweighted]) - row_orig[unweighted]

        targets_centroid = np.column_stack([dc, dr])

        if hasattr(self, "pixel_mask"):
            bad_pix = (~self.pixel_mask[frame_index]).astype(float)
            quality = aperture_mask[targets].astype(float).dot(bad_pix)
        else:
            quality = np.zeros(len(targets))

        return (
            np.column_stack([targets_ap_flux, flux_metrics, targets_centroid, targets_bkg, quality]),
            aperture_mask[targets],
        )

    def _single_frame_aperture_photometry(
        self,
        frame_index: int = 0,
        targets: List[int] = [],
        aperture_type: str = "snr",
        radius: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs aperture photometry for a single frame and given targets in the image.

        Parameters
        ----------
        frame_index : int
            Time index.
        targets : list of int
            List of target indices.
        aperture_type : str, optional
            Aperture type: 'snr' (optimal SNR mask), 'circular' (fixed radius),
            or 'isophote'/'iso' (scene PRF isophote cut). Default is 'snr'.
        radius : float, optional
            Radius in pixels if aperture_type is 'circular'. Default is 3.0.
        """
        if aperture_type == "circular":
            aperture_mask = self._compute_circular_aperture_mask(
                frame_index=frame_index, radius=radius
            )
        elif aperture_type in ("isophote", "iso"):
            aperture_mask = np.array(self.scene_model >= self.prf_cut[::, None])
        else:
            aperture_mask = self._compute_snr_optimal_aperture_mask(
                frame_index=frame_index
            )

        return self._measure_aperture_photometry(
            frame_index=frame_index,
            targets=targets,
            aperture_mask=aperture_mask,
        )

    def do_aperture_photometry(
        self,
        targets: List[int] = [],
        aperture_radius: Union[float, int, Sequence[Union[float, int]]] = 3,
        aperture_type: str = "snr",
    ) -> None:
        """
        Performs aperture photometry for the given targets in the image.

        Parameters
        ----------
        targets : list of int, optional
            List of target names to fit PRF photometry for. If None, all sources in
            the image will be used.
        aperture_radius : float, int, or sequence of numbers, optional
            Radius or list of radii for circular aperture photometry in pixels.
            Default is 3.
        aperture_type : str, optional
            Default aperture method: 'snr' (optimal SNR aperture) or 'isophote'.
            Default is 'snr'.
        """
        if targets is None or len(targets) == 0:
            targets = list(range(len(self.sources)))
        n_targets = len(targets)

        if isinstance(aperture_radius, (int, float, np.integer, np.floating)):
            radii = [aperture_radius]
        elif isinstance(aperture_radius, (list, tuple, np.ndarray)):
            radii = list(aperture_radius)
        else:
            raise TypeError(
                f"`aperture_radius` must be a float, int, or list of numbers, got {type(aperture_radius)}"
            )

        self.draw_scene(frame_index=0)
        self._compute_aperture_cut()
        self._compute_snr_optimal_pixel_count()

        targets_nominal_row = np.zeros((self.nt, n_targets), dtype=np.float64)
        targets_nominal_col = np.zeros((self.nt, n_targets), dtype=np.float64)

        snr_ap_flux = np.zeros((self.nt, n_targets))
        snr_ap_flux_err = np.zeros((self.nt, n_targets))
        snr_ap_metrics = np.zeros((self.nt, n_targets, 2))
        snr_centroid = np.zeros((self.nt, n_targets, 2))
        snr_bkg_flux = np.zeros((self.nt, n_targets))
        snr_bkg_flux_err = np.zeros((self.nt, n_targets))
        snr_ap_quality = np.zeros((self.nt, n_targets))
        snr_masks = []

        iso_ap_flux = np.zeros((self.nt, n_targets))
        iso_ap_flux_err = np.zeros((self.nt, n_targets))
        iso_ap_metrics = np.zeros((self.nt, n_targets, 2))
        iso_centroid = np.zeros((self.nt, n_targets, 2))
        iso_bkg_flux = np.zeros((self.nt, n_targets))
        iso_bkg_flux_err = np.zeros((self.nt, n_targets))
        iso_ap_quality = np.zeros((self.nt, n_targets))
        iso_masks = []

        circ_data = {}
        for r in radii:
            r_key = _format_aperture_key(r)
            circ_data[r_key] = {
                "radius": float(r),
                "flux": np.zeros((self.nt, n_targets)),
                "flux_err": np.zeros((self.nt, n_targets)),
                "metrics": np.zeros((self.nt, n_targets, 2)),
                "centroid": np.zeros((self.nt, n_targets, 2)),
                "bkg_flux": np.zeros((self.nt, n_targets)),
                "bkg_flux_err": np.zeros((self.nt, n_targets)),
                "quality": np.zeros((self.nt, n_targets)),
                "masks": [],
            }

        for tdx in tqdm(
            range(self.nt),
            desc="Performing aperture photometry",
            total=self.nt,
            disable=self.quiet,
        ):
            # Record nominal positions of targets for this frame in original image pixel grid
            srow_nom, scol_nom = self.pixel_coordinates(frame_index=tdx)
            corner_row = self.row[tdx].min()
            corner_col = self.column[tdx].min()
            if np.nanmean(srow_nom) < corner_row:
                row_orig = srow_nom + corner_row
                col_orig = scol_nom + corner_col
            else:
                row_orig = srow_nom
                col_orig = scol_nom
            targets_nominal_row[tdx, :] = row_orig[targets]
            targets_nominal_col[tdx, :] = col_orig[targets]

            # update mean model due to pixel grid
            self.draw_scene(frame_index=tdx)

            # 1. Optimal SNR aperture
            ap_flux, snr_mask = self._single_frame_aperture_photometry(
                frame_index=tdx, targets=targets, aperture_type=aperture_type
            )
            snr_ap_flux[tdx] = ap_flux[:, 0]
            snr_ap_flux_err[tdx] = ap_flux[:, 1]
            snr_ap_metrics[tdx] = ap_flux[:, 2:4]
            snr_centroid[tdx] = ap_flux[:, 4:6]
            snr_bkg_flux[tdx] = ap_flux[:, 6]
            snr_bkg_flux_err[tdx] = ap_flux[:, 7]
            snr_ap_quality[tdx] = ap_flux[:, 8]
            snr_masks.append(snr_mask)

            # 2. Isophote aperture (uses self.prf_cut)
            iso_flux, iso_mask = self._single_frame_aperture_photometry(
                frame_index=tdx, targets=targets, aperture_type="iso"
            )
            iso_ap_flux[tdx] = iso_flux[:, 0]
            iso_ap_flux_err[tdx] = iso_flux[:, 1]
            iso_ap_metrics[tdx] = iso_flux[:, 2:4]
            iso_centroid[tdx] = iso_flux[:, 4:6]
            iso_bkg_flux[tdx] = iso_flux[:, 6]
            iso_bkg_flux_err[tdx] = iso_flux[:, 7]
            iso_ap_quality[tdx] = iso_flux[:, 8]
            iso_masks.append(iso_mask)

            # 3. Circular apertures
            for r_key, cdict in circ_data.items():
                c_flux, c_mask = self._single_frame_aperture_photometry(
                    frame_index=tdx,
                    targets=targets,
                    aperture_type="circular",
                    radius=cdict["radius"],
                )
                cdict["flux"][tdx] = c_flux[:, 0]
                cdict["flux_err"][tdx] = c_flux[:, 1]
                cdict["metrics"][tdx] = c_flux[:, 2:4]
                cdict["centroid"][tdx] = c_flux[:, 4:6]
                cdict["bkg_flux"][tdx] = c_flux[:, 6]
                cdict["bkg_flux_err"][tdx] = c_flux[:, 7]
                cdict["quality"][tdx] = c_flux[:, 8]
                cdict["masks"].append(c_mask)

        snr_masks_arr = np.array(snr_masks)
        iso_masks_arr = np.array(iso_masks)
        self.aperture_masks = snr_masks_arr

        if not hasattr(self, "lcs"):
            self.lcs = {}
        for sdx, snum in enumerate(targets):
            if snum not in self.lcs.keys():
                self.lcs[snum] = {}
            snr_flux_corr, snr_flux_err_corr = correct_aperture_flux(
                flux=snr_ap_flux[:, sdx],
                flux_err=snr_ap_flux_err[:, sdx],
                flfrcsap=snr_ap_metrics[:, sdx, 0],
                crowdsap=snr_ap_metrics[:, sdx, 1],
            )
            snr_dict = {
                "time": self.time,
                "flux": snr_ap_flux[:, sdx],
                "flux_err": snr_ap_flux_err[:, sdx],
                "flux_corr": snr_flux_corr,
                "flux_err_corr": snr_flux_err_corr,
                "flfrcsap": snr_ap_metrics[:, sdx, 0],
                "crowdsap": snr_ap_metrics[:, sdx, 1],
                "centroid_x": targets_nominal_col[:, sdx] + snr_centroid[:, sdx, 0],
                "centroid_y": targets_nominal_row[:, sdx] + snr_centroid[:, sdx, 1],
                "bkg_flux": snr_bkg_flux[:, sdx],
                "bkg_flux_err": snr_bkg_flux_err[:, sdx],
                "aperture_mask": snr_masks_arr[:, sdx, :],
                "quality": snr_ap_quality[:, sdx],
            }
            iso_flux_corr, iso_flux_err_corr = correct_aperture_flux(
                flux=iso_ap_flux[:, sdx],
                flux_err=iso_ap_flux_err[:, sdx],
                flfrcsap=iso_ap_metrics[:, sdx, 0],
                crowdsap=iso_ap_metrics[:, sdx, 1],
            )
            iso_dict = {
                "time": self.time,
                "flux": iso_ap_flux[:, sdx],
                "flux_err": iso_ap_flux_err[:, sdx],
                "flux_corr": iso_flux_corr,
                "flux_err_corr": iso_flux_err_corr,
                "flfrcsap": iso_ap_metrics[:, sdx, 0],
                "crowdsap": iso_ap_metrics[:, sdx, 1],
                "centroid_x": targets_nominal_col[:, sdx] + iso_centroid[:, sdx, 0],
                "centroid_y": targets_nominal_row[:, sdx] + iso_centroid[:, sdx, 1],
                "bkg_flux": iso_bkg_flux[:, sdx],
                "bkg_flux_err": iso_bkg_flux_err[:, sdx],
                "aperture_mask": iso_masks_arr[:, sdx, :],
                "quality": iso_ap_quality[:, sdx],
            }
            self.lcs[snum]["aperture_snr"] = snr_dict
            self.lcs[snum]["aperture_iso"] = iso_dict

            for r_key, cdict in circ_data.items():
                cmasks_arr = np.array(cdict["masks"])
                c_flux_corr, c_flux_err_corr = correct_aperture_flux(
                    flux=cdict["flux"][:, sdx],
                    flux_err=cdict["flux_err"][:, sdx],
                    flfrcsap=cdict["metrics"][:, sdx, 0],
                    crowdsap=cdict["metrics"][:, sdx, 1],
                )
                self.lcs[snum][r_key] = {
                    "time": self.time,
                    "flux": cdict["flux"][:, sdx],
                    "flux_err": cdict["flux_err"][:, sdx],
                    "flux_corr": c_flux_corr,
                    "flux_err_corr": c_flux_err_corr,
                    "flfrcsap": cdict["metrics"][:, sdx, 0],
                    "crowdsap": cdict["metrics"][:, sdx, 1],
                    "centroid_x": targets_nominal_col[:, sdx] + cdict["centroid"][:, sdx, 0],
                    "centroid_y": targets_nominal_row[:, sdx] + cdict["centroid"][:, sdx, 1],
                    "bkg_flux": cdict["bkg_flux"][:, sdx],
                    "bkg_flux_err": cdict["bkg_flux_err"][:, sdx],
                    "aperture_mask": cmasks_arr[:, sdx, :],
                    "quality": cdict["quality"][:, sdx],
                }

        return
    

def _load_file(
    fname: Union[str, List[str], np.ndarray],
    cutout_size: int = 32,
    cutout_center: Union[Tuple[int, int], Tuple[float, float]] = (0, 0),
    file_version: str = "1.0",
) -> Tuple[
    List[Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
    np.ndarray,
]:
    """
    Helper function to load FFI files and parse data. It parses the FITS files to
    extract the image data and metadata. It checks that all files provided in fname
    correspond to FFIs from the same mission.

    Parameters
    ----------
    fname : string or list of strings
        Name of the FFI files
    cutout_size: int
        Size of (square) portion of FFIs to cut out
    cutout_center: tuple of ints or floats
        Coordinates of the center of the cut out (row, column) or (RA, Dec).

    Returns
    -------
    wcs : astropy.wcs
        World coordinates system solution for the FFI. Used to convert RA, Dec to pixels
    time : numpy.array
        Array with time values in MJD
    flux : numpy.ndarray
        3D array of flux values
    flux_err : numpy.ndarray
        3D array of flux errors
    ra_3d : numpy.ndarray
        Array with 3D (time, image) representation of flux RA
    dec_3d : numpy.ndarray
        Array with 3D (time, image) representation of flux Dec
    col_2d : numpy.ndarray
        Array with 2D (image) representation of pixel column
    row_3d : numpy.ndarray
        Array with 2D (image) representation of pixel row
    meta : dict
        Dictionary with metadata
    """
    if not isinstance(fname, (list, np.ndarray)):
        fname = np.sort([fname])
    if len(cutout_center) != 2:
        raise ValueError(
            "`cutout_center` must be a tuple of two values (row, column) or (RA, Dec)."
        )
    if isinstance(cutout_center[0], (int, np.int32, np.int64)):
        rowcol = cutout_center
        radec = (None, None)
        dithered = False
    elif isinstance(cutout_center[0], (float, np.float32, np.float64)):
        radec = cutout_center
        rowcol = (None, None)
        dithered = True
    else:
        raise ValueError(
            "`cutout_center` must be a tuple of two int values (row, column) or float (RA, Dec)."
        )
    field = int(os.path.basename(fname[0]).split("_")[5][5:])
    sca = int(os.path.basename(fname[0]).split("_")[4][3:])
    filter = os.path.basename(fname[0]).split("_")[3]
    if "asdf" in os.path.basename(fname[0]):
        format = "asdf"
    elif "fits" in os.path.basename(fname[0]):
        format = "fits"
    else:
        raise ValueError("Input file is not one of 'asdf' or 'fits'.")

    rcube = RomanCuts(
        field=field,
        sca=sca,
        filter=filter,
        file_list=fname,
        file_format=format,
        file_version=file_version,
    )
    rcube.make_cutout(
        rowcol=rowcol, radec=radec, size=(cutout_size, cutout_size), dithered=dithered
    )
    log.info("Cutout made...")

    # put row,col and ra,dec into 3D arrasy [ntimes, axis1, axis2]
    if dithered:
        row_3d, col_3d = np.vstack(
            [
                [np.meshgrid(r, c, indexing="ij")]
                for r, c in zip(rcube.row, rcube.column)
            ]
        ).transpose((1, 0, 2, 3))
        ra_3d, dec_3d = np.vstack(
            [[x.all_pix2world(r, c, 0)] for x, r, c in zip(rcube.wcss, row_3d, col_3d)]
        ).transpose((1, 0, 2, 3))
    else:
        row_3d, col_3d = np.meshgrid(rcube.row, rcube.column, indexing="ij")
        row_3d = np.atleast_3d(row_3d).transpose((2, 0, 1))
        col_3d = np.atleast_3d(col_3d).transpose((2, 0, 1))
        ra_3d, dec_3d = np.vstack(
            [[x.all_pix2world(row_3d[0], col_3d[0], 0)] for x in rcube.wcss]
        ).transpose((1, 0, 2, 3))

    return (
        rcube.wcss,
        rcube.time,
        rcube.exposureno,
        rcube.flux,
        rcube.flux_err,
        ra_3d,
        dec_3d,
        col_3d,
        row_3d,
        rcube.metadata,
        rcube.quality,
    )
