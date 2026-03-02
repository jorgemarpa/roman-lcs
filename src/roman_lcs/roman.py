"""Subclass of `Machine` that Specifically work with FFIs"""

import logging
import os
from typing import Any, List, Optional, Tuple, Union

import astropy.units as u
import lightkurve as lk
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import simple_norm
from roman_cuts import RomanCuts
from scipy import ndimage, sparse
from tqdm import tqdm

from . import __version__
from .aperture import *
from .machine import Machine
from .prf import RomanPRF
from .utils import _make_A_cartesian, _make_A_polar, solve_linear_model

log = logging.getLogger(__name__)


class RomanMachine(Machine):
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
        wcs : astropy.wcs
            World coordinates system solution for the FFI. Used for plotting.
        flux_3d : numpy.ndarray
            2D image representation of the FFI, used for plotting. Has shape [n_times,
            image_height, image_width]
        image_shape : tuple
            Shape of 2D image
        """

        self.ref_frame = 0
        self.cadenceno = cadenceno

        self.WCSs = wcs
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
            sparse_dist_lim=sparse_dist_lim,
            sources_flux_column=sources_flux_column,
        )
        self._mask_pixels()
        if quality_mask is None:
            self.quality_mask = np.zeros(len(time), dtype=int)
        else:
            self.quality_mask = quality_mask

    def __repr__(self) -> str:
        return f"RomanMachine (N sources, N times, N pixels): {self.shape}"

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

    @staticmethod
    def from_file(
        fname: Union[str, List[str], np.ndarray],
        cutout_size: int = 32,
        cutout_center: Union[Tuple[float, float], Tuple[int, int]] = (0, 0),
        sources: Optional[pd.DataFrame] = None,
        file_version: str = "1.0",
        **kwargs,
    ) -> "RomanMachine":
        """
        Reads data from files and initiates a new object of RomanMachine class.
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
        RomanMachine : Machine object
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
        log.info("Initializing RomanMachine object...")
        return RomanMachine(
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
        self, pixel_saturation_limit: float = 2e4, magnitude_bright_limit: float = 13
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
        self.non_sat_pixel_mask = ~self._saturated_pixels_mask(
            saturation_limit=pixel_saturation_limit
        )
        # tolerance dependens on pixel scale, TESS pixels are 5 times larger than TESS
        self.non_bright_source_mask = ~self._bright_sources_mask(
            magnitude_limit=magnitude_bright_limit, tolerance=10
        )
        self.pixel_mask = self.non_sat_pixel_mask & self.non_bright_source_mask

        # if not hasattr(self, "source_mask"):
        #     self._get_source_mask()
        #     # include saturated pixels in the source mask and uncontaminated mask
        #     self._remove_bad_pixels_from_source_mask()

        return

    def _saturated_pixels_mask(
        self, saturation_limit: float = 1e5, tolerance: int = 3
    ) -> np.ndarray:
        """
        Finds and removes saturated pixels, including bleed columns.

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
        # Which pixels are saturated
        # this nanpercentile takes forever to compute for a single cadence ffi
        # saturated = np.nanpercentile(self.flux, 99, axis=0)
        # assume we'll use ffi for 1 single cadence
        sat_mask = self.flux.max(axis=0) > saturation_limit
        # dilate the mask with tolerance
        sat_mask = ndimage.binary_dilation(sat_mask, iterations=tolerance)

        # add nan values to the mask
        sat_mask |= ~np.isfinite(self.flux.max(axis=0))

        return sat_mask

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

        mask = [
            np.hypot(self.ra[0] - s.ra, self.dec[0] - s.dec) < tolerance
            for _, s in self.sources[bright_mask].iterrows()
        ]
        mask = np.array(mask).sum(axis=0) > 0

        return mask

    def _pointing_offset(self) -> None:
        """
        Computes pointing offsets due to dittering
        """
        self.ra_offset = (self.ra - self.ra[0]).mean(axis=1)
        self.dec_offset = (self.dec - self.dec[0]).mean(axis=1)

    def _remove_bad_pixels_from_source_mask(self) -> None:
        """
        Combines source_mask and uncontaminated_pixel_mask with saturated and bright
        pixel mask.
        """
        self.source_mask = self.source_mask.multiply(self.pixel_mask).tocsr()
        self.source_mask.eliminate_zeros()
        self.uncontaminated_source_mask = self.uncontaminated_source_mask.multiply(
            self.pixel_mask
        ).tocsr()
        self.uncontaminated_source_mask.eliminate_zeros()

    def load_prf_model(
        self,
        input: Optional[str] = None,
        plot: bool = False,
    ) -> Optional[matplotlib.figure.Figure]:
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
        # check if file exists and is the right format
        self.prf_model = RomanPRF(
            file_name=input, sca=self.meta["SCA"], filter=self.meta["FILTER"]
        )
        log.info(f"Loading shape model from {input}")
        if plot:
            return self.plot_prf_model(hires=True, stretch="log")
        return

    def draw_scene(self, frame_index: int = 0) -> None:
        """
        Convenience function to make the scene model
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

        # evaluate PRF at each source position
        scene_model = []
        for sdx in range(self.nsources):
            eval = self.prf_model.evaluate_from_position(
                center=(srow[sdx], scol[sdx]),
                corner=(row.min(), col.min()),
                shape=self.image_shape,
            )[-1]
            # eval = np.rot90(eval.reshape(self.image_shape), k=1).ravel()
            scene_model.append(eval.ravel())
        self.scene_model = np.array(scene_model)
        # we assume the at least one source will be fully in the cutout, so we 
        # normalize the model to that sum
        self.scene_model /= self.scene_model.sum(axis=1).max()
        # then scale by PRF normalization factor
        self.scene_model *= self.prf_model.prf_sum
        return

    def plot_prf_model(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        hires: bool = False,
        stretch: str = "linear",
    ) -> matplotlib.axes.Axes:
        """
        Plot the Point Response Function (PRF) model for the current RomanMachine instance.

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
                [0, 1.05, 1, 0.25], sharex=ax_main
            )  # [left, bottom, width, height]
            # ax_histy (right): positioned to the right of the main axes
            ax_histy = ax_main.inset_axes([1.05, 0, 0.25, 1], sharey=ax_main)

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
        if hires:
            ax_main.set_title(f"Super Sampled PRF | {self.meta['FILTER']}", y=1.1)
        else:
            ax_main.set_title(f"PRF in Cutout Pixel Grid | {self.meta['FILTER']}")

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
        ax_main.set_xlabel("$\Delta$ Column [arcsec]")
        ax_main.set_ylabel("$\Delta$ Row [arcsec]")
        ax_main.set_aspect("equal")

        return ax

    def plot_image(
        self,
        data: str = "image",
        ax: Optional[matplotlib.axes.Axes] = None,
        sources: bool = False,
        frame_index: int = 0,
    ) -> matplotlib.axes.Axes:
        """
        Function to plot the Full Frame Image and Gaia sources.

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned.
        sources : boolean
            Whether to overplot or not the source catalog.
        frame_index : int
            Time index used to plot the image data.

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure.
        """
        if ax is None:
            _ = plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=self.WCSs[frame_index], label="overlays")

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
            raise ValueError
        norm = simple_norm(to_plot.ravel(), "asinh", percent=95)

        bar = ax.pcolormesh(
            self.column_3d[frame_index],
            self.row_3d[frame_index],
            to_plot,
            norm=norm,
            cmap=plt.cm.viridis,
            rasterized=True,
        )
        plt.colorbar(bar, ax=ax, shrink=0.7, label=r"Flux ($e^{-}s^{-1}$)")
        ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.set_xlim(self.column_3d.min() - 3, self.column_3d.max() + 3)
        ax.set_ylim(self.row_3d.min() - 3, self.row_3d.max() + 3)

        ax.set_title(
            f"{self.meta['MISSION']} | {self.meta['DETECTOR']} | {self.meta['FILTER']}\n"
            f"Frame {self.cadenceno[frame_index]} | JD {self.time[frame_index]} \n"
            f"{title}"
        )

        if sources:
            srow, scol = self.pixel_coordinates(frame_index=frame_index)
            # marker size correlates with source magnitude
            # size = self.sources.loc[:, self.meta["FILTER"]].values
            # size = np.exp(0.1 / ((size - 1-) / (10)))
            ax.scatter(
                scol,
                srow,
                c="tab:red",
                marker="o",
                s=5,
                linewidths=0.1,
                alpha=0.8,
            )

        ax.set_aspect("equal", adjustable="box")

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
        if hasattr(self, "non_sat_pixel_mask"):
            ax.scatter(
                self.column_3d.ravel()[~self.non_sat_pixel_mask],
                self.row_3d.ravel()[~self.non_sat_pixel_mask],
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

    def get_lightcurves(self, mode: str = "lk") -> None:
        """
        Bundle light curves as `lightkurve` objects is `mode=="lk"`
        or as a DataFrame if `mode=="table"'.

        Parameters
        ----------
        mode : str
            What type of light curve wil be created
        """
        if mode == "lk":
            lcs = []
            for idx, s in self.sources.iterrows():
                meta = {}
                lc = lk.LightCurve(
                    time=(self.time) * u.d,
                    flux=self.ws[:, idx] * u.electron / u.second,
                    flux_err=self.werrs[:, idx] * u.electron / u.second,
                    meta=meta,
                    time_format="mjd",
                )
                lcs.append(lc)

            self.lcs = lk.LightCurveCollection(lcs)
        elif mode == "table":
            raise NotImplementedError

    def _get_bkg_model_terms(
        self,
        target_idx: List[int] = [0],
        gradient: bool = True,
        bkg_poly_order: int = 3,
    ):
        """
        Returns background model terms for the given target index.

        Parameters
        ----------
        target_idx : int or list of ints
            Index of the target for which to compute background model terms.
            If a list is provided, the first element will be used.
        gradient : bool
            Whether to include gradient terms in the background model.
        bkg_poly_order : int
            Order of the polynomial used for background model fitting.

        Returns
        -------
        bkg_terms : numpy.ndarray
            Array of background model terms for the given target index.
            The shape of the array is (n_terms, n_pixels), where n_terms is the
            number of background model terms and n_pixels is the number of pixels
            in the image.
        """
        tpfshape = self.image_shape
        bkg_terms = []
        bkg_terms.append(np.ones(tpfshape).ravel())

        if gradient:
            dx_ravel = self.dcolumn[target_idx].ravel()
            dy_ravel = self.drow[target_idx].ravel()

            for i in range(1, bkg_poly_order + 1):
                for j in range(0, i + 1):
                    bkg_terms.append(dx_ravel**j * dy_ravel ** (i - j))

        return np.array(bkg_terms)

    def fit_prf_photometry(
        self,
        targets: List[int] = [],
        model_bkg: bool = True,
        bkg_nknots: int = 3,
        do_aperture: bool = True,
    ) -> None:
        """
        Fits PRF photometry the given targets in the image accounting for backgronund
        stars and signal.

        Parameters
        ----------
        targets : list of int, optional
            List of target names to fit PRF photometry for. If None, all sources in
        """
        n_targets = len(targets) if len(targets) > 0 else len(self.sources)
        targets_prf_flux = np.zeros((self.nt, n_targets))
        targets_prf_flux_err = np.zeros((self.nt, n_targets))
        model_chi2 = np.zeros(self.nt)
        scene_model = np.zeros_like(self.flux)
        bkg_model = np.zeros_like(self.flux)

        if do_aperture:
            targets_ap_flux = np.zeros((self.nt, len(targets)))
            targets_ap_flux_err = np.zeros((self.nt, len(targets)))
            aperture_metrics = np.zeros((self.nt, len(targets), 2))
            targets_centroid = np.zeros((self.nt, len(targets), 2))
            aperture_masks = []

        # prior_mu = self.source_flux_estimates
        # prior_sigma = (
        #     np.ones(self.mean_model.shape[0])
        #     * 5
        #     * np.abs(self.source_flux_estimates) ** 0.5
        # )

        for tdx in tqdm(
            range(self.nt),
            desc="Fitting PRF photometry",
            total=self.nt,
            disable=self.quiet,
        ):
            # update scene model due to pixel grid offsets
            self.draw_scene(frame_index=tdx)
            mean_model = sparse.csr_matrix(self.scene_model.copy())
            if len(targets) > 0:
                targets_models = mean_model[targets]
                # get background stars PSF model
                bkg_star_model = mean_model[
                    ~np.isin(self.sources.index.values, targets)
                ]
                bkg_star_flux = np.delete(self.source_flux_estimates, targets)
                # compute background star scene
                bkg_star_model = bkg_star_model.T.dot(bkg_star_flux)[None, :]
                bkg_star_model = sparse.csr_matrix(bkg_star_model)
                # stack linear model with target, bkg stars and bkg signal
                model = sparse.vstack([targets_models, bkg_star_model]).tocsr()
                # model = np.vstack([targets_models, bkg_star_model])
                # prior_mu = np.append(self.source_flux_estimates[targets], [1])
                # prior_sigma = np.ones(prior_mu.shape[0]) * 5 * np.abs(prior_mu) ** 0.5
            else:
                model = mean_model
            if model_bkg:
                bkg_terms = _make_A_cartesian(
                    x=self.dcolumn[targets[0] if len(targets) > 0 else 0].ravel(),
                    y=self.drow[targets[0] if len(targets) > 0 else 0].ravel(),
                    n_knots=bkg_nknots,
                ).T
                model = sparse.vstack([model, bkg_terms]).tocsr()
                # model = np.vstack([model, bkg_terms])
                # prior_mu = np.concatenate([prior_mu, np.ones(bkg_terms.shape[0])])
                # prior_sigma = np.ones(prior_mu.shape[0]) * 5 * np.abs(prior_mu) ** 0.5
            # solve linear model with current flux
            try:
                w, werr = solve_linear_model(
                    # sparse.csr_matrix(model.T),
                    model.T,
                    y=self.flux[tdx],
                    y_err=self.flux_err[tdx],
                    errors=True,
                    # prior_mu=prior_mu,
                    # prior_sigma=prior_sigma,
                )
            except np.linalg.LinAlgError as e:
                log.error(f"Error solving linear model: {e}")
                log.error("Skipping this cadence.")
                targets_prf_flux[tdx, :] = -1e6
                targets_prf_flux_err[tdx, :] = -1e6
                continue
            # assign flux phot values to targets
            targets_prf_flux[tdx, :] = w[:n_targets]
            targets_prf_flux_err[tdx, :] = werr[:n_targets]
            # build full scene model
            scene_model[tdx] = model.T.dot(w).ravel()
            if model_bkg:
                bkg_model[tdx] = model[n_targets:].T.dot(w[n_targets:]).ravel()

            # reduced chi2
            if tdx == 0:
                model_rank = np.linalg.matrix_rank(model.toarray())
            model_chi2[tdx] = ((self.flux[tdx] - scene_model[tdx]) ** 2).sum() / (
                np.prod(self.image_shape) - model_rank
            )

            # do aperture photometry for targets
            if do_aperture:
                ap_flux, aperture_mask = self._single_frame_aperture_photometry(
                    frame_index=tdx, targets=targets
                )
                targets_ap_flux[tdx] = ap_flux[:, 0]
                targets_ap_flux_err[tdx] = ap_flux[:, 1]
                aperture_metrics[tdx] = ap_flux[:, 2:4]
                targets_centroid[tdx] = ap_flux[:, 4:6]
                aperture_masks.append(aperture_mask)

        self.scene_model = scene_model
        self.bkg_model = bkg_model

        if not hasattr(self, "lc"):
            self.lc = {}
        if "psf" not in self.lc.keys():
            self.lc["psf"] = {}
        for sdx, snum in enumerate(targets):
            self.lc["psf"][snum] = {
                "time": self.time,
                "flux": targets_prf_flux[:, sdx],
                "flux_err": targets_prf_flux_err[:, sdx],
                "chi2": model_chi2,
                # "quality": ,
            }

        if do_aperture:
            aperture_masks = np.array(aperture_masks)
            self.lc["aperture"] = {}
            for sdx, snum in enumerate(targets):
                self.lc["aperture"][snum] = {
                    "time": self.time,
                    "flux": targets_ap_flux[:, sdx],
                    "flux_err": targets_ap_flux_err[:, sdx],
                    "flfrcsap": aperture_metrics[:, sdx, 0],
                    "crowdsap": aperture_metrics[:, sdx, 1],
                    "centroid_x": targets_centroid[:, sdx, 0],
                    "centroid_y": targets_centroid[:, sdx, 1],
                    "aperture_mask": aperture_masks[:, sdx, :],
                    # "quality": ,
                }

        return

    def _single_frame_aperture_photometry(
        self,
        frame_index: int = 0,
        targets: List[int] = [],
        percentile_cut: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs aperture photometry for a single frame and given targets in the image.
        """
        if percentile_cut is None:
            percentile_cut = np.array([90] * self.nsources)
        cut = np.array(
            [
                np.nanpercentile(obj.data, per)
                for obj, per in zip(self.scene_model, percentile_cut)
            ]
        )
        aperture_mask = np.array(self.scene_model >= cut[::, None])


        targets_ap_flux = np.array(
            [
                [
                    self.flux[frame_index, ap].sum(),
                    np.sqrt((self.flux_err[frame_index, ap] ** 2).sum()),
                ]
                for ap in aperture_mask[targets]
            ]
        )
        flux_metrics = np.array(
            [
                compute_FLFRCSAP(self.scene_model, aperture_mask)[targets],
                compute_CROWDSAP(self.scene_model, aperture_mask)[targets],
            ]
        ).T

        targets_centroid = np.array(
            [
                [
                    np.average(
                        self.dcolumn[tar][aperture_mask[tar]],
                        weights=self.flux[frame_index, aperture_mask[tar]],
                    ),
                    np.average(
                        self.drow[tar][aperture_mask[tar]],
                        weights=self.flux[frame_index, aperture_mask[tar]],
                    ),
                ]
                for tar in targets
            ]
        )
        return np.hstack(
            [targets_ap_flux, flux_metrics, targets_centroid]
        ), aperture_mask[targets]

    def do_aperture_photometry(self, targets: List[int] = []) -> None:
        """
        Performs aperture photometry for the given targets in the image.

        Parameters
        ----------
        targets : list of int, optional
            List of target names to fit PRF photometry for. If None, all sources in
            the image will be used.
        """
        targets_ap_flux = np.zeros((self.nt, len(targets)))
        targets_ap_flux_err = np.zeros((self.nt, len(targets)))
        aperture_metrics = np.zeros((self.nt, len(targets), 2))
        targets_centroid = np.zeros((self.nt, len(targets), 2))
        aperture_masks = []
        for tdx in tqdm(
            range(self.nt),
            desc="Performing aperture photometry",
            total=self.nt,
            disable=self.quiet,
        ):
            # update mean model due to  pixel grid
            self.draw_scene(frame_index=tdx)
            # get aperture mask
            ap_flux, aperture_mask = self._single_frame_aperture_photometry(
                frame_index=tdx, targets=targets
            )
            targets_ap_flux[tdx] = ap_flux[:, 0]
            targets_ap_flux_err[tdx] = ap_flux[:, 1]
            aperture_metrics[tdx] = ap_flux[:, 2:4]
            targets_centroid[tdx] = ap_flux[:, 4:6]
            aperture_masks.append(aperture_mask)

        aperture_masks = np.array(aperture_masks)

        if not hasattr(self, "lc"):
            self.lc = {}
        if "aperture" not in self.lc.keys():
            self.lc["aperture"] = {}
        for sdx, snum in enumerate(targets):
            self.lc["aperture"][snum] = {
                "time": self.time,
                "flux": targets_ap_flux[:, sdx],
                "flux_err": targets_ap_flux_err[:, sdx],
                "flfrcsap": aperture_metrics[:, sdx, 0],
                "crowdsap": aperture_metrics[:, sdx, 1],
                "centroid_x": targets_centroid[:, sdx, 0],
                "centroid_y": targets_centroid[:, sdx, 1],
                "aperture_mask": aperture_masks[:, sdx, :],
                # "quality": ,
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
