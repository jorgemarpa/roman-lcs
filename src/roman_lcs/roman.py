"""Subclass of `Machine` that Specifically work with FFIs"""

import os
from typing import Optional, Union

import astropy.units as u
import fitsio
import lightkurve as lk
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from scipy import ndimage
from tqdm import tqdm

from . import __version__
from .machine import Machine


class RomanMachine(Machine):
    """
    Subclass of Machine for working with Roman data.
    """

    def __init__(
        self,
        time,
        flux,
        flux_err,
        ra,
        dec,
        sources,
        column,
        row,
        cadenceno=None,
        wcs=None,
        n_r_knots=9,
        n_phi_knots=15,
        cut_r=0.15,
        rmin=0.02,
        rmax=0.8,
        sparse_dist_lim=4,
        quality_mask=None,
        sources_flux_column="flux",
        sources_mag_column="F146",
        meta=None,
    ):
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
        flux_2d : numpy.ndarray
            2D image representation of the FFI, used for plotting. Has shape [n_times,
            image_height, image_width]
        image_shape : tuple
            Shape of 2D image
        """

        self.ref_frame = 0
        self.cadenceno = cadenceno

        self.WCSs = wcs
        self.meta = meta

        # keep 2d image shape
        self.image_shape = flux.shape[1:]
        self.sources_mag_column = sources_mag_column

        # init `machine` object
        super().__init__(
            time,
            flux.reshape(flux.shape[0], -1),
            flux_err.reshape(flux_err.shape[0], -1),
            ra.reshape(ra.shape[0], -1),
            dec.reshape(dec.shape[0], -1),
            sources,
            column,
            row,
            n_r_knots=n_r_knots,
            n_phi_knots=n_phi_knots,
            cut_r=cut_r,
            rmin=rmin,
            rmax=rmax,
            sparse_dist_lim=sparse_dist_lim,
            sources_flux_column=sources_flux_column,
        )
        self._mask_pixels()
        if quality_mask is None:
            self.quality_mask = np.zeros(len(time), dtype=int)
        else:
            self.quality_mask = quality_mask

    def __repr__(self):
        return f"RomanMachine (N sources, N times, N pixels): {self.shape}"

    @property
    def flux_2d(self):
        return self.flux.reshape((self.flux.shape[0], *self.image_shape))

    @property
    def flux_err_2d(self):
        return self.flux_err.reshape((self.flux_err.shape[0], *self.image_shape))

    @property
    def row_2d(self):
        return self.row.reshape((self.image_shape))

    @property
    def column_2d(self):
        return self.column.reshape((self.image_shape))

    @property
    def ra_3d(self):
        return self.ra.reshape((self.ra.shape[0], *self.image_shape))

    @property
    def dec_3d(self):
        return self.dec.reshape((self.dec.shape[0], *self.image_shape))

    @staticmethod
    def from_file(
        fname,
        cutout_size=None,
        cutout_origin=[0, 0],
        sources=None,
        **kwargs,
    ):
        """
        Reads data from files and initiates a new object of RomanMachine class.

        Parameters
        ----------
        fname : str or list of strings
            File name or list of file names of the FFI files.
        cutout_size : int
            Size of the cutout in pixels, assumed to be squared
        cutout_origin : tuple of ints
            Origin pixel coordinates where to start the cut out. Follows matrix indexing
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
            cutout_origin=cutout_origin,
        )

        if ra.shape == flux.shape:
            ra = ra.reshape(flux.shape[0], -1)
        if dec.shape == flux.shape:
            dec = dec.reshape(flux.shape[0], -1)

        return RomanMachine(
            time,
            flux,
            flux_err,
            ra,
            dec,
            sources,
            column.ravel(),
            row.ravel(),
            cadenceno=cadenceno,
            wcs=wcs,
            meta=metadata,
            quality_mask=quality_mask,
            **kwargs,
        )

    def _mask_pixels(
        self, pixel_saturation_limit: float = 2e4, magnitude_bright_limit: float = 13
    ):
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

    def _saturated_pixels_mask(self, saturation_limit: float = 1e5, tolerance: int = 3):
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
        # this nanpercentile takes forever to compute for a single cadance ffi
        # saturated = np.nanpercentile(self.flux, 99, axis=0)
        # assume we'll use ffi for 1 single cadence
        sat_mask = self.flux.max(axis=0) > saturation_limit
        # dilate the mask with tolerance
        sat_mask = ndimage.binary_dilation(sat_mask, iterations=tolerance)

        # add nan values to the mask
        sat_mask |= ~np.isfinite(self.flux.max(axis=0))

        return sat_mask

    def _bright_sources_mask(self, magnitude_limit: float = 13, tolerance: float = 30):
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
            np.hypot(self.column - s.column, self.row - s.row) < tolerance
            for _, s in self.sources[bright_mask].iterrows()
        ]
        mask = np.array(mask).sum(axis=0) > 0

        return mask

    def _pointing_offset(self):
        self.ra_offset = (self.ra - self.ra[0]).mean(axis=1)
        self.dec_offset = (self.dec - self.dec[0]).mean(axis=1)

    def _get_source_mask(
        self,
        source_flux_limit: float = 1,
        reference_frame: int = 0,
        iterations: int = 2,
        plot=False,
    ):
        """
        Adapted version of `machine._get_source_mask()` that masks out saturated and
        bright halo pixels in FFIs. See parameter descriptions in `Machine`.
        """
        super()._get_source_mask(
            source_flux_limit=source_flux_limit,
            reference_frame=reference_frame,
            iterations=iterations,
            plot=plot,
        )
        # self._remove_bad_pixels_from_source_mask()

    def _update_source_mask(self, frame_index: int = 0, source_flux_limit: float = 1):
        super()._update_source_mask(
            frame_index=frame_index,
            source_flux_limit=source_flux_limit,
        )
        # self._remove_bad_pixels_from_source_mask()

    def _remove_bad_pixels_from_source_mask(self):
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


    def build_shape_model(
        self, 
        flux_cut_off: float = 1,
        frame_index: Union[str, int] = 0,
        bin_data: bool = False,
        plot: bool = False,
        **kwargs,
    ):
        """
        Adapted version of `machine.build_shape_model()` that masks out saturated and
        bright halo pixels in FFIs. See parameter descriptions in `Machine`.
        """
        # call method from super calss `machine`
        super().build_shape_model(
            plot=False,
            flux_cut_off=flux_cut_off,
            frame_index=frame_index,
            bin_data=bin_data,
            **kwargs,
        )
        # include sat/halo pixels again into source_mask
        # self._remove_bad_pixels_from_source_mask()
        if plot:
            return self.plot_shape_model(frame_index=frame_index, bin_data=bin_data)

    def save_shape_model(self, output=None):
        """
        Saves the weights of a PRF fit to disk.

        Parameters
        ----------
        output : str, None
            Output file name. If None, one will be generated.
        """
        # asign a file name
        if output is None:
            output = f"./{self.meta['MISSION']}_shape_model_{self.meta['FILTER']}_{self.meta['DETECTOR']}.fits"

        # create data structure (DataFrame) to save the model params
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="psf_w",
                    array=self.psf_w,
                    format="D",
                )
            ]
        )
        # include metadata and descriptions
        table.header["OBJECT"] = ("PRF shape", "PRF shape parameters")
        table.header["DATATYPE"] = ("SimImage", "Type of data used to fit shape model")
        table.header["ORIGIN"] = ("PSFmachine.RomanMachine", "Software of origin")
        table.header["VERSION"] = (__version__, "Software version")
        table.header["TELESCOP"] = (self.meta["TELESCOP"], "Telescope name")
        table.header["MISSION"] = (self.meta["MISSION"], "Mission name")

        table.header["FIELD"] = (self.meta["FIELD"], "Field")
        table.header["DETECTOR"] = (self.meta["DETECTOR"], "Instrument detector")
        table.header["FILTER"] = (self.meta["FILTER"], "Instrument filter")

        table.header["JD-OBS"] = (self.time[0], "JD of observation")
        table.header["n_rknots"] = (
            self.n_r_knots,
            "Number of knots for spline basis in radial axis",
        )
        table.header["n_pknots"] = (
            self.n_phi_knots,
            "Number of knots for spline basis in angle axis",
        )
        table.header["rmin"] = (self.rmin, "Minimum value for knot spacing")
        table.header["rmax"] = (self.rmax, "Maximum value for knot spacing")
        table.header["cut_r"] = (
            self.cut_r,
            "Radial distance to remove angle dependency",
        )
        # spline degree is hardcoded in `_make_A_polar` implementation.
        table.header["spln_deg"] = (3, "Degree of the spline basis")
        table.header["norm"] = (str(False), "Normalized model")

        table.writeto(output, checksum=True, overwrite=True)

    def load_shape_model(self, input=None, plot=False, source_flux_limit=20, flux_cut_off=0.01):
        """
        Load and process a shape model for the sources.

        This method reads a shape model from the specified input source, applies any necessary
        processing, and optionally generates a diagnostic plot of the shape model. The function
        may also filter out low-flux pixels based on the provided cutoff value.

        Parameters
        ----------
        input : str, optional
            The path to the shape model file or other input source. If None, defaults to a predefined
            shape model location.

        plot : bool, optional, default=False
            Whether to display a diagnostic plot of the loaded shape model. If set to True, the plot
            will be shown upon loading the model.

        flux_cut_off : float, optional, default=0.01
            The minimum flux value below which sources will be excluded from the model. This can help
            remove noise or irrelevant data during processing.

        Returns
        -------
        None
            This function does not return any value. It modifies the internal state of the object
            by loading the shape model and potentially creating plots.
        """
        # check if file exists and is the right format
        if not os.path.isfile(input):
            raise FileNotFoundError(f"No shape file: {input}")

        # create source mask and uncontaminated pixel mask
        # if not hasattr(self, "source_mask"):
        self._get_source_mask(
            source_flux_limit=source_flux_limit,
            plot=False,
            reference_frame=self.ref_frame,
            iterations=1,
        )

        # open file
        hdu = fits.open(input)
        # check if shape parameters are for correct mission, quarter, and channel
        if hdu[1].header["MISSION"].strip() != self.meta["MISSION"]:
            raise ValueError("Wrong shape model: file is for mission Roman")
        if int(hdu[1].header["FIELD"]) != self.meta["FIELD"]:
            raise ValueError("Wrong field")
        if hdu[1].header["DETECTOR"].strip() != self.meta["DETECTOR"]:
            raise ValueError("Wrong DETECTOR")

        # load model hyperparameters and weights
        self.n_r_knots = hdu[1].header["n_rknots"]
        self.n_phi_knots = hdu[1].header["n_pknots"]
        self.rmin = hdu[1].header["rmin"]
        self.rmax = hdu[1].header["rmax"]
        self.cut_r = hdu[1].header["cut_r"]
        self.psf_w = hdu[1].data["psf_w"]
        # read from header if weights come from a normalized model.
        self.normalized_shape_model = (
            True if hdu[1].header.get("norm") in ["True", "T", 1] else False
        )
        del hdu

        # create mean model, but PRF shapes from FFI are in pixels! and TPFMachine
        # work in arcseconds
        self._get_mean_model()
        # remove background pixels and recreate mean model
        self._update_source_mask_remove_bkg_pixels(flux_cut_off=flux_cut_off)
        # self._remove_bad_pixels_from_source_mask()

        if plot:
            return self.plot_shape_model(frame_index=self.ref_frame)
        return

    def residuals(self, plot=False, zoom=False, metric="residuals"):
        """
        Get the residuals (model - image) and compute statistics. It creates a model
        of the full image using the `mean_model` and the weights computed when fitting
        the shape model.

        Parameters
        ----------
        plot : bool
            Do plotting.
        zoom : bool
            If plot is True then zoom into a section of the image for better
            visualization.
        metric : string
            Type of metric used to plot. Default is "residuals", "chi2" is also
            available.

        Return
        ------
        fig : matplotlib figure
            Figure.
        """
        if not hasattr(self, "ws"):
            self.fit_model(fit_va=False)

        # evaluate mean model
        ffi_model = self.mean_model.T.dot(self.ws[0])
        ffi_model_err = self.mean_model.T.dot(self.werrs[0])
        # compute residuals
        residuals = ffi_model - self.flux[0]
        weighted_chi = (ffi_model - self.flux[0]) ** 2 / ffi_model_err
        # mask background
        source_mask = ffi_model != 0.0
        # rms
        self.rms = np.sqrt((residuals[source_mask] ** 2).mean())
        self.frac_esidual_median = np.median(
            residuals[source_mask] / self.flux[0][source_mask]
        )
        self.frac_esidual_std = np.std(
            residuals[source_mask] / self.flux[0][source_mask]
        )

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(15, 15))

            ax[0, 0].scatter(
                self.column,
                self.row,
                c=self.flux[0],
                marker="s",
                s=7.5 if zoom else 1,
                norm=colors.SymLogNorm(linthresh=500, vmin=0, vmax=5000, base=10),
            )
            ax[0, 0].set_aspect("equal", adjustable="box")

            ax[0, 1].scatter(
                self.column,
                self.row,
                c=ffi_model,
                marker="s",
                s=7.5 if zoom else 1,
                norm=colors.SymLogNorm(linthresh=500, vmin=0, vmax=5000, base=10),
            )
            ax[0, 1].set_aspect("equal", adjustable="box")

            if metric == "residuals":
                to_plot = residuals
                norm = colors.SymLogNorm(linthresh=500, vmin=-5000, vmax=5000, base=10)
                cmap = "RdBu"
            elif metric == "chi2":
                to_plot = weighted_chi
                norm = colors.LogNorm(vmin=1, vmax=5000)
                cmap = "viridis"
            else:
                raise ValueError("wrong type of metric")

            cbar = ax[1, 0].scatter(
                self.column[source_mask],
                self.row[source_mask],
                c=to_plot[source_mask],
                marker="s",
                s=7.5 if zoom else 1,
                cmap=cmap,
                norm=norm,
            )
            ax[1, 0].set_aspect("equal", adjustable="box")
            plt.colorbar(
                cbar, ax=ax[1, 0], label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042
            )

            ax[1, 1].hist(
                residuals[source_mask] / self.flux[0][source_mask],
                bins=50,
                log=True,
                label=(
                    "RMS (model - data) = %.3f" % self.rms
                    + "\nMedian = %.3f" % self.frac_esidual_median
                    + "\nSTD = %3f" % self.frac_esidual_std
                ),
            )
            ax[1, 1].legend(loc="best")

            ax[0, 0].set_ylabel("Pixel Row Number")
            ax[0, 0].set_xlabel("Pixel Column Number")
            ax[0, 1].set_xlabel("Pixel Column Number")
            ax[1, 0].set_ylabel("Pixel Row Number")
            ax[1, 0].set_xlabel("Pixel Column Number")
            ax[1, 1].set_xlabel("(model - data) / data")
            ax[1, 0].set_title(metric)

            if zoom:
                ax[0, 0].set_xlim(self.column.min(), self.column.min() + 100)
                ax[0, 0].set_ylim(self.row.min(), self.row.min() + 100)
                ax[0, 1].set_xlim(self.column.min(), self.column.min() + 100)
                ax[0, 1].set_ylim(self.row.min(), self.row.min() + 100)
                ax[1, 0].set_xlim(self.column.min(), self.column.min() + 100)
                ax[1, 0].set_ylim(self.row.min(), self.row.min() + 100)

            return fig
        return

    def plot_image(self, ax=None, sources=False, frame_index=0):
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
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=self.WCSs[frame_index], label="overlays")

        norm = simple_norm(self.flux[frame_index].ravel(), "asinh", percent=95)

        bar = ax.imshow(
            self.flux_2d[frame_index],
            norm=norm,
            cmap=plt.cm.viridis,
            origin="lower",
            rasterized=True,
        )
        plt.colorbar(bar, ax=ax, shrink=0.7, label=r"Flux ($e^{-}s^{-1}$)")
        ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.set_xlim(self.column.min() - 5, self.column.max() + 5)
        ax.set_ylim(self.row.min() - 5, self.row.max() + 5)

        ax.set_title(
            f"{self.meta['MISSION']} | {self.meta['DETECTOR']} | {self.meta['FILTER']}\n"
            f"Frame {frame_index} | JD {self.time[frame_index]} "
        )

        pix_coord = (
            self.WCSs[frame_index]
            .all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0.0)
            .T
        )

        if sources:
            ax.scatter(
                pix_coord[0],
                pix_coord[1],
                c="tab:red",
                facecolors="none",
                marker="o",
                s=10,
                linewidths=0.1,
                alpha=0.8,
            )

        ax.set_aspect("equal", adjustable="box")

        return ax

    def plot_pixel_masks(self, ax=None):
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
            fig, ax = plt.subplots(1, figsize=(10, 10))
        if hasattr(self, "non_bright_source_mask"):
            ax.scatter(
                self.column_2d.ravel()[~self.non_bright_source_mask],
                self.row_2d.ravel()[~self.non_bright_source_mask],
                c="y",
                marker="s",
                s=1,
                label="bright mask",
            )
        if hasattr(self, "non_sat_pixel_mask"):
            ax.scatter(
                self.column_2d.ravel()[~self.non_sat_pixel_mask],
                self.row_2d.ravel()[~self.non_sat_pixel_mask],
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


    def get_lightcurves(self, mode: str = "lk"):
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

            self.lcs = lk.LightCurveCollection(self.lcs)
        elif mode == "table":
            raise NotImplementedError


def _load_file(
    fname: list,
    cutout_size: Optional[int]=None,
    cutout_origin: Optional[list]=[0, 0],
):
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
    cutout_origin: tuple
        Coordinates of the origin of the cut out

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
    row_2d : numpy.ndarray
        Array with 2D (image) representation of pixel row
    meta : dict
        Dictionary with metadata
    """
    if not isinstance(fname, (list, np.ndarray)):
        fname = np.sort([fname])
    flux = []
    flux_err = []
    ra_3d = []
    dec_3d = []
    wcss = []
    times = []
    quality_mask = []
    cadno = []

    # WFI ccd science image size
    rmin = 0
    rmax = 250
    cmin = 0
    cmax = 250

    # set starting pixel
    rmin += cutout_origin[0]
    cmin += cutout_origin[1]

    if (rmin > rmax) | (cmin > cmax):
        raise ValueError("`cutout_origin` must be within the image.")
    if cutout_size is not None:
        rmax = np.min([rmin + cutout_size, rmax])
        cmax = np.min([cmin + cutout_size, cmax])

    for k, f in tqdm(enumerate(fname), total=len(fname)):
        if not os.path.isfile(f):
            raise FileNotFoundError("FFI calibrated fits file does not exist: ", f)
        aux = fitsio.FITS(f)

        flux.append(aux[0][rmin:rmax, cmin:cmax].T)
        flux_err.append(aux[1][rmin:rmax, cmin:cmax].T)

        times.append((aux[0].read_header()["TEND"] + aux[0].read_header()["TSTART"]) / 2)
        quality_mask.append(0)
        cadno.append(k)

        dims = flux[-1].shape
        if k == 0:
            row_2d, col_2d = np.mgrid[0 : dims[0], 0 : dims[1]]
        wcss.append(WCS(aux[0].read_header()))
        radec = (
            wcss[-1].all_pix2world(np.array([col_2d.ravel(), row_2d.ravel()]).T, 0.0).T
        )
        ra_3d.append(radec[0].reshape(dims))
        dec_3d.append(radec[1].reshape(dims))

    flux = np.array(flux)
    flux_err = np.array(flux_err)
    times = np.array(times)
    quality_mask = np.array(quality_mask)
    cadno = np.array(cadno)
    ra_3d = np.array(ra_3d)
    dec_3d = np.array(dec_3d)

    meta = {
        "MISSION": "Roman-Sim",
        "TELESCOP": "Roman",
        "SOFTWARE": aux[0].read_header()["SOFTWARE"],
        "RADESYS": aux[0].read_header()["RADESYS"],
        "EQUINOX": aux[0].read_header()["EQUINOX"],
        "FILTER": aux[0].read_header()["FILTER"],
        "FIELD": 1,
        "DETECTOR": aux[0].read_header()["DETECTOR"],
        "EXPOSURE": aux[0].read_header()["EXPOSURE"],
        "READMODE": "ramp",
    }

    return (
        wcss,
        times,
        cadno,
        flux,
        flux_err,
        ra_3d,
        dec_3d,
        col_2d,
        row_2d,
        meta,
        quality_mask,
    )
