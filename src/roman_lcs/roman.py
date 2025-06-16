"""Subclass of `Machine` that Specifically work with FFIs"""

import os
from typing import Optional, Union, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.figure

import astropy.units as u
import lightkurve as lk
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from roman_cuts import RomanCuts
from scipy import ndimage

from . import __version__, log
from .machine import Machine


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
        n_r_knots: int = 9,
        n_phi_knots: int = 15,
        cut_r: float = 0.15,
        rmin: float = 0.02,
        rmax: float = 0.8,
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
        )
        if ra.shape[0] == 1:
            dithered = True
        elif row.shape[0] == 1:
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

    def _get_source_mask(
        self,
        source_flux_limit: float = 1,
        reference_frame: int = 0,
        iterations: int = 2,
        plot: bool = False,
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Adapted version of `machine._get_source_mask()` that masks out saturated and
        bright halo pixels in FFIs. See parameter descriptions in `Machine`.
        """
        fig = super()._get_source_mask(
            source_flux_limit=source_flux_limit,
            reference_frame=reference_frame,
            iterations=iterations,
            plot=plot,
        )
        # self._remove_bad_pixels_from_source_mask()
        return fig

    def _update_source_mask(
        self, frame_index: int = 0, source_flux_limit: float = 1
    ) -> None:
        super()._update_source_mask(
            frame_index=frame_index,
            source_flux_limit=source_flux_limit,
        )
        """
        Adapted version of `machine._update_source_mask()` that masks out saturated and
        bright halo pixels in FFIs. See parameter descriptions in `Machine`.
        """
        # self._remove_bad_pixels_from_source_mask()

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

    def build_shape_model(
        self,
        flux_cut_off: float = 1,
        frame_index: Union[str, int] = 0,
        bin_data: bool = False,
        plot: bool = False,
        **kwargs,
    ) -> Optional[matplotlib.figure.Figure]:
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
        return None

    def save_shape_model(self, output: Optional[str] = None) -> None:
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

    def load_shape_model(
        self,
        input: Optional[str] = None,
        plot: bool = False,
        source_flux_limit: float = 20,
        flux_cut_off: float = 0.01,
    ) -> Optional[matplotlib.figure.Figure]:
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
        if hdu[1].header["MISSION"].strip().lower() != self.meta["MISSION"].strip().lower():
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
        # self._update_source_mask_remove_bkg_pixels(flux_cut_off=flux_cut_off)
        # self._remove_bad_pixels_from_source_mask()

        if plot:
            return self.plot_shape_model(frame_index=self.ref_frame)
        return


    def plot_image(
        self,
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
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=self.WCSs[frame_index], label="overlays")

        norm = simple_norm(self.flux[frame_index].ravel(), "asinh", percent=95)

        bar = ax.pcolormesh(
            self.column_3d[frame_index],
            self.row_3d[frame_index],
            self.flux_3d[frame_index],
            norm=norm,
            cmap=plt.cm.viridis,
            rasterized=True,
        )
        plt.colorbar(bar, ax=ax, shrink=0.7, label=r"Flux ($e^{-}s^{-1}$)")
        ax.grid(True, which="major", axis="both", ls="-", color="w", alpha=0.7)
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.set_xlim(self.column[frame_index].min() - 4, self.column[frame_index].max() + 4)
        ax.set_ylim(self.row[frame_index].min() - 4, self.row[frame_index].max() + 4)

        ax.set_title(
            f"{self.meta['MISSION']} | {self.meta['DETECTOR']} | {self.meta['FILTER']}\n"
            f"Frame {self.cadenceno[frame_index]} | JD {self.time[frame_index]} "
        )

        srow, scol = (
            self.WCSs[frame_index]
            .all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0.0)
            .T
        )

        if sources:
            ax.scatter(
                scol,
                srow,
                c="tab:red",
                marker="o",
                s=12,
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
            fig, ax = plt.subplots(1, figsize=(10, 10))
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


def _load_file(
    fname: Union[str, List[str], np.ndarray],
    cutout_size: int = 32,
    cutout_center: Union[Tuple[int, int], Tuple[float, float]] = (0, 0),
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
    cutout_size: tuple of ints or floats
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
        raise ValueError("`cutout_center` must be a tuple of two values (row, column) or (RA, Dec).")
    if isinstance(cutout_center[0], (int, np.int32, np.int64)):
        rowcol = cutout_center
        radec = (None, None)
        dithered = False
    elif isinstance(cutout_center[0], (float, np.float32, np.float64)):
        radec = cutout_center
        rowcol = (None, None)
        dithered = True
    else:
        raise ValueError("`cutout_center` must be a tuple of two int values (row, column) or float (RA, Dec).")
    
    field = int(fname[0].split("_")[-5][5:])
    sca = int(fname[0].split("_")[-6][3:])
    filter = fname[0].split("_")[-7]
    rcube = RomanCuts(field=field, sca=sca, filter=filter, file_list=fname)
    rcube.make_cutout(rowcol=rowcol, radec=radec, size=(cutout_size, cutout_size), dithered=dithered)

    # put row,col and ra,dec into 3D arrasy [ntimes, axis1, axis2]
    if dithered:
        row_3d, col_3d = np.vstack(
            [[np.meshgrid(r, c, indexing="ij")]for r,c in zip(rcube.row, rcube.column)]
            ).transpose((1,0,2,3))
        ra_3d, dec_3d = rcube.wcss[0].all_pix2world(row_3d[0], col_3d[0], 0)
        ra_3d = np.atleast_3d(ra_3d).transpose((2,0,1))
        dec_3d = np.atleast_3d(dec_3d).transpose((2,0,1))
    else:
        row_3d, col_3d = np.meshgrid(rcube.row, rcube.column, indexing="ij")
        row_3d = np.atleast_3d(row_3d).transpose((2, 0, 1))
        col_3d = np.atleast_3d(col_3d).transpose((2, 0, 1))
        ra_3d, dec_3d = np.vstack(
            [[x.all_pix2world(row_3d[0], col_3d[0], 0)] for x in rcube.wcss]
            ).transpose((1,0,2,3))

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
