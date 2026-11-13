"""
Defines the main Machine object that handles a stack of images and catalog of sources.
"""

from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy import sparse
from tqdm import tqdm


class Machine(object):
    """
    Class for calculating fast PRF photometry on a collection of images and
    a list of in image sources.

    This method is discussed in detail in
    [Hedges et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210608411H/abstract).

    This method solves a linear model to assuming Gaussian priors on the weight of
    each linear components as explained by
    [Luger, Foreman-Mackey & Hogg, 2017](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)
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
        wcs: Optional[list[WCS]] = None,
        time_mask: Optional[np.ndarray] = None,
        sparse_dist_lim: float = 4,
        sources_flux_column: str = "flux",
    ) -> None:
        """
        Parameters
        ----------
        time: numpy.ndarray
            Time values in JD
        flux: numpy.ndarray
            Flux values at each pixels and times in units of electrons / sec
        flux_err: numpy.ndarray
            Flux error values at each pixels and times in units of electrons / sec
        ra: numpy.ndarray
            Right Ascension coordinate of each pixel
        dec: numpy.ndarray
            Declination coordinate of each pixel
        sources: pandas.DataFrame
            DataFrame with source present in the images
        column: np.ndarray
            Data array containing the "columns" of the detector that each pixel is on.
        row: np.ndarray
            Data array containing the "rows" of the detector that each pixel is on.
        time_mask:  np.ndarray of booleans
            A boolean array of shape time. Only values where this mask is `True`
            will be used to calculate the average image for fitting the PSF.
            Use this to e.g. select frames with low VA, or no focus change
        sparse_dist_lim : float
            Radial distance used to include pixels around sources when creating delta
            arrays (dra, ddec, r, and phi) as sparse matrices for efficiency.
            Default is 40" (recommended for kepler). (arcseconds)
        sources_flux_column : str
            Column name in `sources` table to be used as flux estimate. For Kepler data
            gaia.phot_g_mean_flux is recommended, for TESS use gaia.phot_rp_mean_flux.

        Attributes
        ----------
        nsources: int
            Number of sources to be extracted
        nt: int
            Number of onservations in the time series (aka number of cadences)
        npixels: int
            Total number of pixels with flux measurements
        source_flux_estimates: numpy.ndarray
            First estimation of pixel fluxes assuming values given by the sources catalog
            (e.g. Gaia phot_g_mean_flux)
        dcolumn: numpy.ndarray
            Distance in the column direction between pixel and source coordinates, units of
            pixels
        drow: numpy.ndarray
            Distance in the row direction between pixel and source coordinates, units of
            pixels
        r: numpy.ndarray
            Radial distance between pixel and source coordinates (polar coordinates),
            in units of arcseconds
        phi: numpy.ndarray
            Angle between pixel and source coordinates (polar coordinates),
            in units of radians
        source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with pixels that contains flux from sources
        uncontaminated_source_mask: scipy.sparce.csr_matrix
            Sparce mask matrix with selected uncontaminated pixels per source to be used to
            build the PSF model
            the design matrix, options are "linear" or "sqrt".
        quiet: booleans
            Quiets TQDM progress bars.
        contaminant_flux_limit: float
            The limiting magnitude at which a sources is considered as contaminant
        WCSs : astropy.wcs
            World coordinates system solution for the FFI. Used for plotting.
        """

        if not isinstance(sources, pd.DataFrame):
            raise TypeError("<sources> must be a of class Pandas Data Frame")

        # assigning initial attributes
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ra = ra
        self.dec = dec
        self.WCSs = wcs
        self.sources = sources
        self.column = column
        self.row = row
        self.sparse_dist_lim = sparse_dist_lim
        # disble tqdm prgress bar when running in HPC
        self.quiet = False
        self.contaminant_flux_limit = None

        self.pixel_scale = (
            np.hypot(
                np.min(np.abs(np.diff(self.ra))), np.min(np.abs(np.diff(self.dec)))
            )
            * u.deg # type: ignore
        ).to(u.arcsec) # type: ignore

        # Ensure we have a plain numpy ndarray of floats (avoid pandas ExtensionArray)
        self.source_flux_estimates = self.sources[sources_flux_column].to_numpy(dtype=float)

        if time_mask is None:
            self.time_mask = np.ones(len(time), bool)
        else:
            self.time_mask = time_mask

        self.nsources = len(self.sources)
        self.nt = len(self.time)
        self.npixels = self.flux.shape[1]

        # self.ra_centroid, self.dec_centroid = np.zeros((2)) * u.deg
        self.is_sparse = self.nsources * self.npixels >= 2e5
        self._update_delta_arrays(frame_index=0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self) -> str:
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    def pixel_coordinates(self, frame_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the pixel coordinates for all sources in the catalog at a given frame

        Parameters
        ----------
        frame_index: int
            Frame index at which to compute th pixel coordinates

        Returns
        -------
        row, col: np.ndarray
            Row and column pixel coordinates of the sources in catalog.
        """
        if self.WCSs is None:
            raise ValueError("WCS information is not available")
        ROW, COL = (
            self.WCSs[frame_index]
            .all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0.0)
            .T
        )
        return ROW, COL

    def _update_delta_arrays(self, frame_index: int = 0) -> None:
        """
        Wrapper method to update dra, ddec, r and phi.

        Parameters
        ----------
        frame_index : list or str
            Frame index used for ra and dec coordinate grid
        """
        # Hardcoded: sparse implementation is efficient when nsourxes * npixels < 2e5
        # (JMP profile this)
        # https://github.com/SSDataLab/psfmachine/pull/17#issuecomment-866382898
        if self.is_sparse:
            self._update_delta_sparse_arrays(frame_index=frame_index)
        else:
            self._update_delta_numpy_arrays(frame_index=frame_index)

    def _update_delta_numpy_arrays(self, frame_index: int = 0) -> None:
        """
        Creates dra, ddec, r and phi numpy ndarrays .

        Parameters
        ----------
        frame_index : list or str
            Frame index used for ra and dec coordinate grid
        """
        # The distance in ra & dec from each source to each pixel
        # when centroid offset is 0 (i.e. first time creating arrays) create delta
        # arrays from scratch

        row, column = self.pixel_coordinates(frame_index=frame_index)

        self.dcolumn, self.drow = np.asarray(
            [
                [
                    self.column[frame_index] - column[idx],
                    self.row[frame_index] - row[idx],
                ]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)

        # conversion to polar coordinates
        # r is in arcseconds
        self.r = np.hypot(self.dcolumn, self.drow) * self.pixel_scale
        self.phi = np.arctan2(self.drow, self.dcolumn)
        return

    def _update_delta_sparse_arrays(self, frame_index: int = 0) -> None:
        """
        Creates dra, ddec, r and phi arrays as sparse arrays to be used for dense data,
        e.g. Kepler FFIs or cluster fields. Assuming that there is no flux information
        further than `dist_lim` for a given source, we only keep pixels within the
        `dist_lim`.
        dra, ddec, ra, and phi are unitless because they are `sparse.csr_matrix`. But
        keep same scale as '_create_delta_arrays()'.
        dra and ddec in deg. r in arcseconds and phi in rads

        Parameters
        ----------
        frame_index : list or str
            Frame index used for ra and dec coordinate grid
        """
        # iterate over sources to only keep pixels within self.sparse_dist_lim
        # this is inefficient, could be done in a tiled manner? only for squared data
        row, column = self.pixel_coordinates(frame_index=frame_index)
        dcol, drow, sparse_mask = [], [], []
        for i in tqdm(
            range(len(self.sources)),
            desc="Creating delta arrays",
            disable=self.quiet,
        ):
            dcol_aux = self.column[frame_index] - column[i]
            drow_aux = self.row[frame_index] - row[i]
            box_mask = sparse.csr_matrix(
                (np.abs(dcol_aux) <= self.sparse_dist_lim)
                & (np.abs(drow_aux) <= self.sparse_dist_lim)
            )
            dcol.append(box_mask.multiply(dcol_aux))
            drow.append(box_mask.multiply(drow_aux))
            sparse_mask.append(box_mask)

        del dcol_aux, drow_aux, box_mask
        # we stack dra, ddec of each object to create a [nsources, npixels] matrices
        self.dcolumn = sparse.vstack(dcol, "csr")
        self.drow = sparse.vstack(drow, "csr")
        sparse_mask_matrix = sparse.vstack(sparse_mask, "csr")
        sparse_mask_matrix.eliminate_zeros()

        # convertion to polar coordinates. We can't apply np.hypot or np.arctan2 to
        # sparse arrays. We keep track of non-zero index, do math in numpy space,
        # then rebuild r, phi as sparse.
        nnz_inds = sparse_mask_matrix.nonzero()
        # convert radial dist to arcseconds
        r_vals = np.hypot(self.dcolumn.data, self.drow.data) * 3600
        phi_vals = np.arctan2(self.drow.data, self.dcolumn.data)
        self.r = sparse.csr_matrix(
            (r_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask_matrix.shape,
            dtype=float,
        )
        self.phi = sparse.csr_matrix(
            (phi_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask_matrix.shape,
            dtype=float,
        )
        del r_vals, phi_vals, nnz_inds, sparse_mask_matrix
        return
