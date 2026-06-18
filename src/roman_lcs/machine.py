"""
Defines the main Machine object that fit a mean PRF model to sources
"""

from typing import Optional

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
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
        time: npt.ArrayLike,
        flux: npt.ArrayLike,
        flux_err: npt.ArrayLike,
        ra: npt.ArrayLike,
        dec: npt.ArrayLike,
        sources: pd.DataFrame,
        column: npt.ArrayLike,
        row: npt.ArrayLike,
        time_mask: Optional[npt.ArrayLike] = None,
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
        n_r_knots: int
            Number of radial knots in the spline model.
        n_phi_knots: int
            Number of azimuthal knots in the spline model.
        time_nknots: int
            Number og knots for cartesian DM in time model.
        time_resolution: int
            Number of time points to bin by when fitting for velocity aberration.
        time_radius: float
            The radius around sources, out to which the velocity aberration model
            will be fit. (arcseconds)
        rmin: float
            The minimum radius for the PRF model to be fit. (arcseconds)
        rmax: float
            The maximum radius for the PRF model to be fit. (arcseconds)
        cut_r : float
            Radius distance whithin the shape model only depends on radius and not
            angle.
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
        dra: numpy.ndarray
            Distance in right ascension between pixel and source coordinates, units of
            degrees
        ddec: numpy.ndarray
            Distance in declination between pixel and source coordinates, units of
            degrees
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
        mean_model: scipy.sparce.csr_matrix
            Mean PSF model values per pixel used for PSF photometry
        cartesian_knot_spacing: string
            Defines the type of spacing between knots in cartessian space to generate
            the design matrix, options are "linear" or "sqrt".
        quiet: booleans
            Quiets TQDM progress bars.
        contaminant_flux_limit: float
          The limiting magnitude at which a sources is considered as contaminant
        """

        if not isinstance(sources, pd.DataFrame):
            raise TypeError("<sources> must be a of class Pandas Data Frame")

        # assigning initial attributes
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.ra = ra
        self.dec = dec
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
            * u.deg
        ).to(u.arcsecond)

        self.source_flux_estimates = np.copy(self.sources[sources_flux_column].values)

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
        sparse_mask = sparse.vstack(sparse_mask, "csr")
        sparse_mask.eliminate_zeros()

        # convertion to polar coordinates. We can't apply np.hypot or np.arctan2 to
        # sparse arrays. We keep track of non-zero index, do math in numpy space,
        # then rebuild r, phi as sparse.
        nnz_inds = sparse_mask.nonzero()
        # convert radial dist to arcseconds
        r_vals = np.hypot(self.dcolumn.data, self.drow.data) * 3600
        phi_vals = np.arctan2(self.drow.data, self.dcolumn.data)
        self.r = sparse.csr_matrix(
            (r_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask.shape,
            dtype=float,
        )
        self.phi = sparse.csr_matrix(
            (phi_vals, (nnz_inds[0], nnz_inds[1])),
            shape=sparse_mask.shape,
            dtype=float,
        )
        del r_vals, phi_vals, nnz_inds, sparse_mask
        return

    # def _get_source_mask(
    #     self,
    #     source_flux_limit: float = 1,
    #     reference_frame: int = 0,
    #     iterations: int = 2,
    #     plot: bool = False,
    # ) -> Optional[Any]:
    #     """
    #     Find the round pixel mask that identifies pixels with contributions from ANY of source.
    #     The source mask is created from one frame, then with `self.radius` it can be updated
    #     to other frames with different coordinate grids using `self._update_source_mask()`

    #     Firstly, makes a `rough_mask` that is 1 arcsec in radius. Then fits a simple
    #     linear trend in radius and flux. Uses this linear trend to identify pixels
    #     that are likely to be over the flux limit, the `source_mask`.

    #     We then iterate, masking out contaminated pixels in the `source_mask`, to get a better fit
    #     to the simple linear trend.

    #     Parameters
    #     ----------
    #     source_flux_limit: float
    #         Lower limit at which the source flux meets the background level
    #     iterations: int
    #         Number of iterations to fit polynomial
    #     plot: boolean
    #         Make a diagnostic plot
    #     """
    #     # make sure delta arrays are from the reference frame.
    #     # self._update_delta_arrays(frame_index=reference_frame)
    #     self.radius = 4 * self.pixel_scale.to(u.arcsecond).value
    #     if not sparse.issparse(self.r):
    #         self.rough_mask = sparse.csr_matrix(self.r.value < self.radius)
    #     else:
    #         self.rough_mask = sparse_lessthan(self.r, self.radius)
    #     self.source_mask = self.rough_mask.copy()
    #     self.source_mask.eliminate_zeros()
    #     # self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)
    #     self._get_uncontaminated_pixel_mask()

    #     for _ in range(iterations):
    #         mask = self.uncontaminated_source_mask
    #         r = mask.multiply(self.r).data
    #         max_f = np.log10(
    #             mask.astype(float)
    #             .multiply(self.flux[reference_frame])
    #             .multiply(1 / self.source_flux_estimates[:, None])
    #             .data
    #         )
    #         if sparse.issparse(self.r):
    #             rbins = np.linspace(0, self.r.data.max(), 100)
    #         else:
    #             rbins = np.linspace(0, self.r.value.max(), 100)
    #         masks = np.asarray(
    #             [
    #                 (r > rbins[idx]) & (r <= rbins[idx + 1])
    #                 for idx in range(len(rbins) - 1)
    #             ]
    #         )
    #         fbins = np.asarray([np.nanpercentile(max_f[m], 25) for m in masks])
    #         fbins_e = np.asarray([np.nanstd(max_f[m]) for m in masks])
    #         rbins = rbins[1:] - np.median(np.diff(rbins))
    #         k = np.isfinite(fbins)
    #         if not k.any():
    #             raise ValueError("Can not find source mask")
    #         pol = np.polyfit(rbins[k], fbins[k], deg=1, w=fbins_e[k])

    #         if sparse.issparse(self.r):
    #             mean_model = self.r.copy()
    #             mean_model.data = 10 ** np.polyval(pol, mean_model.data)
    #             self.source_mask = (
    #                 mean_model.multiply(self.source_flux_estimates[:, None])
    #             ) > source_flux_limit
    #         else:
    #             mean_model = 10 ** np.polyval(pol, self.r.value)
    #             self.source_mask = (
    #                 sparse.csr_matrix(mean_model * self.source_flux_estimates[:, None])
    #                 > source_flux_limit
    #             )
    #         self.uncontaminated_source_mask = _find_uncontaminated_pixels(
    #             self.source_mask
    #         )

    #     self.radius = self.source_mask.multiply(self.r).max(axis=1).toarray().ravel()
    #     self.radius[self.radius < self.pixel_scale.value] = (
    #         self.pixel_scale.value * 1.5
    #     )
    #     if sparse.issparse(self.r):
    #         self.source_mask = sparse_lessthan(self.r, self.radius)
    #     else:
    #         self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])
    #     self._get_uncontaminated_pixel_mask()

    #     if plot:
    #         if sparse.issparse(self.r):
    #             rdata = self.r.data
    #             mmdata = mean_model.data
    #             mmdata2 = mean_model.multiply(self.source_flux_estimates[:, None]).data
    #         else:
    #             rdata = self.r.value.ravel()
    #             mmdata = mean_model.ravel()
    #             mmdata2 = (mean_model * self.source_flux_estimates[:, None]).ravel()

    #         fig, ax = plt.subplots(1, 3, figsize=(15, 3))

    #         ax[0].set_title("All Sources Radius")
    #         ax[0].scatter(
    #             r, 10**max_f, s=2, alpha=0.5, label="Pixel data", rasterized=True
    #         )
    #         ax[0].scatter(rdata, mmdata, s=2, label="Mean Model", rasterized=True)
    #         ax[0].legend(loc="upper right")
    #         ax[0].set_xlim(rbins[k].min() - 0.04, rbins[k].max() + 0.04)
    #         ax[0].set_ylim(-0.05, 1.5)
    #         ax[0].set_xlabel("r [arcsec]")
    #         ax[0].set_ylabel("Normalized flux")

    #         ax[1].set_title("Binned Flux Source Profile")
    #         ax[1].errorbar(rbins[k], fbins[k], yerr=fbins_e[k], label="Data")
    #         ax[1].plot(rbins[k], np.polyval(pol, rbins[k]), label="Polynomial")
    #         ax[1].legend(loc="upper right")
    #         ax[1].set_xlabel("r [arcsec]")
    #         ax[1].set_ylabel("Normalized Log Flux")

    #         ax[2].set_title("Evaluated Source Radius")
    #         ax[2].scatter(
    #             rdata,
    #             mmdata2,
    #             s=1,
    #             alpha=0.6,
    #             label="Evaluated pixel flux",
    #             rasterized=True,
    #         )
    #         ax[2].axhline(
    #             source_flux_limit,
    #             c="tab:red",
    #             zorder=50000,
    #             label="Source flux limit",
    #             rasterized=True,
    #         )
    #         ax[2].legend(loc="upper right")
    #         ax[2].set_yscale("log")
    #         ax[2].set_xlim(-0.1, 1.2)
    #         ax[2].set_ylim(0.1, 1e4)
    #         ax[2].set_xlabel("r [arcsec]")
    #         ax[2].set_ylabel("Flux [e-/s]")

    #         return fig
    #     return

    # def _update_source_mask(
    #     self, frame_index: int = 0, source_flux_limit: float = 1
    # ) -> None:
    #     """
    #     Update source mask using self.radius when the ra,dec coordinate grid changes

    #     Parameters
    #     ----------
    #     rame_index : list or str
    #         Framce index used for ra and dec coordinate grid
    #     """
    #     # check if surce radius exist, if not, we run source_mask first
    #     if not hasattr(self, "radius"):
    #         self._get_source_mask(
    #             source_flux_limit=source_flux_limit, reference_frame=0
    #         )

    #     # update delta arrays to use the asked frame
    #     self._update_delta_arrays(frame_index=frame_index)

    #     # update the source mask and uncontaminated pixels
    #     if sparse.issparse(self.r):
    #         self.source_mask = sparse_lessthan(self.r, self.radius)
    #     else:
    #         self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])
    #     self._get_uncontaminated_pixel_mask()

    #     return

    # def _get_uncontaminated_pixel_mask(self) -> None:
    #     """
    #     creates a mask of shape nsources x npixels where targets are not contaminated.
    #     This mask is used to select pixels to build the PSF model.
    #     """

    #     # we flag sources fainter than mag_limit as non-contaminant
    #     if isinstance(self.contaminant_flux_limit, (float, int)):
    #         aux = self.source_mask.multiply(
    #             self.source_flux_estimates[:, None] > self.contaminant_flux_limit
    #         )
    #         aux.eliminate_zeros()
    #         self.uncontaminated_source_mask = aux.multiply(
    #             np.asarray(aux.sum(axis=0) == 1)[0]
    #         ).tocsr()
    #     # all sources are accounted for contamination
    #     else:
    #         self.uncontaminated_source_mask = self.source_mask.multiply(
    #             np.asarray(self.source_mask.sum(axis=0) == 1)[0]
    #         ).tocsr()

    #     # have to remove leaked zeros
    #     self.uncontaminated_source_mask.eliminate_zeros()
    #     return

    # def _get_centroids(self):
    #     """
    #     Find the ra and dec centroid of the image, at each time.
    #     """
    #     # centroids are astropy quantities
    #     self.ra_centroid = np.zeros(self.nt)
    #     self.dec_centroid = np.zeros(self.nt)
    #     self.ra_centroid_err = np.zeros(self.nt)
    #     self.dec_centroid_err = np.zeros(self.nt)
    #     for t in range(self.nt):
    #         # update sparse arrays due to offsets
    #         self._update_delta_arrays(frame_index=t)

    #         dra_m = self.source_mask.multiply(self.dra).data
    #         ddec_m = self.source_mask.multiply(self.ddec).data
            
    #         wgts = self.source_mask.multiply(
    #             np.sqrt(np.abs(self.flux[t]))
    #         ).data
    #         # mask out non finite values and background pixels
    #         k = (np.isfinite(wgts)) & (
    #             self.source_mask.multiply(self.flux[t]).data > 10
    #         )
    #         self.ra_centroid[t] = np.average(dra_m[k], weights=wgts[k])
    #         self.dec_centroid[t] = np.average(ddec_m[k], weights=wgts[k])
    #         self.ra_centroid_err[t] = weighted_std(dra_m[k], weights=wgts[k], ddof=0)
    #         self.dec_centroid_err[t] = weighted_std(ddec_m[k], weights=wgts[k], ddof=0)
    #     del dra_m, ddec_m
    #     self.ra_centroid = (self.ra_centroid*u.deg).to("arcsec")
    #     self.dec_centroid = (self.dec_centroid*u.deg).to("arcsec")
    #     self.ra_centroid_err = (self.ra_centroid_err*u.deg).to("arcsec")
    #     self.dec_centroid_err = (self.dec_centroid_err*u.deg).to("arcsec")

    # def _update_source_mask_remove_bkg_pixels(
    #     self, flux_cut_off: float = 1, frame_index: Union[str, int] = "mean"
    # ) -> None:
    #     """
    #     Update the `source_mask` to remove pixels that do not contribuite to the PRF
    #     shape.
    #     First, re-estimate the source flux usign the precomputed `mean_model`.
    #     This re-estimation is used to remove sources with bad prediction and update
    #     the `source_mask` by removing background pixels that do not contribuite to
    #     the PRF shape.
    #     Pixels with normalized flux > `flux_cut_off` are kept.

    #     Parameters
    #     ----------
    #     flux_cut_off : float
    #         Lower limit for the normalized flux predicted from the mean model.
    #     frame_index : string or int
    #         The frame index to be used, if "mean" then use the
    #         mean value across time
    #     """

    #     # Re-estimate source flux
    #     # -----
    #     prior_mu = self.source_flux_estimates
    #     prior_sigma = (
    #         np.ones(self.mean_model.shape[0]) * 10 * self.source_flux_estimates
    #     )

    #     if frame_index == "mean":
    #         f = self.flux.mean(axis=0)
    #         # fe = (self.flux_err **2 ).sum(axis=0) ** 0.5 / self.nt
    #     elif isinstance(frame_index, (int, np.int32, np.int64)):
    #         f = self.flux[frame_index]
    #         # fe = self.flux_err[frame_index]

    #     X = self.mean_model.copy()
    #     X = X.T

    #     sigma_w_inv = X.T.dot(X.multiply(1 / 1)).toarray()
    #     sigma_w_inv += np.diag(1 / (prior_sigma**2))
    #     B = X.T.dot((f / 1))
    #     B += prior_mu / (prior_sigma**2)
    #     ws = np.linalg.solve(sigma_w_inv, B)
    #     werrs = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

    #     # -----

    #     # Rebuild source mask
    #     ok = np.abs(ws - self.source_flux_estimates) / werrs > 3
    #     ok &= ((ws / self.source_flux_estimates) < 10) & (
    #         (self.source_flux_estimates / ws) < 10
    #     )
    #     ok &= ws > 10
    #     ok &= werrs > 0

    #     self.source_flux_estimates[ok] = ws[ok]

    #     self.source_mask = (
    #         self.mean_model.multiply(
    #             self.mean_model.T.dot(self.source_flux_estimates)
    #         ).tocsr()
    #         > flux_cut_off
    #     )

    #     # Recreate uncontaminated mask
    #     self._get_uncontaminated_pixel_mask()
    #     # self.uncontaminated_source_mask = self.uncontaminated_source_mask.multiply(
    #     #    (self.mean_model.max(axis=1) < 1)
    #     # )

    #     # create the final normalized mean model!
    #     # self._get_normalized_mean_model()
    #     self._get_mean_model()
    #     self.flux_cut_off = flux_cut_off
