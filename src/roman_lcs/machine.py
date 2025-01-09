"""
Defines the main Machine object that fit a mean PRF model to sources
"""

from typing import Optional, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.stats import sigma_clip
from scipy import sparse, stats
from tqdm import tqdm

from . import __version__
from .utils import (
    _find_uncontaminated_pixels,
    _make_A_polar,
    solve_linear_model,
    sparse_lessthan,
    threshold_bin,
)


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
        n_r_knots: int = 10,
        n_phi_knots: int = 15,
        rmin: float = 0.1,
        rmax: float = 2,
        cut_r: float = 0.2,
        sparse_dist_lim: float = 4,
        sources_flux_column: str = "flux",
    ):
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
        self.n_r_knots = n_r_knots
        self.n_phi_knots = n_phi_knots
        self.rmin = rmin
        self.rmax = rmax
        self.cut_r = cut_r
        self.sparse_dist_lim = sparse_dist_lim * u.arcsecond
        self.cartesian_knot_spacing = "sqrt"
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

        self.ra_centroid, self.dec_centroid = np.zeros((2)) * u.deg

        self._update_delta_arrays(frame_index=0)

    @property
    def shape(self):
        return (self.nsources, self.nt, self.npixels)

    def __repr__(self):
        return f"Machine (N sources, N times, N pixels): {self.shape}"

    def pixel_coordinates(self, frame_index: int = 0):
        ROW, COL = (
            self.WCSs[frame_index]
            .all_world2pix(self.sources.loc[:, ["ra", "dec"]].values, 0.0)
            .T
        )
        return ROW, COL

    def _update_delta_arrays(self, frame_index: int = 0):
        """
        Wrapper method to update dra, ddec, r and phi.

        Parameters
        ----------
        rame_index : list or str
            Framce index used for ra and dec coordinate grid
        """
        # Hardcoded: sparse implementation is efficient when nsourxes * npixels < 2e5
        # (JMP profile this)
        # https://github.com/SSDataLab/psfmachine/pull/17#issuecomment-866382898
        if self.nsources * self.npixels < 2e5:
            self._updated_delta_numpy_arrays(frame_index=frame_index)
        else:
            self._update_delta_sparse_arrays(frame_index=frame_index)

    def _updated_delta_numpy_arrays(self, frame_index: int = 0):
        """
        Creates dra, ddec, r and phi numpy ndarrays .

        Parameters
        ----------
        rame_index : list or str
            Framce index used for ra and dec coordinate grid
        """
        # The distance in ra & dec from each source to each pixel
        # when centroid offset is 0 (i.e. first time creating arrays) create delta
        # arrays from scratch

        self.dra, self.ddec = np.asarray(
            [
                [
                    self.ra[frame_index] - self.sources["ra"][idx],
                    self.dec[frame_index] - self.sources["dec"][idx],
                ]
                for idx in range(len(self.sources))
            ]
        ).transpose(1, 0, 2)
        self.dra = self.dra * (u.deg)
        self.ddec = self.ddec * (u.deg)

        # convertion to polar coordinates
        self.r = np.hypot(self.dra, self.ddec).to("arcsec")
        self.phi = np.arctan2(self.ddec, self.dra)

    def _update_delta_sparse_arrays(self, frame_index: int = 0):
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
        rame_index : list or str
            Framce index used for ra and dec coordinate grid
        """
        # iterate over sources to only keep pixels within self.sparse_dist_lim
        # this is inefficient, could be done in a tiled manner? only for squared data
        dra, ddec, sparse_mask = [], [], []
        for i in tqdm(
            range(len(self.sources)),
            desc="Creating delta arrays",
            disable=self.quiet,
        ):
            dra_aux = self.ra[frame_index] - self.sources["ra"].iloc[i]
            ddec_aux = self.dec[frame_index] - self.sources["dec"].iloc[i]
            box_mask = sparse.csr_matrix(
                (np.abs(dra_aux) <= self.sparse_dist_lim.to("deg").value)
                & (np.abs(ddec_aux) <= self.sparse_dist_lim.to("deg").value)
            )
            dra.append(box_mask.multiply(dra_aux))
            ddec.append(box_mask.multiply(ddec_aux))
            sparse_mask.append(box_mask)

        del dra_aux, ddec_aux, box_mask
        # we stack dra, ddec of each object to create a [nsources, npixels] matrices
        self.dra = sparse.vstack(dra, "csr")
        self.ddec = sparse.vstack(ddec, "csr")
        sparse_mask = sparse.vstack(sparse_mask, "csr")
        sparse_mask.eliminate_zeros()

        # convertion to polar coordinates. We can't apply np.hypot or np.arctan2 to
        # sparse arrays. We keep track of non-zero index, do math in numpy space,
        # then rebuild r, phi as sparse.
        nnz_inds = sparse_mask.nonzero()
        # convert radial dist to arcseconds
        r_vals = np.hypot(self.dra.data, self.ddec.data) * 3600
        phi_vals = np.arctan2(self.ddec.data, self.dra.data)
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

    def _get_source_mask(
        self,
        source_flux_limit: float = 1,
        reference_frame: int = 0,
        iterations: int = 2,
        plot: bool = False,
    ):
        """
        Find the round pixel mask that identifies pixels with contributions from ANY of source.
        The source mask is created from one frame, then with `self.radius` it can be updated
        to other frames with different coordinate grids using `self._update_source_mask()`

        Firstly, makes a `rough_mask` that is 1 arcsec in radius. Then fits a simple
        linear trend in radius and flux. Uses this linear trend to identify pixels
        that are likely to be over the flux limit, the `source_mask`.

        We then iterate, masking out contaminated pixels in the `source_mask`, to get a better fit
        to the simple linear trend.

        Parameters
        ----------
        source_flux_limit: float
            Lower limit at which the source flux meets the background level
        iterations: int
            Number of iterations to fit polynomial
        plot: boolean
            Make a diagnostic plot
        """
        # make sure delta arrays are from the reference frame.
        self._update_delta_arrays(frame_index=reference_frame)
        self.radius = 3 * self.pixel_scale.to(u.arcsecond).value
        if not sparse.issparse(self.r):
            self.rough_mask = sparse.csr_matrix(self.r.value < self.radius)
        else:
            self.rough_mask = sparse_lessthan(self.r, self.radius)
        self.source_mask = self.rough_mask.copy()
        self.source_mask.eliminate_zeros()
        self.uncontaminated_source_mask = _find_uncontaminated_pixels(self.source_mask)

        for _ in range(iterations):
            mask = self.uncontaminated_source_mask
            r = mask.multiply(self.r).data
            max_f = np.log10(
                mask.astype(float)
                .multiply(self.flux[reference_frame])
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
            if sparse.issparse(self.r):
                rbins = np.linspace(0, self.r.data.max(), 100)
            else:
                rbins = np.linspace(0, self.r.value.max(), 100)
            masks = np.asarray(
                [
                    (r > rbins[idx]) & (r <= rbins[idx + 1])
                    for idx in range(len(rbins) - 1)
                ]
            )
            fbins = np.asarray([np.nanpercentile(max_f[m], 25) for m in masks])
            rbins = rbins[1:] - np.median(np.diff(rbins))
            k = np.isfinite(fbins)
            if not k.any():
                raise ValueError("Can not find source mask")
            l = np.polyfit(rbins[k], fbins[k], 1)

            if sparse.issparse(self.r):
                mean_model = self.r.copy()
                mean_model.data = 10 ** np.polyval(l, mean_model.data)
                self.source_mask = (
                    mean_model.multiply(self.source_flux_estimates[:, None])
                ) > source_flux_limit
            else:
                mean_model = 10 ** np.polyval(l, self.r.value)
                self.source_mask = (
                    sparse.csr_matrix(mean_model * self.source_flux_estimates[:, None])
                    > source_flux_limit
                )
            self.uncontaminated_source_mask = _find_uncontaminated_pixels(
                self.source_mask
            )

        self.radius = self.source_mask.multiply(self.r).max(axis=1).toarray().ravel()
        self.radius[self.radius < self.pixel_scale.value] = (
            self.pixel_scale.value * 1.25
        )
        if sparse.issparse(self.r):
            self.source_mask = sparse_lessthan(self.r, self.radius)
        else:
            self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])
        self._get_uncontaminated_pixel_mask()

        if plot:
            if sparse.issparse(self.r):
                rdata = self.r.data
                mmdata = mean_model.data
                mmdata2 = mean_model.multiply(self.source_flux_estimates[:, None]).data
            else:
                rdata = self.r.value.ravel()
                mmdata = mean_model.ravel()
                mmdata2 = (mean_model * self.source_flux_estimates[:, None]).ravel()

            fig, ax = plt.subplots(1, 3, figsize=(15, 3))

            ax[0].set_title("All Sources Radius")
            ax[0].scatter(
                r, 10**max_f, s=2, alpha=0.5, label="Pixel data", rasterized=True
            )
            ax[0].scatter(
                rdata, mmdata, s=2, label="Mean Model", rasterized=True
            )
            ax[0].legend(loc="upper right")
            ax[0].set_xlim(rbins[k].min() - 0.04, rbins[k].max() + 0.04)
            ax[0].set_ylim(-0.05, 1.5)
            ax[0].set_xlabel("r [arcsec]")
            ax[0].set_ylabel("Normalized flux")

            ax[1].set_title("Binned Flux Source Profile")
            ax[1].plot(rbins[k], fbins[k], label="Data")
            ax[1].plot(rbins[k], np.polyval(l, rbins[k]), label="Polynomial")
            ax[1].legend(loc="upper right")
            ax[1].set_xlabel("r [arcsec]")
            ax[1].set_ylabel("Normalized Log Flux")

            ax[2].set_title("Evaluated Source Radius")
            ax[2].scatter(
                rdata,
                mmdata2,
                s=1,
                alpha=0.6,
                label="Evaluated pixel flux",
                rasterized=True,
            )
            ax[2].axhline(
                source_flux_limit,
                c="tab:red",
                zorder=50000,
                label="Source flux limit",
                rasterized=True,
            )
            ax[2].legend(loc="upper right")
            ax[2].set_yscale("log")
            ax[2].set_xlim(-0.1, 1.2)
            ax[2].set_ylim(0.1, 1e4)
            ax[2].set_xlabel("r [arcsec]")
            ax[2].set_ylabel("Flux [e-/s]")
            
            return fig
        return

    def _update_source_mask(self, frame_index: int = 0, source_flux_limit: float = 1):
        """
        Update source mask using self.radius when the ra,dec coordinate grid changes

        Parameters
        ----------
        rame_index : list or str
            Framce index used for ra and dec coordinate grid
        """
        # check if surce radius exist, if not, we run source_mask first
        if not hasattr(self, "radius"):
            self._get_source_mask(
                source_flux_limit=source_flux_limit, reference_frame=0
            )

        # update delta arrays to use the asked frame
        self._update_delta_arrays(frame_index=frame_index)

        # update the source mask and uncontaminated pixels
        if sparse.issparse(self.r):
            self.source_mask = sparse_lessthan(self.r, self.radius)
        else:
            self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])
        self._get_uncontaminated_pixel_mask()

        return

    def _get_uncontaminated_pixel_mask(self):
        """
        creates a mask of shape nsources x npixels where targets are not contaminated.
        This mask is used to select pixels to build the PSF model.
        """

        # we flag sources fainter than mag_limit as non-contaminant
        if isinstance(self.contaminant_flux_limit, (float, int)):
            aux = self.source_mask.multiply(
                self.source_flux_estimates[:, None] > self.contaminant_flux_limit
            )
            aux.eliminate_zeros()
            self.uncontaminated_source_mask = aux.multiply(
                np.asarray(aux.sum(axis=0) == 1)[0]
            ).tocsr()
        # all sources are accounted for contamination
        else:
            self.uncontaminated_source_mask = self.source_mask.multiply(
                np.asarray(self.source_mask.sum(axis=0) == 1)[0]
            ).tocsr()

        # have to remove leaked zeros
        self.uncontaminated_source_mask.eliminate_zeros()
        return

    def _get_centroids(self):
        """
        Find the ra and dec centroid of the image, at each time.
        """
        raise NotImplementedError

    def build_shape_model(
        self,
        flux_cut_off: float = 1,
        frame_index: Union[str, int] = 0,
        bin_data: bool = False,
        plot: bool = False,
        **kwargs,
    ):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        flux_cut_off: float
            the flux in COUNTS at which to stop evaluating the model!
        frame_index : string or int
            The frame index used to build the shape model, if "mean" then use the
            mean value across time
        bin_data : boolean
            Bin flux data spatially to increase SNR before fitting the shape model
        plot : boolean
            Make a diagnostic plot
        **kwargs
            Keyword arguments to be passed to `_get_source_mask()`
        """

        # Mask of shape nsources x number of pixels, one where flux from a
        # source exists
        # if not hasattr(self, "source_mask"):
        self._update_source_mask(frame_index=frame_index, **kwargs)

        # for iter in range(niters):
        flux_estimates = self.source_flux_estimates[:, None]

        if frame_index == "mean":
            f = (self.flux[self.time_mask]).mean(axis=0)
            # fe = (self.flux_err[self.time_mask] ** 2).sum(axis=0) ** 0.5 / self.nt
        elif isinstance(frame_index, int):
            f = self.flux[frame_index]
            # fe = self.flux_err[frame_index]

        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(f)
            .multiply(1 / flux_estimates)
            .data
        )
        # Actual Kepler errors cause all sorts of instability
        # mean_f_err = (
        #     self.uncontaminated_source_mask.astype(float)
        #     .multiply(fe / (f * np.log(10)))
        #     .multiply(1 / flux_estimates)
        #     .data
        # )
        # We only need these weights for the wings, so we'll use poisson noise
        mean_f_err = (
            self.uncontaminated_source_mask.astype(float)
            .multiply((f**0.5) / (f * np.log(10)))
            .multiply(1 / flux_estimates)
            .data
        )
        mean_f_err.data = np.abs(mean_f_err.data)

        # take value from Quantity is not necessary
        phi_b = self.uncontaminated_source_mask.multiply(self.phi).data
        r_b = self.uncontaminated_source_mask.multiply(self.r).data

        if bin_data:
            # number of bins is hardcoded to work with FFI or TPFs accordingly
            # I found 30 wirks good with TPF stacks (<10000 pixels),
            # 90 with FFIs (tipically >50k pixels), and 60 in between.
            # this could be improved later if necessary
            nbins = (
                30 if mean_f.shape[0] <= 1e4 else (60 if mean_f.shape[0] <= 5e4 else 90)
            )
            _, phi_b, r_b, mean_f, mean_f_err = threshold_bin(
                phi_b,
                r_b,
                mean_f,
                z_err=mean_f_err,
                bins=nbins,
                abs_thresh=5,
            )

        # build a design matrix A with b-splines basis in radius and angle axis.
        A = _make_A_polar(
            phi_b.ravel(),
            r_b.ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        prior_sigma = np.ones(A.shape[1]) * 10
        prior_mu = np.zeros(A.shape[1]) - 10

        nan_mask = np.isfinite(mean_f.ravel())

        # we solve for A * psf_w = mean_f
        psf_w, psf_w_err = solve_linear_model(
            A,
            y=mean_f.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        bad = sigma_clip(mean_f.ravel() - A.dot(psf_w), sigma=5).mask

        psf_w, psf_w_err = solve_linear_model(
            A,
            y=mean_f.ravel(),
            #            y_err=mean_f_err.ravel(),
            k=nan_mask & ~bad,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        self.clean_pixel_mask = nan_mask & ~bad
        self.psf_w = psf_w
        self.psf_w_err = psf_w_err
        self.normalized_shape_model = False

        # We then build the same design matrix for all pixels with flux
        # this non-normalized mean model is temporary and used to re-create a better
        # `source_mask`
        self._get_mean_model()
        # remove background pixels and recreate mean model
        self._update_source_mask_remove_bkg_pixels(
            flux_cut_off=flux_cut_off, frame_index=frame_index
        )

        if plot:
            return self.plot_shape_model(frame_index=frame_index, bin_data=bin_data)
        return

    def _get_mean_model(self):
        """
        Convenience function to make the scene model
        """
        Ap = _make_A_polar(
            self.source_mask.multiply(self.phi).data,
            self.source_mask.multiply(self.r).data,
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )

        # And create a `mean_model` that has the psf model for all pixels with fluxes
        mean_model = sparse.csr_matrix(self.r.shape)
        m = 10 ** Ap.dot(self.psf_w)
        m[~np.isfinite(m)] = 0
        mean_model[self.source_mask] = m
        mean_model.eliminate_zeros()
        self.mean_model = mean_model

    def _update_source_mask_remove_bkg_pixels(
        self, flux_cut_off: float = 1, frame_index: Union[str, int] = "mean"
    ):
        """
        Update the `source_mask` to remove pixels that do not contribuite to the PRF
        shape.
        First, re-estimate the source flux usign the precomputed `mean_model`.
        This re-estimation is used to remove sources with bad prediction and update
        the `source_mask` by removing background pixels that do not contribuite to
        the PRF shape.
        Pixels with normalized flux > `flux_cut_off` are kept.

        Parameters
        ----------
        flux_cut_off : float
            Lower limit for the normalized flux predicted from the mean model.
        frame_index : string or int
            The frame index to be used, if "mean" then use the
            mean value across time
        """

        # Re-estimate source flux
        # -----
        prior_mu = self.source_flux_estimates
        prior_sigma = (
            np.ones(self.mean_model.shape[0]) * 10 * self.source_flux_estimates
        )

        if frame_index == "mean":
            f = self.flux.mean(axis=0)
            # fe = (self.flux_err **2 ).sum(axis=0) ** 0.5 / self.nt
        elif isinstance(frame_index, int):
            f = self.flux[frame_index]
            # fe = self.flux_err[frame_index]

        X = self.mean_model.copy()
        X = X.T

        sigma_w_inv = X.T.dot(X.multiply(1 / 1)).toarray()
        sigma_w_inv += np.diag(1 / (prior_sigma**2))
        B = X.T.dot((f / 1))
        B += prior_mu / (prior_sigma**2)
        ws = np.linalg.solve(sigma_w_inv, B)
        werrs = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

        # -----

        # Rebuild source mask
        ok = np.abs(ws - self.source_flux_estimates) / werrs > 3
        ok &= ((ws / self.source_flux_estimates) < 10) & (
            (self.source_flux_estimates / ws) < 10
        )
        ok &= ws > 10
        ok &= werrs > 0

        self.source_flux_estimates[ok] = ws[ok]

        self.source_mask = (
            self.mean_model.multiply(
                self.mean_model.T.dot(self.source_flux_estimates)
            ).tocsr()
            > flux_cut_off
        )

        # Recreate uncontaminated mask
        self._get_uncontaminated_pixel_mask()
        # self.uncontaminated_source_mask = self.uncontaminated_source_mask.multiply(
        #    (self.mean_model.max(axis=1) < 1)
        # )

        # create the final normalized mean model!
        # self._get_normalized_mean_model()
        self._get_mean_model()
        self.flux_cut_off = flux_cut_off

    def _get_normalized_mean_model(self, npoints=300, plot=False):
        """
        Renomarlize shape model to sum 1

        Parameters
        ----------
        npoints : int
            Number of points used to build a high resolution grid in polar coordinates
        plot : boolean
            Create a diagnostic plot
        """

        # create a high resolution polar grid
        r = self.source_mask.multiply(self.r).data
        phi_hd = np.linspace(-np.pi, np.pi, npoints)
        r_hd = np.linspace(0, r.max(), npoints)
        phi_hd, r_hd = np.meshgrid(phi_hd, r_hd)

        # high res DM
        Ap = _make_A_polar(
            phi_hd.ravel(),
            r_hd.ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        # evaluate the high res model
        mean_model_hd = Ap.dot(self.psf_w)
        mean_model_hd[~np.isfinite(mean_model_hd)] = np.nan
        mean_model_hd = mean_model_hd.reshape(phi_hd.shape)

        # mask out datapoint that don't contribuite to the psf
        mean_model_hd_ma = mean_model_hd.copy()
        mask = mean_model_hd > -3
        mean_model_hd_ma[~mask] = -np.inf
        mask &= ~((r_hd > 14) & (np.gradient(mean_model_hd_ma, axis=0) > 0))
        mean_model_hd_ma[~mask] = -np.inf

        # double integral using trapezoidal rule
        self.mean_model_integral = np.trapz(
            np.trapz(10**mean_model_hd_ma, r_hd[:, 0], axis=0),
            phi_hd[0, :],
            axis=0,
        )
        # renormalize weights and build new shape model
        if not self.normalized_shape_model:
            self.psf_w *= np.log10(self.mean_model_integral)
            self.normalized_shape_model = True
        self._get_mean_model()

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(9, 5))
            im = ax[0].scatter(
                phi_hd.ravel(),
                r_hd.ravel(),
                c=mean_model_hd_ma.ravel(),
                vmin=-3,
                vmax=-1,
                s=1,
                label=r"$\int = $" + f"{self.mean_model_integral:.4f}",
            )
            im = ax[1].scatter(
                r_hd.ravel() * np.cos(phi_hd.ravel()),
                r_hd.ravel() * np.sin(phi_hd.ravel()),
                c=mean_model_hd_ma.ravel(),
                vmin=-3,
                vmax=-1,
                s=1,
            )
            ax[0].legend()
            fig.colorbar(im, ax=ax, location="bottom")
            plt.show()

    def get_psf_metrics(self, npoints_per_pixel=10):
        """
        Computes three metrics for the PSF model:
            source_psf_fraction: the amount of PSF in the data. Tells how much of a
                sources is used to estimate the PSF, values are in between [0, 1].
            perturbed_ratio_mean: the ratio between the mean model and perturbed model
                for each source. Usefull to find when the time model affects the
                mean value of the light curve.
            perturbed_std: the standard deviation of the perturbed model for each
                source. USeful to find when the time model introduces variability in the
                light curve.

        If npoints_per_pixel > 0, it creates high npoints_per_pixel shape models for
        each source by dividing each pixels into a grid of
        [npoints_per_pixel x npoints_per_pixel]. This provides a better estimate of
        `source_psf_fraction`.

        Parameters
        ----------
        npoints_per_pixel : int
            Value in which each pixel axis is split to increase npoints_per_pixel.
            Default is 0 for no subpixel npoints_per_pixel.

        """
        if npoints_per_pixel > 0:
            # find from which observation (TPF) a sources comes
            obs_per_pixel = self.source_mask.multiply(self.pix2obs).tocsr()
            tpf_idx = []
            for k in range(self.source_mask.shape[0]):
                pix = obs_per_pixel[k].data
                mode = stats.mode(pix)[0]
                if len(mode) > 0:
                    tpf_idx.append(mode[0])
                else:
                    tpf_idx.append(
                        [x for x, ss in enumerate(self.tpf_meta["sources"]) if k in ss][
                            0
                        ]
                    )
            tpf_idx = np.array(tpf_idx)

            # get the pix coord for each source, we know how to increase resolution in
            # the pixel space but not in WCS
            row = self.source_mask.multiply(self.row).tocsr()
            col = self.source_mask.multiply(self.column).tocsr()
            mean_model_hd_sum = []
            # iterating per sources avoids creating a new super large `source_mask`
            # with high resolution, which a priori is hard
            for k in range(self.nsources):
                # find row, col combo for each source
                row_ = row[k].data
                col_ = col[k].data
                colhd, rowhd = [], []
                # pixels are divided into `resolution` - 1 subpixels
                for c, r in zip(col_, row_):
                    x = np.linspace(c - 0.5, c + 0.5, npoints_per_pixel + 1)
                    y = np.linspace(r - 0.5, r + 0.5, npoints_per_pixel + 1)
                    x, y = np.meshgrid(x, y)
                    colhd.extend(x[:, :-1].ravel())
                    rowhd.extend(y[:-1].ravel())
                colhd = np.array(colhd)
                rowhd = np.array(rowhd)
                # convert to ra, dec beacuse machine shape model works in sky coord
                rahd, dechd = self.tpfs[tpf_idx[k]].wcs.wcs_pix2world(
                    colhd - self.tpfs[tpf_idx[k]].column,
                    rowhd - self.tpfs[tpf_idx[k]].row,
                    0,
                )
                drahd = rahd - self.sources["ra"][k]
                ddechd = dechd - self.sources["dec"][k]
                drahd = drahd * (u.deg)
                ddechd = ddechd * (u.deg)
                rhd = np.hypot(drahd, ddechd).to("arcsec").value
                phihd = np.arctan2(ddechd, drahd).value
                # create a high resolution DM
                Ap = _make_A_polar(
                    phihd.ravel(),
                    rhd.ravel(),
                    rmin=self.rmin,
                    rmax=self.rmax,
                    cut_r=self.cut_r,
                    n_r_knots=self.n_r_knots,
                    n_phi_knots=self.n_phi_knots,
                )
                # evaluate the HD model
                modelhd = 10 ** Ap.dot(self.psf_w)
                # compute the model sum for source, how much of the source is in data
                mean_model_hd_sum.append(np.trapz(modelhd, dx=1 / npoints_per_pixel**2))

            # get normalized psf fraction metric
            self.source_psf_fraction = np.array(
                mean_model_hd_sum
            )  # / np.nanmax(mean_model_hd_sum)
        else:
            self.source_psf_fraction = np.array(self.mean_model.sum(axis=1)).ravel()

        # time model metrics
        if hasattr(self, "P"):
            perturbed_lcs = np.vstack(
                [
                    np.array(self.perturbed_model(time_index=k).sum(axis=1)).ravel()
                    for k in range(self.time.shape[0])
                ]
            )
            self.perturbed_ratio_mean = (
                np.nanmean(perturbed_lcs, axis=0)
                / np.array(self.mean_model.sum(axis=1)).ravel()
            )
            self.perturbed_std = np.nanstd(perturbed_lcs, axis=0)

    def plot_shape_model(self, frame_index="mean", bin_data=False):
        """
        Diagnostic plot of shape model.

        Parameters
        ----------
        frame_index : string or int
            The frame index used to plot the shape model, if "mean" then use the
            mean value across time
        bin_data : bool
            Bin or not the pixel data in a 2D historgram, default is False.

        Returns
        -------
        fig : matplotlib.Figure
            Figure.
        """
        if frame_index == "mean":
            mean_f = np.log10(
                self.uncontaminated_source_mask.astype(float)
                .multiply(self.flux[self.time_mask].mean(axis=0))
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )
        elif isinstance(frame_index, int):
            mean_f = np.log10(
                self.uncontaminated_source_mask.astype(float)
                .multiply(self.flux[frame_index])
                .multiply(1 / self.source_flux_estimates[:, None])
                .data
            )

        dx, dy = (
            self.uncontaminated_source_mask.multiply(self.dra),
            self.uncontaminated_source_mask.multiply(self.ddec),
        )
        dx = dx.data * u.deg.to(u.arcsecond)
        dy = dy.data * u.deg.to(u.arcsecond)

        radius = np.maximum(np.abs(dx).max(), np.abs(dy).max()) * 1.1
        vmin, vmax = np.nanpercentile(mean_f, [5, 93])

        if bin_data:
            nbins = 30 if mean_f.shape[0] <= 5e3 else 90
            _, dx, dy, mean_f, _ = threshold_bin(
                dx, dy, mean_f, bins=nbins, abs_thresh=5
            )

        fig, ax = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        im = ax[0, 0].scatter(
            dx,
            dy,
            c=mean_f,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[0, 0].set(
            ylabel=r'$\delta y$ ["]',
            title="Data (cadence %s)" % str(frame_index),
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        # arrow to show centroid offset correction
        if hasattr(self, "ra_centroid_avg"):
            ax[0, 0].arrow(
                0,
                0,
                self.ra_centroid_avg.to("arcsec").value,
                self.dec_centroid_avg.to("arcsec").value,
                width=1e-6,
                shape="full",
                head_width=0.05,
                head_length=0.1,
                color="tab:red",
            )

        phi, r = np.arctan2(dy, dx), np.hypot(dx, dy)
        im = ax[0, 1].scatter(
            phi, r, c=mean_f, cmap="viridis", vmin=vmin, vmax=vmax, s=3, rasterized=True
        )
        ax[0, 1].set(
            ylabel='$r$ ["]',
            title="Data (cadence %s)" % str(frame_index),
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )

        A = _make_A_polar(
            phi,
            r,
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )
        # if the mean model is normalized, we revert it only for plotting to make
        # easier the comparisson between the data and model.
        # the normalization is a multiplicative factor
        if self.normalized_shape_model:
            model = A.dot(self.psf_w / np.log10(self.mean_model_integral))
        else:
            model = A.dot(self.psf_w)
        im = ax[1, 1].scatter(
            phi,
            r,
            c=model,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[1, 1].set(
            ylabel=r'$r$ ["]',
            title="Model",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )
        im = ax[1, 0].scatter(
            dx,
            dy,
            c=model,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=3,
            rasterized=True,
        )
        ax[1, 0].set(
            ylabel=r'$\delta y$ ["]',
            title="Model",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        ax[0, 0].set_aspect("equal", adjustable="box")
        ax[1, 0].set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(im, ax=ax[:2, 1], shrink=0.7, location="right")
        cbar.set_label("log$_{10}$ Normalized Flux")
        mean_f = 10**mean_f
        model = 10**model

        # im = ax[2, 0].scatter(
        #     dx,
        #     dy,
        #     c=(model - mean_f) / mean_f,
        #     cmap="RdBu",
        #     vmin=-1,
        #     vmax=1,
        #     s=3,
        #     rasterized=True,
        # )
        # ax[2, 1].scatter(
        #     phi,
        #     r,
        #     c=(model - mean_f) / mean_f,
        #     cmap="RdBu",
        #     vmin=-1,
        #     vmax=1,
        #     s=3,
        #     rasterized=True,
        # )
        # ax[2, 0].set_aspect("equal", adjustable="box")
        ax[-1, 0].set(
            xlabel=r'$\delta x$ ["]',
            ylabel=r'$\delta y$ ["]',
            title="Residuals",
            xlim=(-radius, radius),
            ylim=(-radius, radius),
        )
        ax[-1, 1].set(
            xlabel=r"$\phi$ [$^\circ$]",
            ylabel='$r$ ["]',
            title="Residuals",
            ylim=(0, radius),
            yticks=np.linspace(0, radius, 5, dtype=int),
        )
        # cbar = fig.colorbar(im, ax=ax[2, 1], shrink=0.9, location="right")
        # cbar.set_label("(F$_M$ - F$_D$)/F$_D$")

        return fig

    def fit_model(self):
        """
        Finds the best fitting weights for every source, simultaneously

        Parameters
        ----------
        fit_va : boolean
            Fitting model accounting for velocity aberration. If `True`, then a time
            model has to be built previously with `build_time_model`.
        """

        prior_mu = self.source_flux_estimates  # np.zeros(A.shape[1])
        prior_sigma = (
            np.ones(self.mean_model.shape[0])
            * 5
            * np.abs(self.source_flux_estimates) ** 0.5
        )

        self.model_flux = np.zeros(self.flux.shape) * np.nan
        self.ws = np.zeros((self.nt, self.mean_model.shape[0]))
        self.werrs = np.zeros((self.nt, self.mean_model.shape[0]))

        for tdx in tqdm(
            range(self.nt),
            desc=f"Fitting {self.nsources} Sources (w. VA)",
            # disable=self.quiet,
        ):
            # update source mask for current frame
            self._update_source_mask(frame_index=tdx)
            self._get_mean_model()
            self._update_source_mask_remove_bkg_pixels(
                flux_cut_off=self.flux_cut_off, frame_index=tdx
            )
            X = self.mean_model.copy()
            X = X.T

            # sigma_w_inv = X.T.dot(
            #     X.multiply(1 / 1 ** 2)
            # ).toarray()
            # sigma_w_inv += np.diag(1 / (prior_sigma**2))
            # B = X.T.dot((self.flux[tdx] / 1 ** 2))
            # B += prior_mu / (prior_sigma**2)
            # self.ws[tdx] = np.linalg.solve(sigma_w_inv, np.nan_to_num(B))
            # self.werrs[tdx] = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
            try:
                self.ws[tdx], self.werrs[tdx] = solve_linear_model(
                    X,
                    self.flux[tdx],
                    y_err=self.flux_err[tdx],
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    errors=True,
                )
            except np.linalg.LinAlgError:
                print("WARNING: matrix is singular, trying without errors, this could lead to nans")
                self.ws[tdx], self.werrs[tdx] = solve_linear_model(
                    X,
                    self.flux[tdx],
                    # y_err=self.flux_err[tdx],
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    errors=True,
                )
            self.model_flux[tdx] = X.dot(self.ws[tdx])

        # check bad estimates
        nodata = np.asarray(self.mean_model.sum(axis=1))[:, 0] == 0
        # These sources are poorly estimated
        # nodata |= (self.mean_model.max(axis=1) > 1).toarray()[:, 0]
        self.ws[:, nodata] *= np.nan
        self.werrs[:, nodata] *= np.nan

        return
