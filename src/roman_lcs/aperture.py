"""
Collection of aperture utils lifted from
[Kepler-Apertures](https://github.com/jorgemarpa/kepler-apertures) and adapted to work
with PSFMachine.

Some this functions inputs and operate on a `Machine` object but we move them out of
`mahine.py` to keep the latter smowhow clean and short.

"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, sparse
from tqdm import tqdm


def optimize_aperture(
    psf_model,
    target_complete=0.9,
    target_crowd=0.9,
    max_iter=100,
    percentile_bounds=[0, 100],
    quiet=False,
):
    """
    Function to optimize the aperture mask for a given source.

    The optimization is done using scipy Brent's algorithm and it uses a custom
    loss function `goodness_metric_obj_fun` that uses a Leaky ReLU term to
    achive the target value for both metrics.

    Parameters
    ----------
    psf_model : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    target_complete : float
        Value of the target completeness metric.
    target_crowd : float
        Value of the target crowdeness metric.
    max_iter : int
        Numer of maximum iterations to be performed by the optimizer.
    percentile_bounds : tuple
        Tuple of minimun and maximun values for allowed percentile values during
        the optimization. Default is the widest range of [0, 100].

    Returns
    -------
    optimal_percentile : numpy.ndarray
        An array with the percentile value to defines the "optimal" aperture for
        each source.
    """
    # optimize percentile cut for every source
    optimal_percentile = []
    for sdx in tqdm(
        range(psf_model.shape[0]),
        desc="Optimizing apertures per source",
        disable=quiet,
    ):
        optim_params = {
            "percentile_bounds": percentile_bounds,
            "target_complete": target_complete,
            "target_crowd": target_crowd,
            "max_iter": max_iter,
            "psf_models": psf_model,
            "sdx": sdx,
        }
        minimize_result = optimize.minimize_scalar(
            goodness_metric_obj_fun,
            method="Bounded",
            bounds=percentile_bounds,
            options={"maxiter": max_iter, "disp": False},
            args=(optim_params),
        )
        optimal_percentile.append(minimize_result.x)
    return np.array(optimal_percentile)


def goodness_metric_obj_fun(percentile, optim_params):
    """
    The objective function to minimize with scipy.optimize.minimize_scalar called
    during optimization of the photometric aperture.

    Parameters
    ----------
    percentile : int
        Percentile of the normalized flux distribution that defines the isophote.
    optim_params : dictionary
        Dictionary with the variables needed to evaluate the metric:
            * psf_models
            * sdx
            * target_complete
            * target_crowd

    Returns
    -------
    penalty : float
        Value of the objective function to be used for optiization.
    """
    psf_models = optim_params["psf_models"]
    sdx = optim_params["sdx"]
    # Find the value where to cut
    cut = np.nanpercentile(psf_models[sdx].data, percentile)
    # create "isophot" mask with current cut
    mask = (psf_models[sdx] > cut).toarray()[0]

    # Do not compute and ignore if target score < 0
    if optim_params["target_complete"] > 0:
        # compute_FLFRCSAP returns an array of size 1 when doing only one source
        completMetric = compute_FLFRCSAP(psf_models[sdx], mask)[0]
    else:
        completMetric = 1.0

    # Do not compute and ignore if target score < 0
    if optim_params["target_crowd"] > 0:
        crowdMetric = compute_CROWDSAP(psf_models, mask, idx=sdx)
    else:
        crowdMetric = 1.0

    # Once we hit the target we want to ease-back on increasing the metric
    # However, we don't want to ease-back to zero pressure, that will
    # unconstrain the penalty term and cause the optmizer to run wild.
    # So, use a "Leaky ReLU"
    # metric' = threshold + (metric - threshold) * leakFactor
    leakFactor = 0.01
    if (
        optim_params["target_complete"] > 0
        and completMetric >= optim_params["target_complete"]
    ):
        completMetric = optim_params["target_complete"] + leakFactor * (
            completMetric - optim_params["target_complete"]
        )

    if optim_params["target_crowd"] > 0 and crowdMetric >= optim_params["target_crowd"]:
        crowdMetric = optim_params["target_crowd"] + leakFactor * (
            crowdMetric - optim_params["target_crowd"]
        )

    penalty = -(completMetric + crowdMetric)

    return penalty


def plot_flux_metric_diagnose(psf_model, idx=0, ax=None, optimal_percentile=None):
    """
    Function to evaluate the flux metrics for a single source as a function of
    the parameter that controls the aperture size.
    The flux metrics are computed by taking into account the PSF models of
    neighbor sources.

    This function is meant to be used only to generate diagnostic figures.

    Parameters
    ----------
    psf_model : scipy.sparce.csr_matrix
        Sparse matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    idx : int
        Index of the source for which the metrcs will be computed. Has to be a
        number between 0 and psf_models.shape[0].
    ax : matplotlib.axes
        Axis to be used to plot the figure

    Returns
    -------
    ax : matplotlib.axes
        Figure axes
    """
    compl, crowd, cut = [], [], []
    for p in range(0, 101, 1):
        cut.append(p)
        mask = (psf_model[idx] >= np.nanpercentile(psf_model[idx].data, p)).toarray()[0]
        crowd.append(compute_CROWDSAP(psf_model, mask, idx))
        compl.append(compute_FLFRCSAP(psf_model[idx], mask))

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.plot(cut, compl, label=r"FLFRCSAP", c="tab:blue")
    ax.plot(cut, crowd, label=r"CROWDSAP", c="tab:green")
    if optimal_percentile:
        ax.axvline(optimal_percentile, c="tab:red", label="optimal")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Metric")
    ax.legend()
    return ax


def estimate_source_centroids_aperture(aperture_mask, flux, column, row):
    """
    Computes the centroid via 2D moments methods for all sources all times. It needs
    `aperture_mask` to be computed first by runing `compute_aperture_photometry`.

    Parameters
    ----------
    aperture_mask : numpy.ndarray
        Aperture mask, shape is [n_surces, n_pixels]
    flux: numpy.ndarray
        Flux values at each pixels and times in units of electrons / sec
    column : numpy.ndarray
        Data array containing the "columns" of the detector that each pixel is on.
    row : numpy.ndarray
        Data array containing the "rows" of the detector that each pixel is on.

    Returns
    -------
    centroid_col : numpy.ndarray
        Column pixel number of the moments centroid, shape is [nsources, ntimes].
    centroid_row : numpy.ndarray
        Row pixel number of the moments centroid, shape is [nsources, ntimes].
    """
    centroid_col, centroid_row = [], []
    for idx in range(aperture_mask.shape[0]):
        total_flux = np.nansum(flux[:, aperture_mask[idx]], axis=1)
        centroid_col.append(
            np.nansum(
                np.tile(column[aperture_mask[idx]], (flux.shape[0], 1))
                * flux[:, aperture_mask[idx]],
                axis=1,
            )
            / total_flux
        )
        centroid_row.append(
            np.nansum(
                np.tile(row[aperture_mask[idx]], (flux.shape[0], 1))
                * flux[:, aperture_mask[idx]],
                axis=1,
            )
            / total_flux
        )
    return np.array(centroid_col), np.array(centroid_row)


def compute_FLFRCSAP(psf_models, aperture_mask):
    """
    Compute fraction of target flux enclosed in the optimal aperture to total flux
    for a given source (flux completeness).
    Follows definition by Kinemuchi et al. 2012.

    Parameters
    ----------
    psf_models : scipy.sparse.csr_matrix or numpy.ndarray
        Matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    aperture_mask: numpy.ndarray
        Array of boolean indicating the aperture for the target source. It has shape of
        [n_sources, n_pixels] or [n_pixels].

    Returns
    -------
    FLFRCSAP: numpy.ndarray
        Completeness metric
    """
    if sparse.issparse(psf_models):
        num = np.asarray(psf_models.multiply(aperture_mask.astype(float)).sum(axis=1)).ravel()
        denom = np.asarray(psf_models.sum(axis=1)).ravel()
    else:
        num = np.asarray(np.multiply(psf_models, aperture_mask.astype(float)).sum(axis=1)).ravel()
        denom = np.asarray(psf_models.sum(axis=1)).ravel()
    out = np.zeros_like(num, dtype=float)
    return np.divide(num, denom, out=out, where=denom > 0)


def compute_CROWDSAP(psf_models, aperture_mask, idx=None, flux_weights=None):
    """
    Compute the ratio of target flux relative to flux from all sources within
    the photometric aperture (i.e. 1 - Crowdedness).
    Follows definition by Kinemuchi et al. 2012:
        CROWDSAP = sum_{p in aperture} F_target(p) / sum_{p in aperture} F_total(p)

    Parameters
    ----------
    psf_models : scipy.sparse.csr_matrix or numpy.ndarray
        Matrix with the PSF models for all targets in the scene. It has shape
        [n_sources, n_pixels].
    aperture_mask : numpy.ndarray
        Array of boolean indicating the aperture for the target source. It has shape of
        [n_sources, n_pixels] or [n_pixels].
    idx : int, optional
        Source index for which the metric is computed. Value has to be between 0 and
        psf_models first dimension size.
        If None, it returns the metric for all sources.
    flux_weights : numpy.ndarray, optional
        Array of flux estimates for each source with shape [n_sources]. If provided,
        psf_models will be weighted by these fluxes before computing metrics.

    Returns
    -------
    CROWDSAP : numpy.ndarray or float
        Crowdedness metric
    """
    if flux_weights is not None:
        weights = np.maximum(np.asarray(flux_weights, dtype=float), 0.0)
        if sparse.issparse(psf_models):
            psf_models = sparse.diags(weights).dot(psf_models)
        else:
            psf_models = np.multiply(psf_models, weights[:, None])

    if sparse.issparse(psf_models):
        tot_flux = np.asarray(psf_models.sum(axis=0)).ravel()
    else:
        tot_flux = np.asarray(np.sum(psf_models, axis=0)).ravel()

    if idx is not None:
        idx = int(idx)
        ap_mask = np.asarray(aperture_mask, dtype=bool)
        if ap_mask.ndim > 1:
            if ap_mask.shape[0] == psf_models.shape[0]:
                mask = ap_mask[idx].ravel()
            else:
                mask = ap_mask.ravel()
        else:
            mask = ap_mask.ravel()

        if not np.any(mask):
            return 1.0

        if sparse.issparse(psf_models):
            target_flux = float(psf_models[idx].multiply(mask.astype(float)).sum())
        else:
            target_flux = float(np.sum(psf_models[idx, mask]))

        tot_in_mask = float(np.sum(tot_flux[mask]))
        if tot_in_mask <= 0.0:
            return 1.0
        return float(np.clip(target_flux / tot_in_mask, 0.0, 1.0))

    else:
        ap_mask = np.asarray(aperture_mask, dtype=bool)
        if ap_mask.ndim == 1:
            ap_mask = np.broadcast_to(ap_mask, (psf_models.shape[0], len(ap_mask)))

        if sparse.issparse(psf_models):
            num = np.asarray(psf_models.multiply(ap_mask.astype(float)).sum(axis=1)).ravel()
        else:
            num = np.asarray(np.sum(psf_models * ap_mask.astype(float), axis=1)).ravel()

        denom = np.asarray(ap_mask.astype(float).dot(tot_flux)).ravel()
        out = np.ones_like(num, dtype=float)
        np.divide(num, denom, out=out, where=denom > 0)
        return np.clip(out, 0.0, 1.0)


def make_snr_optimal_aperture(
    signal: np.ndarray,
    error: np.ndarray,
    sat_flux: float = 6e4,
    plot: bool = False,
) -> np.ndarray:
    """
    Generate an optimal photometric aperture mask by maximizing the cumulative Signal-to-Noise Ratio (SNR).

    The optimal aperture is determined by finding the pixel count that maximizes the total 
    SNR or where the SNR improvement flattens out, ensuring that heavily saturated pixels 
    are always enclosed before the cutoff is applied.

    Parameters
    ----------
    signal : numpy.ndarray
        Array representing the estimated target flux or PRF model per pixel in electrons/sec.
    error : numpy.ndarray
        Array of the corresponding flux errors (e.g., background and Poisson noise) per pixel.
    sat_flux : float, optional
        Flux threshold above which pixels are considered saturated. Defaults to 6e4.
    plot : bool, optional
        If True, generates a diagnostic plot showing the cumulative SNR curve and the
        selected optimal pixel count cutoff. Defaults to False.

    Returns
    -------
    optimal_mask : numpy.ndarray
        A boolean mask of the same shape as the input `signal`, where True indicates
        that a pixel is included in the optimal aperture.
    """
    # compute the variance per pixel
    variance = error**2

    # compute the SNR as signal / std = signal / sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.divide(
            signal,
            np.sqrt(variance),
            out=np.zeros_like(signal, dtype=float),
            where=variance > 0,
        )
    # replace nan with zeros as pixels with no contribution
    snr[~np.isfinite(snr)] = 0.0

    # sort indices decreasing
    flat_signal = signal.ravel()
    flat_var = variance.ravel()
    flat_snr = snr.ravel()
    sort_indices = np.argsort(flat_snr)[::-1]

    # compute cumulative sum starting from the pixels with highest SNR
    # do this for the signal and for the variance to get the cumulative noise
    cumsum_signal = np.nancumsum(flat_signal[sort_indices])
    cumsum_noise = np.sqrt(np.nancumsum(flat_var[sort_indices]))

    # find saturated pixels in order
    sorted_signal = flat_signal[sort_indices]
    indices_satpix = np.where(sorted_signal >= sat_flux)[0]

    # compute cumulative SNR
    with np.errstate(divide="ignore", invalid="ignore"):
        cumsum_snr = np.divide(
            cumsum_signal,
            cumsum_noise,
            out=np.zeros_like(cumsum_signal, dtype=float),
            where=cumsum_noise > 0,
        )
    cumsum_snr[~np.isfinite(cumsum_snr)] = 0.0

    # For saturated targets, it can happen that the first two pixels have too much
    # signal and the cumulative SNR curve has a isolated maximum at the beginning
    # of the curve. 
    # First find local maximum while avoiding edge effects
    indices = np.where(
        (cumsum_snr[1:-1] > cumsum_snr[:-2])
        & (cumsum_snr[1:-1] > cumsum_snr[2:])
    )[0] + 1
    # make sure sat pixels are included
    if len(indices_satpix) > 0:
        indices = indices[indices >= indices_satpix.max()]

    # then find the location of global maximum.
    if len(indices) > 0:
        optimal_pixel_max = indices[np.argmax(cumsum_snr[indices])] + 1
    else:
        if len(indices_satpix) > 0:
            sat_max = indices_satpix.max()
            optimal_pixel_max = sat_max + np.argmax(cumsum_snr[sat_max:]) + 1
        else:
            optimal_pixel_max = np.argmax(cumsum_snr) + 1

    # use gradients to find the point where adding more pixels does not improve SNR significantly
    grads = np.gradient(cumsum_snr)
    # percentile a gradient cut-off to find the optimal pixel count
    grad_cut = np.percentile(np.abs(grads), 5)
    wh = np.where((grads <= grad_cut) & (grads >= -grad_cut))[0]
    if len(wh) > 0:
        optimal_pixel_grad = wh[0] + 1
        optimal_pixel_count = np.minimum(optimal_pixel_max, optimal_pixel_grad)
        if len(indices_satpix) > 0:
            optimal_pixel_count = max(optimal_pixel_count, indices_satpix.max() + 1)
    else:
        optimal_pixel_count = optimal_pixel_max

    optimal_pixel_count = max(1, min(len(flat_signal), int(optimal_pixel_count)))

    # Create the initial unconstrained optimal mask
    optimal_indices = sort_indices[:optimal_pixel_count]
    optimal_mask = np.zeros_like(flat_snr, dtype=bool)
    optimal_mask[optimal_indices] = True
    optimal_mask = optimal_mask.reshape(signal.shape)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.set_title("Cumulative SNR")
        ax.plot(np.arange(len(flat_signal)) + 1, cumsum_snr, ".-", label="SNR")
        ax.axvline(
            optimal_pixel_count,
            color="tab:red",
            label=f"Optimal Pixel Count = {optimal_pixel_count}",
            zorder=10,
        )
        ax.legend()
        ax.set_ylabel("Aperture SNR")
        ax.set_xlabel("Pixel Number")
        plt.show()

    return optimal_mask


def correct_aperture_flux(
    flux: np.ndarray,
    flux_err: np.ndarray,
    flfrcsap: np.ndarray,
    crowdsap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct aperture flux and flux error for contamination (crowding) and enclosed flux fraction.

    Formulas:
        flux_removed = flux - (1 - crowdsap) * np.nanmedian(flux)
        flux_corr = flux_removed / flfrcsap
        flux_err_corr = flux_err / flfrcsap

    Parameters
    ----------
    flux : np.ndarray
        Raw aperture flux array.
    flux_err : np.ndarray
        Raw aperture flux uncertainty array.
    flfrcsap : np.ndarray
        Fraction of target flux enclosed in the aperture (completeness).
    crowdsap : np.ndarray
        Ratio of target flux to total flux in the aperture (crowding metric).

    Returns
    -------
    flux_corr : np.ndarray
        Aperture flux corrected for crowding and flux completeness.
    flux_err_corr : np.ndarray
        Corresponding corrected flux uncertainties.
    """
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)
    flfrcsap = np.asarray(flfrcsap)
    crowdsap = np.asarray(crowdsap)

    med_flux = np.nanmedian(flux)
    flux_removed = flux - (1.0 - crowdsap) * med_flux
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_corr = np.where(flfrcsap > 0, flux_removed / flfrcsap, np.nan)
        flux_err_corr = np.where(flfrcsap > 0, flux_err / flfrcsap, np.nan)

    return flux_corr, flux_err_corr


