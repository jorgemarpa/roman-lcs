import warnings

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from scipy import sparse

from roman_lcs import LCExtractor, PRFFitFlag
from roman_lcs.aperture import (
    compute_CROWDSAP,
    compute_FLFRCSAP,
    correct_aperture_flux,
    make_snr_optimal_aperture,
)


class DummyPRF:
    prf_sum = 1.0

    def evaluate_from_position(self, center, shape, corner=(0, 0)):
        ny, nx = shape
        row_c, col_c = center
        y, x = np.mgrid[0:ny, 0:nx]
        stamp = np.exp(-0.5 * ((x - col_c) ** 2 + (y - row_c) ** 2) / 2.0)
        return x, y, stamp


@pytest.fixture
def sample_extractor():
    w = WCS(naxis=2)
    w.wcs.crpix = [62.5, 62.5]
    w.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    w.wcs.crval = [270.0, -30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    sliced_low = SlicedLowLevelWCS(w, (slice(50, 75), slice(50, 75)))
    sliced_wcs = HighLevelWCSWrapper(sliced_low)

    nt, ny, nx = 3, 25, 25
    # Positive flux with a peak at center
    flux = np.random.RandomState(42).normal(10.0, 1.0, (nt, ny, nx))
    flux[:, 12, 12] += 500.0  # Center star peak
    flux_err = np.ones((nt, ny, nx)) * 0.1
    time = np.array([2460000.0, 2460001.0, 2460002.0])
    cadenceno = np.array([1, 2, 3])

    col_3d, row_3d = np.meshgrid(np.arange(50, 75), np.arange(50, 75), indexing="xy")
    col_3d = np.broadcast_to(col_3d, (nt, ny, nx))
    row_3d = np.broadcast_to(row_3d, (nt, ny, nx))
    ra_3d = np.full((nt, ny, nx), 270.0)
    dec_3d = np.full((nt, ny, nx), -30.0)

    # 2 sources: target at center (ra=270, dec=-30 corresponds to col=12, row=12), and background source
    sources = pd.DataFrame({
        "ra": [270.0, 270.0001],
        "dec": [-30.0, -30.0001],
        "flux": [1000.0, 200.0],
        "F146": [18.0, 20.0],
    })

    ext = LCExtractor(
        time=time,
        flux=flux,
        flux_err=flux_err,
        ra=ra_3d,
        dec=dec_3d,
        sources=sources,
        column=col_3d,
        row=row_3d,
        cadenceno=cadenceno,
        wcs=[sliced_wcs, sliced_wcs, sliced_wcs],
        meta={"SCA": 1, "FILTER": "F146"},
        sources_flux_column="flux",
        sources_mag_column="F146",
    )
    ext.prf_model = DummyPRF()
    return ext


def test_make_snr_optimal_aperture():
    """Test make_snr_optimal_aperture algorithm including saturation constraints and dimensions."""
    # 1. 2D Gaussian test case
    ny, nx = 25, 25
    y, x = np.mgrid[0:ny, 0:nx]
    signal_2d = 500.0 * np.exp(-0.5 * ((x - 12) ** 2 + (y - 12) ** 2) / 2.0)
    error_2d = np.ones((ny, nx)) * 2.0

    mask_2d = make_snr_optimal_aperture(signal_2d, error_2d)
    assert mask_2d.shape == (ny, nx)
    assert mask_2d.dtype == bool
    # Center pixel must be included
    assert mask_2d[12, 12]
    # Aperture size should be reasonable (e.g. 5 to 50 pixels)
    assert 5 <= mask_2d.sum() <= 50

    # 2. 1D array support
    signal_1d = signal_2d.ravel()
    error_1d = error_2d.ravel()
    mask_1d = make_snr_optimal_aperture(signal_1d, error_1d)
    assert mask_1d.shape == (ny * nx,)
    assert mask_1d.sum() == mask_2d.sum()

    # 3. Saturation handling: saturated pixels must always be enclosed
    sat_signal = np.copy(signal_2d)
    sat_signal[12, 12] = 100000.0  # Above 6e4
    sat_signal[12, 13] = 80000.0   # Above 6e4
    sat_mask = make_snr_optimal_aperture(sat_signal, error_2d, sat_flux=6e4)
    assert sat_mask[12, 12]
    assert sat_mask[12, 13]


def test_default_circular_aperture(sample_extractor):
    """Test default aperture_radius=3 in fit_prf_photometry."""
    sample_extractor.fit_prf_photometry(targets=[0], model_bkg=False, do_aperture=True)

    lc_dict = sample_extractor.lcs[0]
    # Check top-level keys
    assert "psf" in lc_dict
    assert "aperture_snr" in lc_dict
    assert "aperture_iso" in lc_dict
    assert "aperture_opt" not in lc_dict
    assert "aperture_3pix" in lc_dict
    assert "aperture" not in lc_dict

    # Verify aperture_iso is distinct from aperture_snr and uses prf_cut
    assert lc_dict["aperture_iso"] is not lc_dict["aperture_snr"]
    assert hasattr(sample_extractor, "prf_cut")
    assert len(sample_extractor.prf_cut) == sample_extractor.nsources
    # Check that aperture_iso mask encloses >= 90% of PRF flux
    assert np.all(lc_dict["aperture_iso"]["flfrcsap"] >= 0.90)

    # Check fields in aperture_3pix
    circ_lc = lc_dict["aperture_3pix"]
    expected_fields = [
        "time",
        "flux",
        "flux_err",
        "flux_corr",
        "flux_err_corr",
        "flfrcsap",
        "crowdsap",
        "centroid_x",
        "centroid_y",
        "aperture_mask",
        "quality",
    ]
    for field in expected_fields:
        assert field in circ_lc
        assert len(circ_lc[field]) == sample_extractor.nt

    # Check aperture mask dimensions and geometric properties
    mask = circ_lc["aperture_mask"][0]
    assert mask.shape == (25 * 25,)
    # Circular mask radius 3 should contain pixels
    n_pix = mask.sum()
    assert 20 <= n_pix <= 40  # pi * 3^2 ≈ 28.3

    # Check lightkurve export
    lk_dict = sample_extractor.get_lightcurves(mode="lk")
    assert 0 in lk_dict
    lc = lk_dict[0]
    assert "psf_flux" in lc.colnames
    assert "aperture_3pix_flux" in lc.colnames
    assert "aperture_3pix_flux_err" in lc.colnames
    assert "ap_bkg_flux" in lc.colnames
    assert "aperture_3pix_bkg_flux" in lc.colnames


def test_custom_list_aperture_radii(sample_extractor):
    """Test passing a list of aperture radii [2, 3, 5] and float radius 2.5."""
    sample_extractor.fit_prf_photometry(
        targets=[0],
        model_bkg=False,
        do_aperture=True,
        aperture_radius=[2, 3, 5, 2.5],
    )

    lc_dict = sample_extractor.lcs[0]
    assert "psf" in lc_dict
    assert "aperture_snr" in lc_dict
    assert "aperture_iso" in lc_dict
    assert "aperture_opt" not in lc_dict
    assert "aperture_2pix" in lc_dict
    assert "aperture_3pix" in lc_dict
    assert "aperture_5pix" in lc_dict
    assert "aperture_2.5pix" in lc_dict

    # Flux should increase monotonically with radius since all flux is positive
    f2 = lc_dict["aperture_2pix"]["flux"][0]
    f25 = lc_dict["aperture_2.5pix"]["flux"][0]
    f3 = lc_dict["aperture_3pix"]["flux"][0]
    f5 = lc_dict["aperture_5pix"]["flux"][0]
    assert f2 < f25 < f3 < f5


def test_do_aperture_photometry_standalone(sample_extractor):
    """Test do_aperture_photometry standalone method with custom radius."""
    sample_extractor.draw_scene(frame_index=0)
    sample_extractor.do_aperture_photometry(targets=[0], aperture_radius=4)

    lc_dict = sample_extractor.lcs[0]
    assert "aperture_snr" in lc_dict
    assert "aperture_iso" in lc_dict
    assert "aperture_opt" not in lc_dict
    assert "aperture_4pix" in lc_dict
    assert "aperture" not in lc_dict


def test_background_model_and_subtraction(sample_extractor):
    """Test background model fitting, attributes, error propagation, and aperture subtraction."""
    # Add a flat background of 50.0 e-/s
    pristine_flux = sample_extractor.flux.copy()
    sample_extractor.flux += 50.0

    sample_extractor.fit_prf_photometry(
        targets=[0],
        model_bkg=True,
        bkg_nknots=3,
        do_aperture=True,
        aperture_radius=[2, 3, 5],
    )

    # 1. Verify class attributes and shapes
    assert hasattr(sample_extractor, "bkg_flux")
    assert hasattr(sample_extractor, "bkg_flux_err")
    assert sample_extractor.bkg_flux.shape == sample_extractor.flux.shape
    assert sample_extractor.bkg_flux_err.shape == sample_extractor.flux.shape

    # 3D properties
    assert sample_extractor.bkg_flux_3d.shape == (sample_extractor.nt, *sample_extractor.image_shape)
    assert sample_extractor.bkg_flux_err_3d.shape == (sample_extractor.nt, *sample_extractor.image_shape)

    # Verify self.flux was NOT modified in place
    expected_flux = pristine_flux + 50.0
    assert np.allclose(sample_extractor.flux, expected_flux)

    # Background flux should be approximately 50.0 everywhere
    assert np.all(sample_extractor.bkg_flux > 30.0)
    assert np.all(sample_extractor.bkg_flux_err >= 0.0)
    assert np.all(np.isfinite(sample_extractor.bkg_flux_err))

    # 2. Verify aperture outputs in self.lcs
    lc_0 = sample_extractor.lcs[0]
    assert "aperture_opt" not in lc_0
    for key in ["aperture_snr", "aperture_iso", "aperture_2pix", "aperture_3pix", "aperture_5pix"]:
        assert "bkg_flux" in lc_0[key]
        assert "bkg_flux_err" in lc_0[key]
        assert len(lc_0[key]["bkg_flux"]) == sample_extractor.nt
        assert len(lc_0[key]["bkg_flux_err"]) == sample_extractor.nt
        assert np.all(lc_0[key]["bkg_flux"] > 0)
        assert np.all(lc_0[key]["bkg_flux_err"] > 0)

    # Background flux in circular apertures must scale monotonically with aperture area
    bkg_2 = lc_0["aperture_2pix"]["bkg_flux"][0]
    bkg_3 = lc_0["aperture_3pix"]["bkg_flux"][0]
    bkg_5 = lc_0["aperture_5pix"]["bkg_flux"][0]
    assert bkg_2 < bkg_3 < bkg_5

    # Check that raw flux = subtracted_flux + bkg_flux
    mask_3 = lc_0["aperture_3pix"]["aperture_mask"][0]
    raw_flux_in_ap = sample_extractor.flux[0, mask_3].sum()
    measured_sub_flux = lc_0["aperture_3pix"]["flux"][0]
    measured_bkg_flux = lc_0["aperture_3pix"]["bkg_flux"][0]
    assert np.isclose(raw_flux_in_ap, measured_sub_flux + measured_bkg_flux, rtol=1e-5)


def test_bkg_flux_excludes_background_stars(sample_extractor):
    """Verify that bkg_flux strictly models diffuse background, while bkg_model includes background stars."""
    # sample_extractor has 2 sources: target [0] at center and background source [1]
    # Place a bright background star at source [1]
    sample_extractor.source_flux_estimates[1] = 5000.0

    sample_extractor.fit_prf_photometry(
        targets=[0],
        model_bkg=True,
        bkg_nknots=3,
        do_aperture=False,
    )

    # 1. Attributes and properties exist
    assert hasattr(sample_extractor, "bkg_flux")
    assert hasattr(sample_extractor, "bkg_flux_err")
    assert hasattr(sample_extractor, "bkg_model")
    assert hasattr(sample_extractor, "bkg_model_err")
    assert sample_extractor.bkg_model.shape == sample_extractor.flux.shape
    assert sample_extractor.bkg_model_err.shape == sample_extractor.flux.shape
    assert sample_extractor.bkg_model_3d.shape == (sample_extractor.nt, *sample_extractor.image_shape)
    assert sample_extractor.bkg_model_err_3d.shape == (sample_extractor.nt, *sample_extractor.image_shape)

    # 2. Ensure self.bkg_flux does not include the background star, but bkg_model does
    srow, scol = sample_extractor.pixel_coordinates(frame_index=0)
    col_min = sample_extractor.column[0].min()
    row_min = sample_extractor.row[0].min()
    bkg_star_col = int(round(scol[1] - col_min if scol[1] >= col_min else scol[1]))
    bkg_star_row = int(round(srow[1] - row_min if srow[1] >= row_min else srow[1]))

    bkg_flux_at_star = sample_extractor.bkg_flux_3d[0, bkg_star_row, bkg_star_col]
    bkg_model_at_star = sample_extractor.bkg_model_3d[0, bkg_star_row, bkg_star_col]
    mean_bkg_flux = np.mean(sample_extractor.bkg_flux_3d[0])

    # The smooth spline background at the star position should be comparable to the mean background
    assert abs(bkg_flux_at_star - mean_bkg_flux) < 100.0

    # bkg_model includes the background star, so bkg_model at the star position should be >= bkg_flux
    assert bkg_model_at_star >= bkg_flux_at_star


def test_aperture_metrics_zero_flux_pixels():
    """Verify compute_CROWDSAP and compute_FLFRCSAP safely handle zero-flux pixels without RuntimeWarnings."""
    # Create scene with 2 sources across 4 pixels, where column 0 and column 3 have 0 total flux
    psf_dense = np.array([
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    mask_dense = np.array([
        [False, True, True, False],
        [False, False, True, False],
    ])
    psf_sparse = sparse.csr_matrix(psf_dense)

    # 1. Ensure zero RuntimeWarnings are raised (divide by zero or invalid multiply)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        # Dense
        crowd_dense = compute_CROWDSAP(psf_dense, mask_dense, idx=None)
        compl_dense = compute_FLFRCSAP(psf_dense, mask_dense)
        crowd_dense_0 = compute_CROWDSAP(psf_dense, mask_dense[0], idx=0)

        # Sparse
        crowd_sparse = compute_CROWDSAP(psf_sparse, mask_dense, idx=None)
        compl_sparse = compute_FLFRCSAP(psf_sparse, mask_dense)
        crowd_sparse_0 = compute_CROWDSAP(psf_sparse, mask_dense[0], idx=0)

    # 2. Check values and dense/sparse equivalence following Kinemuchi et al. 2012 integrated flux ratio
    # Target 0: target flux in mask = 1.0 + 2.0 = 3.0; total flux in mask = 1.0 + 3.0 = 4.0 -> 3/4
    # Target 1: target flux in mask = 1.0; total flux in mask = 3.0 -> 1/3
    expected_crowd = np.array([3.0 / 4.0, 1.0 / 3.0])
    assert np.allclose(crowd_dense, expected_crowd)
    assert np.allclose(crowd_sparse, expected_crowd)
    assert np.isclose(crowd_dense_0, 3.0 / 4.0)
    assert np.isclose(crowd_sparse_0, 3.0 / 4.0)

    # 3. Test flux_weights support
    weights = np.array([100.0, 10.0])
    # Target 0: 300 / (300 + 10) = 300 / 310
    # Target 1: 10 / (200 + 10) = 10 / 210
    crowd_weighted = compute_CROWDSAP(psf_dense, mask_dense, flux_weights=weights)
    assert np.allclose(crowd_weighted, [300.0 / 310.0, 10.0 / 210.0])
    crowd_weighted_sparse = compute_CROWDSAP(psf_sparse, mask_dense, flux_weights=weights)
    assert np.allclose(crowd_weighted_sparse, [300.0 / 310.0, 10.0 / 210.0])

    expected_compl = np.array([3.0 / 3.0, 1.0 / 1.0])
    assert np.allclose(compl_dense, expected_compl)
    assert np.allclose(compl_sparse, expected_compl)

    # 4. Test empty aperture and zero-flux source handling
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        empty_mask = np.array([[False, False, False, False]])
        assert compute_CROWDSAP(psf_dense[:1], empty_mask, idx=None)[0] == 1.0
        assert compute_CROWDSAP(psf_dense[:1], empty_mask[0], idx=0) == 1.0

        zero_star = np.array([[0.0, 0.0, 0.0, 0.0]])
        assert compute_FLFRCSAP(zero_star, mask_dense[:1])[0] == 0.0


def test_corrected_aperture_flux(sample_extractor):
    """Test flux_corr and flux_err_corr keys and formula correctness."""
    sample_extractor.fit_prf_photometry(targets=[0], model_bkg=False, do_aperture=True)

    for ap_key in ["aperture_snr", "aperture_iso", "aperture_3pix"]:
        ap_dict = sample_extractor.lcs[0][ap_key]
        assert "flux_corr" in ap_dict
        assert "flux_err_corr" in ap_dict

        flux = ap_dict["flux"]
        flux_err = ap_dict["flux_err"]
        flfrcsap = ap_dict["flfrcsap"]
        crowdsap = ap_dict["crowdsap"]

        expected_flux_removed = flux - (1.0 - crowdsap) * np.nanmedian(flux)
        expected_flux_corr = expected_flux_removed / flfrcsap
        expected_flux_err_corr = flux_err / flfrcsap

        assert np.allclose(ap_dict["flux_corr"], expected_flux_corr, equal_nan=True)
        assert np.allclose(ap_dict["flux_err_corr"], expected_flux_err_corr, equal_nan=True)

    # Direct function test with edge cases
    f = np.array([100.0, 110.0, 90.0])
    ferr = np.array([2.0, 2.2, 1.8])
    flfrc = np.array([0.9, 0.9, 0.9])
    crowd = np.array([0.95, 0.95, 0.95])

    f_corr, ferr_corr = correct_aperture_flux(f, ferr, flfrc, crowd)
    med = 100.0
    f_rem = f - (1.0 - 0.95) * med
    assert np.allclose(f_corr, f_rem / 0.9)
    assert np.allclose(ferr_corr, ferr / 0.9)


def test_gradient_offset_recovery(sample_extractor):
    """Test iterative gradient-based sub-pixel centroid recovery and coordinate storage."""
    ext = sample_extractor
    # Inject known sub-pixel shift at cadence 1
    inj_drow = 0.06
    inj_dcol = -0.05
    srow_nom, scol_nom = ext.pixel_coordinates(frame_index=1)
    if np.nanmean(srow_nom) >= ext.row[1].min():
        srow_c = srow_nom[0] - ext.row[1].min()
        scol_c = scol_nom[0] - ext.column[1].min()
    else:
        srow_c = srow_nom[0]
        scol_c = scol_nom[0]

    # Generate synthetic image at shifted position
    _, _, shifted_stamp = ext.prf_model.evaluate_from_position(
        center=(srow_c + inj_drow, scol_c + inj_dcol),
        shape=ext.image_shape,
    )
    # Target flux is 1000.0
    ext.flux_3d[1] = shifted_stamp * 1000.0
    ext.flux[1] = ext.flux_3d[1].ravel()

    # Fit with fit_offsets=True
    ext.fit_prf_photometry(
        targets=[0],
        model_bkg=False,
        do_aperture=True,
        fit_offsets=True,
        offset_threshold=0.005,
    )

    # 1. Recovered offsets match injected values
    assert np.isclose(ext.prf_drow[1], inj_drow, atol=0.01)
    assert np.isclose(ext.prf_dcol[1], inj_dcol, atol=0.01)

    # 2. Variable attributes and PSF light curve dictionary keys
    psf_dict = ext.lcs[0]["psf"]
    assert "drow" in psf_dict
    assert "dcol" in psf_dict
    assert "row" in psf_dict
    assert "column" in psf_dict
    assert "fit_flag" in psf_dict

    assert np.allclose(psf_dict["drow"], ext.prf_drow)
    assert np.allclose(psf_dict["dcol"], ext.prf_dcol)
    expected_row = srow_nom[0] + ext.row[1].min() if np.nanmean(srow_nom) < ext.row[1].min() else srow_nom[0]
    expected_col = scol_nom[0] + ext.column[1].min() if np.nanmean(scol_nom) < ext.column[1].min() else scol_nom[0]
    assert np.isclose(psf_dict["row"][1], expected_row)
    assert np.isclose(psf_dict["column"][1], expected_col)

    # 3. Fit flag was SUCCESS for cadence 1
    assert psf_dict["fit_flag"][1] == PRFFitFlag.SUCCESS


def test_prf_fit_flags(sample_extractor):
    """Test PRFFitFlag bitmask diagnostics for non-convergence, max iterations, and masking."""
    ext = sample_extractor
    inj_drow = 0.25
    inj_dcol = 0.25
    srow_nom, scol_nom = ext.pixel_coordinates(frame_index=0)
    if np.nanmean(srow_nom) >= ext.row[0].min():
        srow_c = srow_nom[0] - ext.row[0].min()
        scol_c = scol_nom[0] - ext.column[0].min()
    else:
        srow_c = srow_nom[0]
        scol_c = scol_nom[0]

    _, _, shifted_stamp = ext.prf_model.evaluate_from_position(
        center=(srow_c + inj_drow, scol_c + inj_dcol),
        shape=ext.image_shape,
    )
    ext.flux_3d[0] = shifted_stamp * 1000.0
    ext.flux[0] = ext.flux_3d[0].ravel()

    # With offset_max_iter=1 and very small threshold, it should not converge
    ext.fit_prf_photometry(
        targets=[0],
        model_bkg=False,
        do_aperture=False,
        fit_offsets=True,
        offset_max_iter=1,
        offset_threshold=1e-6,
    )

    flag_0 = ext.prf_fit_flags[0]
    assert (flag_0 & PRFFitFlag.NOT_CONVERGED) != 0
    assert (flag_0 & PRFFitFlag.MAX_ITER_REACHED) != 0

    # Test insufficient unmasked pixels
    ext.pixel_mask[2] = False
    ext.fit_prf_photometry(
        targets=[0],
        model_bkg=False,
        do_aperture=False,
        fit_offsets=False,
    )
    flag_2 = ext.prf_fit_flags[2]
    assert (flag_2 & PRFFitFlag.INSUFFICIENT_UNMASKED_PIXELS) != 0


def test_stable_snr_optimal_aperture_pixel_count(sample_extractor):
    """Test that aperture_snr maintains a strictly constant pixel count across all cadences."""
    ext = sample_extractor
    ext.fit_prf_photometry(targets=[0], model_bkg=False, do_aperture=True)

    snr_masks = ext.lcs[0]["aperture_snr"]["aperture_mask"]
    assert len(snr_masks) == ext.nt
    pixel_counts = [mask.sum() for mask in snr_masks]
    # All cadences must have the exact same pixel count
    assert all(c == pixel_counts[0] for c in pixel_counts)
    assert hasattr(ext, "snr_optimal_pixel_count")
    assert ext.snr_optimal_pixel_count[0] == pixel_counts[0]


def test_aperture_flux_consistency_across_apertures(sample_extractor):
    """Test that corrected flux from isophote and snr-optimal apertures are consistent."""
    ext = sample_extractor
    ext.fit_prf_photometry(targets=[0], model_bkg=False, do_aperture=True)

    snr_corr = np.nanmean(ext.lcs[0]["aperture_snr"]["flux_corr"])
    iso_corr = np.nanmean(ext.lcs[0]["aperture_iso"]["flux_corr"])

    # CROWDSAP for isophote aperture should remain close to 1.0 (not depressed by faint tails)
    iso_crowdsap = np.nanmean(ext.lcs[0]["aperture_iso"]["crowdsap"])
    assert iso_crowdsap > 0.95

    # Corrected flux between isophote and snr-optimal must agree closely (within 5%)
    assert np.isclose(iso_corr, snr_corr, rtol=0.05)


def test_compute_median_background(sample_extractor):
    """Test _compute_median_background accurately isolates diffuse background using sigma clipping."""
    ext = sample_extractor
    # True background is ~10.0 (from sample_extractor normal(10.0, 1.0))
    med_bkg = ext._compute_median_background(sigma=3.0)

    # 1. Shape and attributes
    assert isinstance(med_bkg, np.ndarray)
    assert len(med_bkg) == ext.nt
    assert hasattr(ext, "median_bkg")
    assert np.allclose(ext.median_bkg, med_bkg)
    assert hasattr(ext, "bkg_pixel_mask")
    assert ext.bkg_pixel_mask.shape == ext.flux.shape
    assert ext.bkg_pixel_mask_3d.shape == (ext.nt, *ext.image_shape)

    # 2. Rejects bright star peak at (12, 12)
    # The star peak is 500.0, so (12, 12) should definitely be masked out from background
    assert not ext.bkg_pixel_mask_3d[0, 12, 12]

    # 3. Median background estimate is accurate (~10.0, not biased by 500.0 star)
    assert np.all(np.abs(med_bkg - 10.0) < 1.0)


def test_median_background_subtraction_and_propagation(sample_extractor):
    """Test that median background is subtracted during fitting and propagated to bkg_flux and bkg_model."""
    ext = sample_extractor
    # Add a time-dependent background offset: [50.0, 75.0, 100.0]
    bkg_offsets = np.array([50.0, 75.0, 100.0])
    for t in range(ext.nt):
        ext.flux[t] += bkg_offsets[t]

    ext.fit_prf_photometry(
        targets=[0],
        model_bkg=True,
        bkg_nknots=3,
        do_aperture=True,
        aperture_radius=3,
    )

    # 1. median_bkg reflects the injected offsets (+ original 10.0)
    assert hasattr(ext, "median_bkg")
    expected_med_bkg = 10.0 + bkg_offsets
    assert np.allclose(ext.median_bkg, expected_med_bkg, atol=1.5)

    # 2. bkg_flux contains total background (median + 2D terms)
    for t in range(ext.nt):
        assert np.isclose(np.mean(ext.bkg_flux[t]), expected_med_bkg[t], atol=2.0)

    # 3. bkg_model contains total background + background stars
    assert np.all(ext.bkg_model >= ext.bkg_flux - 1e-12)

    # 4. Total scene model includes median_bkg
    assert np.isclose(np.mean(ext.scene_model_cube[0]), np.mean(ext.flux[0]), atol=5.0)

    # 5. Flux in aperture photometry has background subtracted correctly
    lc = ext.lcs[0]["aperture_3pix"]
    raw_ap_flux = ext.flux[0, lc["aperture_mask"][0]].sum()
    bkg_ap_flux = lc["bkg_flux"][0]
    sub_ap_flux = lc["flux"][0]
    assert np.isclose(raw_ap_flux, sub_ap_flux + bkg_ap_flux, rtol=1e-5)


def test_aperture_centroid_pixel_coordinates(sample_extractor):
    """Test that aperture centroid_x and centroid_y match the image pixel grid and give correct offsets."""
    ext = sample_extractor
    ext.fit_prf_photometry(targets=[0], model_bkg=False, do_aperture=True)

    lc_snr = ext.lcs[0]["aperture_snr"]
    nom_col = ext.lcs[0]["psf"]["column"]
    nom_row = ext.lcs[0]["psf"]["row"]

    # In sample_extractor, flux peak is at cutout (12, 12) -> col=62, row=62 in pixel grid
    # Nominal WCS is at col=61.5, row=61.5 -> expected offset dx ~ +0.5, dy ~ +0.5
    for t in range(ext.nt):
        cx = lc_snr["centroid_x"][t]
        cy = lc_snr["centroid_y"][t]
        assert np.isclose(cx, 62.0, atol=0.5)
        assert np.isclose(cy, 62.0, atol=0.5)

        dx = cx - nom_col[t]
        dy = cy - nom_row[t]
        assert np.isclose(dx, 0.4, atol=0.2)
        assert np.isclose(dy, 0.4, atol=0.2)





