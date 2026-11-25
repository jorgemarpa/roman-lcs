import numpy as np
import pytest

from roman_lcs.utils import (
    compute_lightcurve_metrics,
    compute_p2p_scatter,
    compute_precision_metrics,
    compute_rms,
)


def test_compute_p2p_scatter():
    # 1. Exact mathematical test
    # diff = [1, 2, 3] -> med = 2 -> |diff - 2| = [1, 0, 1] -> MAD = 1.0
    # y = [0, 1, 3, 6]
    y = np.array([0.0, 1.0, 3.0, 6.0])
    expected_p2p = (1.4826 / np.sqrt(2.0)) * 1.0
    assert np.isclose(compute_p2p_scatter(y), expected_p2p)

    # 2. Gaussian noise recovery: p2p should accurately estimate sigma
    np.random.seed(42)
    true_sigma = 0.05
    flux = 1000.0 + np.random.normal(0, true_sigma, 10000)
    measured_p2p = compute_p2p_scatter(flux)
    assert np.isclose(measured_p2p, true_sigma, rtol=0.05)

    # 3. Normalization and ppm flags
    p2p_norm = compute_p2p_scatter(flux, normalize=True)
    assert np.isclose(p2p_norm, true_sigma / 1000.0, rtol=0.05)

    p2p_ppm = compute_p2p_scatter(flux, ppm=True)
    assert np.isclose(p2p_ppm, (true_sigma / 1000.0) * 1e6, rtol=0.05)


def test_compute_rms():
    # 1. Constant array has 0 RMS
    assert compute_rms(np.ones(100) * 50.0) == 0.0

    # 2. Gaussian noise
    np.random.seed(42)
    true_sigma = 2.5
    flux = 500.0 + np.random.normal(0, true_sigma, 10000)
    measured_rms = compute_rms(flux)
    assert np.isclose(measured_rms, true_sigma, rtol=0.05)

    # 3. Outlier clipping
    flux_with_outlier = flux.copy()
    flux_with_outlier[0] = 500.0 + 1000.0  # extreme outlier
    rms_clipped = compute_rms(flux_with_outlier, sigma_clip=5.0)
    assert np.isclose(rms_clipped, true_sigma, rtol=0.05)

    # 4. Normalization and ppm
    rms_ppm = compute_rms(flux, ppm=True)
    assert np.isclose(rms_ppm, (true_sigma / 500.0) * 1e6, rtol=0.05)


def test_numpy_arrays_and_edge_cases():
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    f = 100.0 + np.random.normal(0, 1.0, 100)

    # 1. Standard numpy array input
    m = compute_lightcurve_metrics(f, time=t)
    assert "cdpp" in m
    assert "rms" in m
    assert "p2p" in m
    assert "mad" in m
    assert m["cdpp"] > 0
    assert m["rms"] > 0
    assert m["p2p"] > 0
    assert m["mad"] > 0

    # 2. NaN filtering
    f_nan = f.copy()
    f_nan[10] = np.nan
    f_nan[25] = np.inf
    m_nan = compute_lightcurve_metrics(f_nan, time=t)
    assert np.isfinite(m_nan["cdpp"])
    assert np.isfinite(m_nan["rms"])
    assert np.isfinite(m_nan["p2p"])

    # 3. Alias test
    m_alias = compute_precision_metrics(f, time=t)
    assert m_alias == m

    # 4. Out-of-order time array gets sorted
    perm = np.random.permutation(len(t))
    p2p_sorted = compute_p2p_scatter(f[perm], time=t[perm])
    p2p_orig = compute_p2p_scatter(f, time=t)
    assert np.isclose(p2p_sorted, p2p_orig)

    # 5. Length mismatch raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_rms(f, time=t[:50])
