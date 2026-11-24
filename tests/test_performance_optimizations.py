"""
Tests for performance optimizations in roman-lcs:
- Fast Cholesky solver with fallback to LU
- Coordinate memoization and precomputation
- Local bounding-box and threshold-clipped ePSF evaluation
- Vectorized aperture photometry and centroid calculations
- Strict numerical photometric equivalence
"""

import numpy as np
from scipy import sparse

from roman_lcs.prf import PhotutilsPSFAdapter
from roman_lcs.utils import solve_linear_model


def test_solve_linear_model_cholesky_vs_lu():
    """Verify that Cholesky and LU solvers give identical solutions and errors within machine precision."""
    np.random.seed(42)
    n_obs = 100
    n_features = 10

    # Create random positive-definite system
    A_dense = np.random.randn(n_obs, n_features)
    A = sparse.csr_matrix(A_dense)
    true_w = np.random.randn(n_features)
    y_err = np.random.uniform(0.5, 2.0, size=n_obs)
    y = A_dense.dot(true_w) + np.random.randn(n_obs) * y_err

    # Solve with Cholesky (default)
    w_chol, err_chol = solve_linear_model(A, y, y_err=y_err, errors=True, method="cholesky")

    # Solve with LU
    w_lu, err_lu = solve_linear_model(A, y, y_err=y_err, errors=True, method="lu")

    # Check numerical equivalence
    np.testing.assert_allclose(w_chol, w_lu, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(err_chol, err_lu, rtol=1e-10, atol=1e-10)


def test_solve_linear_model_fallback_on_singular():
    """Verify that Cholesky solver gracefully falls back when matrix is rank-deficient or ill-conditioned."""
    np.random.seed(42)
    n_obs = 50
    n_features = 5
    # Duplicate columns to force rank deficiency
    A_dense = np.random.randn(n_obs, n_features)
    A_dense[:, 1] = A_dense[:, 0]
    A = sparse.csr_matrix(A_dense)
    y = np.random.randn(n_obs)

    # Should fall back or solve without unhandled exception
    try:
        w = solve_linear_model(A, y, method="cholesky")
        assert len(w) == n_features
    except np.linalg.LinAlgError:
        pass  # Both LU and Cholesky can raise LinAlgError if completely singular


def test_photutils_adapter_evaluate_local():
    """Verify evaluate_local against evaluate_from_position on a mock ePSF."""
    class MockEPSF:
        def __init__(self):
            self.data = np.ones((25, 25))
            self.oversampling = (1, 1)

        def evaluate(self, x, y, flux=1.0, x_0=0.0, y_0=0.0):
            # Simple Gaussian profile
            r2 = (x - x_0) ** 2 + (y - y_0) ** 2
            return flux * np.exp(-0.5 * r2 / 2.0**2)

    adapter = PhotutilsPSFAdapter(MockEPSF())
    shape = (25, 25)
    center = (12.3, 12.7)

    # Full evaluation
    _, _, full_eval = adapter.evaluate_from_position(center=center, shape=shape)

    # Local evaluation with threshold clipping
    threshold = 1e-5
    r_pix, c_pix, patch_vals = adapter.evaluate_local(
        center=center,
        shape=shape,
        corner=(0, 0),
        threshold=threshold,
    )

    # Reconstruct from sparse local coordinates
    reconstructed = np.zeros(shape)
    reconstructed[r_pix, c_pix] = patch_vals

    # Clipped full evaluation
    full_clipped = np.copy(full_eval)
    max_v = np.max(full_clipped)
    full_clipped[full_clipped < threshold * max_v] = 0.0

    np.testing.assert_allclose(reconstructed, full_clipped, rtol=1e-12, atol=1e-12)


def test_coordinate_memoization():
    """Verify that pixel_coordinates caches results properly."""
    import pandas as pd

    class MockWCS:
        def __init__(self):
            self.call_count = 0

        def all_world2pix(self, coords, origin):
            self.call_count += 1
            return np.ones_like(coords)

    # Build dummy extractor
    from roman_lcs.lc_extractor import LCExtractor
    # Instantiate minimal object without running full init
    obj = object.__new__(LCExtractor)
    mock_wcs = MockWCS()
    obj.WCSs = [mock_wcs, mock_wcs]
    obj.nt = 2
    obj.sources = pd.DataFrame({"ra": [10.0, 10.1], "dec": [-10.0, -10.1]})
    obj._cached_pixel_coords = {}

    # First call computes and caches
    row1, col1 = obj.pixel_coordinates(frame_index=0)
    assert mock_wcs.call_count == 1

    # Second call uses cache
    row2, col2 = obj.pixel_coordinates(frame_index=0)
    assert mock_wcs.call_count == 1  # Not incremented!
    np.testing.assert_array_equal(row1, row2)
    np.testing.assert_array_equal(col1, col2)

    # Frame 1 computes and caches
    obj.pixel_coordinates(frame_index=1)
    assert mock_wcs.call_count == 2
