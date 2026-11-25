import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS

from roman_lcs import LCExtractor


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
