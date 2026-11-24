import numpy as np
import pandas as pd
import pytest
from astropy.wcs.wcsapi import HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roman_lcs import LCExtractor


def test_plot_image_wcs_projection():
    # Construct a simple FITS WCS and slice it
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    w.wcs.crval = [270.0, -30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Sliced WCS representing a 25x25 cutout
    sliced_low = SlicedLowLevelWCS(w, (slice(50, 75), slice(50, 75)))
    sliced_wcs = HighLevelWCSWrapper(sliced_low)

    # Fake data
    nt, ny, nx = 2, 25, 25
    flux = np.ones((nt, ny, nx))
    flux_err = np.ones((nt, ny, nx)) * 0.1
    time = np.array([2460000.0, 2460001.0])
    cadenceno = np.array([1, 2])
    col_3d, row_3d = np.meshgrid(np.arange(50, 75), np.arange(50, 75), indexing="xy")
    col_3d = np.broadcast_to(col_3d, (nt, ny, nx))
    row_3d = np.broadcast_to(row_3d, (nt, ny, nx))
    ra_3d = np.full((nt, ny, nx), 270.0)
    dec_3d = np.full((nt, ny, nx), -30.0)

    sources = pd.DataFrame({
        "ra": [270.0],
        "dec": [-30.0],
        "flux": [1000.0],
        "F146": [18.0],
    })

    extractor = LCExtractor(
        time=time,
        flux=flux,
        flux_err=flux_err,
        ra=ra_3d,
        dec=dec_3d,
        sources=sources,
        column=col_3d,
        row=row_3d,
        cadenceno=cadenceno,
        wcs=[sliced_wcs, sliced_wcs],
        meta={"SCA": 1, "FILTER": "F146"},
        sources_flux_column="flux",
        sources_mag_column="F146",
    )

    # 1. Plot with default (coords="sky", WCS projection + grid enabled)
    ax_sky = extractor.plot_image(sources=True, frame_index=0)
    assert hasattr(ax_sky, "coords")
    ax_sky.figure.canvas.draw()
    for coord in ax_sky.coords:
        assert len(coord._grid_lines) > 0
        for p in coord._grid_lines:
            assert not np.isnan(p.vertices).any()
    plt.close("all")

    # 2. Plot with coords="pixel" (no WCS projection)
    ax_pixel = extractor.plot_image(coords="pixel", sources=True, frame_index=0)
    assert not hasattr(ax_pixel, "coords")
    assert ax_pixel.get_xlabel() == "Column [pixel]"
    assert ax_pixel.get_ylabel() == "Row [pixel]"
    plt.close("all")

    # 3. Plot with coords="sky" and grid=False
    ax_sky_nogrid = extractor.plot_image(coords="sky", grid=False, frame_index=0)
    assert hasattr(ax_sky_nogrid, "coords")
    plt.close("all")

    # 4. Test draw_scene and plot_image(data="scene")
    class DummyPRF:
        prf_sum = 1.0
        def evaluate_from_position(self, center, shape, corner=(0, 0)):
            ny, nx = shape
            row_c, col_c = center
            y, x = np.mgrid[0:ny, 0:nx]
            # Simple Gaussian centered on (row_c, col_c)
            stamp = np.exp(-0.5 * ((x - col_c) ** 2 + (y - row_c) ** 2) / 2.0)
            return x, y, stamp

    extractor.prf_model = DummyPRF()
    extractor.draw_scene(frame_index=0)
    assert not np.isnan(extractor.scene_model).any()
    assert extractor.scene_model.max() > 0

    ax_scene = extractor.plot_image(data="scene", frame_index=0)
    assert hasattr(ax_scene, "coords")
    plt.close("all")
