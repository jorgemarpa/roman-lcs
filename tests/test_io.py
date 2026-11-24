import os
import tempfile
import numpy as np
import pytest
from astropy.io import fits

from roman_lcs.io import (
    format_aperture_col_prefix,
    write_lightcurve_fits,
    read_lightcurve_fits,
    _get_short_aperture_id,
)
from roman_lcs.scripts.extract_lightcurve import parse_args, normalize_detector
from tests.test_aperture_photometry import sample_extractor, DummyPRF


def test_format_aperture_helpers():
    assert format_aperture_col_prefix("aperture_snr") == "AP_SNR"
    assert format_aperture_col_prefix("aperture_iso") == "AP_ISO"
    assert format_aperture_col_prefix("aperture_3pix") == "AP_3PIX"
    assert format_aperture_col_prefix("aperture_2.5pix") == "AP_2_5PIX"

    assert _get_short_aperture_id("aperture_snr") == "SNR"
    assert _get_short_aperture_id("aperture_iso") == "ISO"
    assert _get_short_aperture_id("aperture_3pix") == "3P"
    assert _get_short_aperture_id("aperture_10pix") == "10P"


def test_write_and_read_lightcurve_fits_roundtrip():
    nt = 15
    ny, nx = 25, 25
    time = np.linspace(2460000.0, 2460010.0, nt)
    cadenceno = np.arange(100, 100 + nt)

    # Mock masks
    mask_iso = np.zeros((ny, nx), dtype=bool)
    mask_iso[10:15, 10:15] = True  # 25 pixels
    mask_3pix = np.zeros((ny, nx), dtype=bool)
    mask_3pix[11:14, 11:14] = True  # 9 pixels

    lcs = {
        "psf": {
            "time": time,
            "flux": np.linspace(500.0, 520.0, nt, dtype=np.float32),
            "flux_err": np.full(nt, 2.5, dtype=np.float32),
            "chi2": np.full(nt, 1.05, dtype=np.float32),
            "quality": np.zeros(nt, dtype=np.int32),
            "drow": np.full(nt, 0.05, dtype=np.float32),
            "dcol": np.full(nt, -0.03, dtype=np.float32),
            "row": np.full(nt, 12.5, dtype=np.float64),
            "column": np.full(nt, 12.5, dtype=np.float64),
            "fit_flag": np.zeros(nt, dtype=np.int32),
        },
        "aperture_iso": {
            "time": time,
            "flux": np.linspace(480.0, 495.0, nt, dtype=np.float32),
            "flux_err": np.full(nt, 3.0, dtype=np.float32),
            "bkg_flux": np.full(nt, 50.0, dtype=np.float32),
            "bkg_flux_err": np.full(nt, 0.8, dtype=np.float32),
            "flfrcsap": np.full(nt, 0.92, dtype=np.float32),
            "crowdsap": np.full(nt, 0.88, dtype=np.float32),
            "centroid_x": np.full(nt, 12.0, dtype=np.float32),
            "centroid_y": np.full(nt, 12.0, dtype=np.float32),
            "quality": np.zeros(nt, dtype=np.int32),
            "aperture_mask": mask_iso,
        },
        "aperture_3pix": {
            "time": time,
            "flux": np.linspace(450.0, 465.0, nt, dtype=np.float32),
            "flux_err": np.full(nt, 2.8, dtype=np.float32),
            "bkg_flux": np.full(nt, 18.0, dtype=np.float32),
            "bkg_flux_err": np.full(nt, 0.5, dtype=np.float32),
            "flfrcsap": np.full(nt, 0.85, dtype=np.float32),
            "crowdsap": np.full(nt, 0.95, dtype=np.float32),
            "centroid_x": np.full(nt, 12.0, dtype=np.float32),
            "centroid_y": np.full(nt, 12.0, dtype=np.float32),
            "quality": np.zeros(nt, dtype=np.int32),
            "aperture_mask": mask_3pix,
        },
    }

    meta = {
        "PID": 1055,
        "FILTER": "F146",
        "DETECTOR": "WFI02",
        "SCA": 2,
        "RA_OBJ": 250.370832,
        "DEC_OBJ": 36.505418,
        "TARGETID": "STAR_001",
        "CUTSIZE": 25,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_fits = os.path.join(tmpdir, "test_lc.fits")
        hdul = write_lightcurve_fits(
            lcs=lcs,
            output_path=out_fits,
            meta=meta,
            cadenceno=cadenceno,
            overwrite=True,
        )

        # 1. Inspect FITS structure directly
        assert os.path.isfile(out_fits)
        with fits.open(out_fits) as h:
            assert len(h) >= 2
            pri_hdr = h[0].header
            assert pri_hdr["PROGRAM"] == 1055
            assert pri_hdr["FILTER"] == "F146"
            assert pri_hdr["DETECTOR"] == "WFI02"
            assert np.isclose(pri_hdr["RA_OBJ"], 250.370832)
            assert np.isclose(pri_hdr["DEC_OBJ"], 36.505418)
            assert pri_hdr["TARGETID"] == "STAR_001"
            assert pri_hdr["NTIMES"] == nt

            # 2. Check LIGHTCURVE BinTableHDU
            tbl_hdu = h["LIGHTCURVE"]
            col_names = tbl_hdu.columns.names
            expected_cols = [
                "CADENCENO",
                "TIME",
                "PSF_FLUX",
                "PSF_FLUX_ERR",
                "PSF_CHI2",
                "PSF_QUALITY",
                "PSF_DROW",
                "PSF_DCOL",
                "PSF_ROW",
                "PSF_COL",
                "PSF_FIT_FLAG",
                "AP_ISO_FLUX",
                "AP_ISO_FLUX_ERR",
                "AP_ISO_BKG_FLUX",
                "AP_ISO_BKG_FLUX_ERR",
                "AP_ISO_FLFRCSAP",
                "AP_ISO_CROWDSAP",
                "AP_ISO_CENTROID_X",
                "AP_ISO_CENTROID_Y",
                "AP_ISO_QUALITY",
                "AP_3PIX_FLUX",
                "AP_3PIX_FLUX_ERR",
                "AP_3PIX_BKG_FLUX",
                "AP_3PIX_BKG_FLUX_ERR",
                "AP_3PIX_FLFRCSAP",
                "AP_3PIX_CROWDSAP",
                "AP_3PIX_CENTROID_X",
                "AP_3PIX_CENTROID_Y",
                "AP_3PIX_QUALITY",
            ]
            for col in expected_cols:
                assert col in col_names, f"Missing column {col}"
                assert len(tbl_hdu.data[col]) == nt

            # Check header summary metrics for apertures (user requested)
            assert "FLF_ISO" in tbl_hdu.header
            assert "CRW_ISO" in tbl_hdu.header
            assert "PIX_ISO" in tbl_hdu.header
            assert np.isclose(tbl_hdu.header["FLF_ISO"], 0.92, atol=1e-4)
            assert np.isclose(tbl_hdu.header["CRW_ISO"], 0.88, atol=1e-4)
            assert np.isclose(tbl_hdu.header["PIX_ISO"], 25.0, atol=1e-4)

            assert "FLF_3P" in tbl_hdu.header
            assert "CRW_3P" in tbl_hdu.header
            assert "PIX_3P" in tbl_hdu.header
            assert np.isclose(tbl_hdu.header["FLF_3P"], 0.85, atol=1e-4)
            assert np.isclose(tbl_hdu.header["CRW_3P"], 0.95, atol=1e-4)
            assert np.isclose(tbl_hdu.header["PIX_3P"], 9.0, atol=1e-4)

            # Precision metrics in primary and table headers
            for hdr_obj in (pri_hdr, tbl_hdu.header):
                assert "CDPP_PSF" in hdr_obj
                assert "RMS_PSF" in hdr_obj
                assert "P2P_PSF" in hdr_obj
                assert "MAD_PSF" in hdr_obj
                assert hdr_obj["CDPP_PSF"] > 0
                assert hdr_obj["RMS_PSF"] > 0

                assert "CDPP_ISO" in hdr_obj
                assert "RMS_ISO" in hdr_obj
                assert "P2P_ISO" in hdr_obj
                assert "MAD_ISO" in hdr_obj

                assert "CDPP_3P" in hdr_obj
                assert "RMS_3P" in hdr_obj
                assert "P2P_3P" in hdr_obj
                assert "MAD_3P" in hdr_obj

            # 3. Check APERTURE ImageHDU
            assert "APERTURE" in h
            assert h["APERTURE"].data.shape == (2, ny, nx)
            assert h["APERTURE"].header["APNAME0"] == "aperture_iso"
            assert h["APERTURE"].header["APNAME1"] == "aperture_3pix"

        # 4. Round-trip test via read_lightcurve_fits
        read_lcs, read_pri_hdr = read_lightcurve_fits(out_fits)
        assert "psf" in read_lcs
        assert "aperture_iso" in read_lcs
        assert "aperture_3pix" in read_lcs

        assert np.allclose(read_lcs["psf"]["time"], lcs["psf"]["time"])
        assert np.allclose(read_lcs["psf"]["flux"], lcs["psf"]["flux"])
        assert np.allclose(read_lcs["psf"]["flux_err"], lcs["psf"]["flux_err"])
        assert np.allclose(read_lcs["psf"]["drow"], lcs["psf"]["drow"])
        assert np.allclose(read_lcs["psf"]["dcol"], lcs["psf"]["dcol"])
        assert np.allclose(read_lcs["psf"]["row"], lcs["psf"]["row"])
        assert np.allclose(read_lcs["psf"]["column"], lcs["psf"]["column"])
        assert np.array_equal(read_lcs["psf"]["fit_flag"], lcs["psf"]["fit_flag"])

        assert np.allclose(read_lcs["aperture_iso"]["flux"], lcs["aperture_iso"]["flux"])
        assert np.allclose(read_lcs["aperture_iso"]["flfrcsap"], lcs["aperture_iso"]["flfrcsap"])
        assert np.allclose(read_lcs["aperture_iso"]["crowdsap"], lcs["aperture_iso"]["crowdsap"])
        assert np.array_equal(read_lcs["aperture_iso"]["aperture_mask"], mask_iso)

        assert np.allclose(read_lcs["aperture_3pix"]["flux"], lcs["aperture_3pix"]["flux"])
        assert np.array_equal(read_lcs["aperture_3pix"]["aperture_mask"], mask_3pix)


def test_extractor_to_fits(sample_extractor):
    sample_extractor.fit_prf_photometry(
        targets=[0],
        model_bkg=True,
        bkg_nknots=3,
        do_aperture=True,
        aperture_radius=3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_fits = os.path.join(tmpdir, "extractor_lc.fits")
        extra_meta = {
            "EPSFFILE": "roman_epsf_pid1055_wfi02_f146_global.fits",
            "CATFILE": "01055_sca02_clean_export.csv",
        }
        hdul = sample_extractor.to_fits(target=0, path=out_fits, extra_meta=extra_meta)
        assert os.path.isfile(out_fits)

        read_lcs, hdr = read_lightcurve_fits(out_fits)
        assert "psf" in read_lcs
        assert "aperture_snr" in read_lcs
        assert "aperture_iso" in read_lcs
        assert "aperture_3pix" in read_lcs
        assert "flux_corr" in read_lcs["aperture_snr"]
        assert "flux_err_corr" in read_lcs["aperture_snr"]
        assert len(read_lcs["psf"]["flux"]) == sample_extractor.nt

        # Check extra_meta propagation to primary header
        assert hdr["EPSFFILE"] == "roman_epsf_pid1055_wfi02_f146_global.fits"
        assert hdr["CATFILE"] == "01055_sca02_clean_export.csv"

        # Check precision metrics in primary header from extractor.to_fits()
        assert "CDPP_PSF" in hdr
        assert "RMS_PSF" in hdr
        assert "CDPP_SNR" in hdr
        assert "RMS_SNR" in hdr
        assert "CDPP_ISO" in hdr
        assert "CDPP_3P" in hdr


def test_cli_argument_parsing():
    args = parse_args([
        "--ra", "250.370832",
        "--dec", "36.505418",
        "--pid", "1055",
        "--filter", "F146",
        "--detector", "2",
        "--cutout-size", "25",
        "--aperture-radii", "3", "5",
        "--catalog", "/tmp/cat.csv",
        "--epsf", "/tmp/epsf.fits",
        "--image-list", "/tmp/images.txt",
        "--output-dir", "/tmp",
    ])
    assert np.isclose(args.ra, 250.370832)
    assert np.isclose(args.dec, 36.505418)
    assert args.pid == 1055
    assert args.filter == "F146"
    assert args.detector == "2"
    assert args.cutout_size == 25
    assert args.aperture_radii == [3.0, 5.0]
    assert args.model_bkg is True
    assert args.fit_offsets is True
    assert args.catalog == "/tmp/cat.csv"
    assert args.epsf == "/tmp/epsf.fits"
    assert args.image_list == "/tmp/images.txt"
    assert args.output_dir == "/tmp"

    # Verify --data-dir is no longer recognized
    with pytest.raises(SystemExit):
        parse_args(["--ra", "1.0", "--dec", "2.0", "--pid", "1055", "--filter", "F146", "--detector", "2", "--data-dir", "/tmp"])

    det_int, det_str = normalize_detector("WFI02")
    assert det_int == 2
    assert det_str == "wfi02"

    det_int2, det_str2 = normalize_detector(7)
    assert det_int2 == 7
    assert det_str2 == "wfi07"


def test_write_lightcurve_fits_s3_mandatory_bucket_name():
    lcs = {"psf": {"time": np.array([1.0, 2.0]), "flux": np.array([10.0, 11.0])}}

    # to_s3=True without bucket_name in kwargs should raise ValueError
    with pytest.raises(ValueError, match="`bucket_name` is a mandatory argument"):
        write_lightcurve_fits(lcs, output_path="lc.fits", to_s3=True)

    # Empty bucket_name should raise ValueError
    with pytest.raises(ValueError, match="`bucket_name` is a mandatory argument"):
        write_lightcurve_fits(lcs, output_path="lc.fits", to_s3=True, bucket_name="")

    with pytest.raises(ValueError, match="`bucket_name` is a mandatory argument"):
        write_lightcurve_fits(lcs, output_path="lc.fits", to_s3=True, bucket_name="   ")

    # s3:// path without bucket_name should also require bucket_name
    with pytest.raises(ValueError, match="`bucket_name` is a mandatory argument"):
        write_lightcurve_fits(lcs, output_path="s3://my-bucket/lc.fits")

    # Missing output_path when writing to S3
    with pytest.raises(ValueError, match="An `output_path` .* must be provided"):
        write_lightcurve_fits(lcs, output_path=None, to_s3=True, bucket_name="my-bucket")


def test_write_and_read_lightcurve_fits_s3_mock():
    import fsspec

    # Use in-memory filesystem to simulate S3FS
    mem_fs = fsspec.filesystem("memory")
    mem_fs.mkdir("roman-test-bucket")

    lcs = {
        "psf": {
            "time": np.array([2460001.0, 2460002.0, 2460003.0]),
            "flux": np.array([100.0, 102.5, 99.8], dtype=np.float32),
            "flux_err": np.array([1.0, 1.1, 1.0], dtype=np.float32),
        }
    }
    meta = {"PID": 1055, "FILTER": "F146", "DETECTOR": "WFI02", "TARGETID": "TEST_S3_01"}

    s3_path = "pid1055/lightcurves/test_lc.fits"
    hdul = write_lightcurve_fits(
        lcs=lcs,
        output_path=s3_path,
        meta=meta,
        to_s3=True,
        bucket_name="roman-test-bucket",
        s3_fs=mem_fs,
        overwrite=True,
    )
    assert hdul is not None
    assert mem_fs.exists("roman-test-bucket/pid1055/lightcurves/test_lc.fits")

    # Test overwrite=False raises FileExistsError
    with pytest.raises(FileExistsError, match="Target S3 file already exists"):
        write_lightcurve_fits(
            lcs=lcs,
            output_path=s3_path,
            meta=meta,
            to_s3=True,
            bucket_name="roman-test-bucket",
            s3_fs=mem_fs,
            overwrite=False,
        )

    # Read back from mock S3
    read_lcs, read_hdr = read_lightcurve_fits(
        filepath=f"s3://roman-test-bucket/{s3_path}",
        bucket_name="roman-test-bucket",
        s3_fs=mem_fs,
    )
    assert "psf" in read_lcs
    assert np.allclose(read_lcs["psf"]["time"], lcs["psf"]["time"])
    assert np.allclose(read_lcs["psf"]["flux"], lcs["psf"]["flux"])
    assert read_hdr["TARGETID"] == "TEST_S3_01"
    assert read_hdr["PROGRAM"] == 1055


def test_extractor_to_fits_s3(sample_extractor):
    import fsspec

    mem_fs = fsspec.filesystem("memory")
    mem_fs.mkdir("roman-extractor-bucket")

    sample_extractor.fit_prf_photometry(
        targets=[0],
        model_bkg=True,
        bkg_nknots=3,
        do_aperture=True,
        aperture_radius=3,
    )

    s3_key = "lightcurves/extractor_target0.fits"
    hdul = sample_extractor.to_fits(
        target=0,
        path=s3_key,
        to_s3=True,
        bucket_name="roman-extractor-bucket",
        s3_fs=mem_fs,
    )
    assert mem_fs.exists(f"roman-extractor-bucket/{s3_key}")

    read_lcs, hdr = read_lightcurve_fits(
        filepath=s3_key,
        bucket_name="roman-extractor-bucket",
        s3_fs=mem_fs,
    )
    assert "psf" in read_lcs
    assert len(read_lcs["psf"]["flux"]) == sample_extractor.nt

