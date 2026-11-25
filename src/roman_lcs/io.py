"""
Module for serializing and deserializing Roman light curves to and from FITS files.
Stores PSF and multiple aperture light curves within a unified binary table HDU.
"""

import datetime
import io
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

try:
    import s3fs
except ImportError:
    s3fs = None

from . import __version__
from .utils import compute_lightcurve_metrics

log = logging.getLogger(__name__)


def format_aperture_col_prefix(key: str) -> str:
    """
    Format aperture dictionary key to FITS column prefix.

    Examples
    --------
    >>> format_aperture_col_prefix("aperture_iso")
    'AP_ISO'
    >>> format_aperture_col_prefix("aperture_3pix")
    'AP_3PIX'
    """
    clean_key = key.strip()
    if clean_key.startswith("aperture_"):
        suffix = clean_key[len("aperture_"):].upper().replace(".", "_")
        return f"AP_{suffix}"
    return clean_key.upper().replace(".", "_")


def _get_short_aperture_id(key: str) -> str:
    """
    Generate an abbreviated identifier (<= 3 chars) for header keywords.

    Examples
    --------
    >>> _get_short_aperture_id("aperture_snr")
    'SNR'
    >>> _get_short_aperture_id("aperture_iso")
    'ISO'
    >>> _get_short_aperture_id("aperture_3pix")
    '3P'
    >>> _get_short_aperture_id("aperture_10pix")
    '10P'
    """
    if key in ("aperture_snr", "snr"):
        return "SNR"
    elif key in ("aperture_iso", "isophote", "iso"):
        return "ISO"
    elif key in ("aperture_opt", "optimal"):
        return "OP"
    elif key.startswith("aperture_") and key.endswith("pix"):
        num_part = key[len("aperture_"):-len("pix")].replace(".", "_")
        return f"{num_part}P"[:3]
    return key.replace("aperture_", "")[:3].upper()


def write_lightcurve_fits(
    lcs: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    cadenceno: Optional[np.ndarray] = None,
    overwrite: bool = True,
    to_s3: bool = False,
    s3_fs: Optional[Any] = None,
    s3_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> fits.HDUList:
    """
    Write a target light curve dictionary containing PSF and multiple aperture
    measurements to a standardized FITS file, optionally uploading to an AWS S3 bucket.

    Parameters
    ----------
    lcs : dict
        Light curve dictionary for a single target, containing keys such as
        'psf', 'aperture_snr', 'aperture_iso', 'aperture_3pix', etc.
    output_path : str, optional
        Filepath or S3 key where the FITS file will be saved.
    meta : dict, optional
        Observational metadata keywords to store in the PRIMARY header.
    cadenceno : np.ndarray, optional
        Cadence sequence index array. If None, uses np.arange(len(time)).
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    to_s3 : bool, optional
        Whether to write the output directly to an S3 bucket using `s3fs`. Default is False.
    s3_fs : s3fs.S3FileSystem or fsspec filesystem, optional
        Pre-configured S3/fsspec filesystem instance (useful for custom auth or testing).
    s3_kwargs : dict, optional
        Keyword arguments passed to `s3fs.S3FileSystem(...)` if `s3_fs` is None.
    **kwargs : Any
        Additional keyword arguments. Accepts `bucket_name` (str) as the mandatory AWS S3
        bucket name when writing to S3.

    Returns
    -------
    hdul : astropy.io.fits.HDUList
        Constructed HDUList containing PRIMARY, LIGHTCURVE, and optional APERTURE HDUs.
    """
    bucket_name = kwargs.pop("bucket_name", None)

    write_to_s3 = (
        to_s3
        or (bucket_name is not None)
        or (output_path is not None and str(output_path).startswith("s3://"))
    )

    if write_to_s3:
        if not bucket_name or not str(bucket_name).strip():
            raise ValueError("`bucket_name` is a mandatory argument when writing to S3.")
        if not output_path or not str(output_path).strip():
            raise ValueError("An `output_path` (S3 key or path) must be provided when writing to S3.")
    if meta is None:
        meta = {}

    # 1. Resolve shared time array
    time_arr = None
    for k in ("psf", "aperture_snr", "aperture_iso"):
        if k in lcs and "time" in lcs[k]:
            time_arr = np.asarray(lcs[k]["time"], dtype=np.float64)
            break
    if time_arr is None:
        for k, v in lcs.items():
            if isinstance(v, dict) and "time" in v:
                time_arr = np.asarray(v["time"], dtype=np.float64)
                break
    if time_arr is None:
        raise ValueError("No 'time' array found in any light curve component in `lcs`.")

    nt = len(time_arr)
    if cadenceno is None:
        cadenceno = np.arange(nt, dtype=np.int32)
    else:
        cadenceno = np.asarray(cadenceno, dtype=np.int32)

    # 2. Build Primary HDU with metadata
    pri_hdu = fits.PrimaryHDU()
    pri_header = pri_hdu.header

    pri_header["TELESCOP"] = (meta.get("TELESCOP", "Roman"), "Telescope name")
    pri_header["INSTRUME"] = (meta.get("INSTRUME", "WFI"), "Instrument name")
    pri_header["MISSION"] = (meta.get("MISSION", "Roman"), "Mission name")

    pid_val = meta.get("PROGRAM", meta.get("PID", meta.get("program_id", None)))
    if pid_val is not None:
        pri_header["PROGRAM"] = (int(pid_val), "Program/Observation identification number")
        pri_header["PID"] = (int(pid_val), "Program identification number")

    filter_val = meta.get("FILTER", meta.get("filter_name", None))
    if filter_val is not None:
        pri_header["FILTER"] = (str(filter_val).upper(), "Optical filter name")

    det_val = meta.get("DETECTOR", meta.get("SCA", None))
    if det_val is not None:
        pri_header["DETECTOR"] = (str(det_val), "Detector / Sensor Chip Assembly identifier")

    ra_val = meta.get("RA_OBJ", meta.get("RA", meta.get("ra", None)))
    if ra_val is not None:
        pri_header["RA_OBJ"] = (float(ra_val), "[deg] Target Right Ascension (ICRS)")

    dec_val = meta.get("DEC_OBJ", meta.get("DEC", meta.get("dec", None)))
    if dec_val is not None:
        pri_header["DEC_OBJ"] = (float(dec_val), "[deg] Target Declination (ICRS)")

    targetid = meta.get("TARGETID", meta.get("target_id", None))
    if targetid is not None:
        pri_header["TARGETID"] = (str(targetid), "Target catalog identifier")

    cutsize = meta.get("CUTSIZE", meta.get("cutout_size", None))
    if cutsize is not None:
        pri_header["CUTSIZE"] = (int(cutsize), "[pix] Cutout dimension")

    pri_header["CREATOR"] = ("trexs-roman-lcs", "Software of origin")
    pri_header["PROCVER"] = (str(__version__), "Pipeline processing software version")
    pri_header["DATE"] = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "UTC file creation date",
    )
    pri_header["TSTART"] = (float(np.nanmin(time_arr)), "Observation start time")
    pri_header["TEND"] = (float(np.nanmax(time_arr)), "Observation end time")
    pri_header["NTIMES"] = (int(nt), "Number of observation epochs")

    # Document PSF_FIT_FLAG bitmask
    pri_header.add_comment("PSF_FIT_FLAG bit definitions:")
    pri_header.add_comment("  Bit 0 (1): Offset did not reach convergence threshold")
    pri_header.add_comment("  Bit 1 (2): Offset fit reached maximum iterations")
    pri_header.add_comment("  Bit 2 (4): Linear model solve failed (LinAlgError)")
    pri_header.add_comment("  Bit 3 (8): Linear model solution non-finite")
    pri_header.add_comment("  Bit 4 (16): Insufficient unmasked pixels in cutout")
    pri_header.add_comment("  Bit 5 (32): Sub-pixel offset diverged / exceeded limit")
    pri_header.add_comment("  Bit 6 (64): Negative fitted target flux")
    pri_header.add_comment("  Bit 7 (128): High reduced chi-squared")

    # Store any extra scalar metadata
    standard_keys = {
        "TELESCOP", "INSTRUME", "MISSION", "PROGRAM", "PID", "FILTER",
        "DETECTOR", "SCA", "RA_OBJ", "DEC_OBJ", "RA", "DEC", "TARGETID",
        "CUTSIZE", "CREATOR", "PROCVER", "DATE", "TSTART", "TEND", "NTIMES",
    }
    for k, v in meta.items():
        k_upper = str(k).upper()[:8]
        if k_upper not in standard_keys and k_upper not in pri_header:
            if isinstance(v, (int, np.integer)):
                pri_header[k_upper] = int(v)
            elif isinstance(v, (float, np.floating)):
                pri_header[k_upper] = float(v)
            elif isinstance(v, (str, bool)):
                pri_header[k_upper] = v

    # 3. Assemble columns for LIGHTCURVE table
    cols = [
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(name="TIME", format="D", unit="d", array=time_arr),
    ]

    # PSF columns
    if "psf" in lcs:
        psf = lcs["psf"]
        if "flux" in psf:
            cols.append(fits.Column(name="PSF_FLUX", format="E", unit="DN/s", array=np.asarray(psf["flux"], dtype=np.float32)))
        if "flux_err" in psf:
            cols.append(fits.Column(name="PSF_FLUX_ERR", format="E", unit="DN/s", array=np.asarray(psf["flux_err"], dtype=np.float32)))
        if "chi2" in psf:
            cols.append(fits.Column(name="PSF_CHI2", format="E", array=np.asarray(psf["chi2"], dtype=np.float32)))
        if "quality" in psf:
            cols.append(fits.Column(name="PSF_QUALITY", format="J", array=np.asarray(psf["quality"], dtype=np.int32)))
        if "drow" in psf:
            cols.append(fits.Column(name="PSF_DROW", format="E", unit="pix", array=np.asarray(psf["drow"], dtype=np.float32)))
        if "dcol" in psf:
            cols.append(fits.Column(name="PSF_DCOL", format="E", unit="pix", array=np.asarray(psf["dcol"], dtype=np.float32)))
        if "row" in psf:
            cols.append(fits.Column(name="PSF_ROW", format="D", unit="pix", array=np.asarray(psf["row"], dtype=np.float64)))
        if "column" in psf:
            cols.append(fits.Column(name="PSF_COL", format="D", unit="pix", array=np.asarray(psf["column"], dtype=np.float64)))
        if "fit_flag" in psf:
            cols.append(fits.Column(name="PSF_FIT_FLAG", format="J", array=np.asarray(psf["fit_flag"], dtype=np.int32)))

    # Aperture columns & metadata for header
    ap_header_cards = []
    ap_masks = {}

    # Precision metrics for PSF light curve
    if "psf" in lcs and "flux" in lcs["psf"]:
        psf_flux = np.asarray(lcs["psf"]["flux"], dtype=np.float64)
        if np.any(np.isfinite(psf_flux)):
            try:
                psf_metrics = compute_lightcurve_metrics(psf_flux, time=time_arr)
                for m_key, fits_k, desc in [
                    ("cdpp", "CDPP_PSF", "CDPP noise proxy for PSF"),
                    ("rms", "RMS_PSF", "RMS scatter for PSF"),
                    ("p2p", "P2P_PSF", "P2P scatter for PSF"),
                    ("mad", "MAD_PSF", "MAD scatter for PSF"),
                ]:
                    val = float(psf_metrics[m_key])
                    comment = f"[ppm] {desc}"
                    pri_header[fits_k] = (val, comment)
                    ap_header_cards.append((fits_k, val, comment))
            except Exception as e:
                log.warning(f"Failed to compute precision metrics for PSF: {e}")

    ap_field_specs = [
        ("flux", "FLUX", "E", "DN/s"),
        ("flux_err", "FLUX_ERR", "E", "DN/s"),
        ("flux_corr", "FLUX_CORR", "E", "DN/s"),
        ("flux_err_corr", "FLUX_ERR_CORR", "E", "DN/s"),
        ("bkg_flux", "BKG_FLUX", "E", "DN/s"),
        ("bkg_flux_err", "BKG_FLUX_ERR", "E", "DN/s"),
        ("flfrcsap", "FLFRCSAP", "E", None),
        ("crowdsap", "CROWDSAP", "E", None),
        ("centroid_x", "CENTROID_X", "E", "pix"),
        ("centroid_y", "CENTROID_Y", "E", "pix"),
        ("quality", "QUALITY", "J", None),
    ]

    # Process all aperture types in deterministic order (aperture_snr first, then aperture_iso, then sorted circulars)
    ap_keys = [k for k in lcs.keys() if k.startswith("aperture_") and k != "aperture_opt"]
    if "aperture_snr" in ap_keys and "aperture_iso" in ap_keys:
        if lcs["aperture_snr"] is lcs["aperture_iso"]:
            ap_keys.remove("aperture_iso")
    ap_keys.sort(key=lambda x: (0 if x == "aperture_snr" else (1 if x == "aperture_iso" else 2), x))

    for ap_key in ap_keys:
        ap_dict = lcs[ap_key]
        prefix = format_aperture_col_prefix(ap_key)
        short_id = _get_short_aperture_id(ap_key)

        for src_field, col_suffix, fmt, unit in ap_field_specs:
            if src_field in ap_dict:
                data_val = ap_dict[src_field]
                col_name = f"{prefix}_{col_suffix}"
                dtype = np.int32 if fmt == "J" else np.float32
                cols.append(
                    fits.Column(
                        name=col_name,
                        format=fmt,
                        unit=unit,
                        array=np.asarray(data_val, dtype=dtype),
                    )
                )

        # Compute summary metrics for header cards
        if "flfrcsap" in ap_dict:
            mean_flfrc = float(np.nanmean(ap_dict["flfrcsap"]))
            ap_header_cards.append((f"FLF_{short_id}", mean_flfrc, f"Mean FLFRCSAP completeness for {ap_key}"))
        if "crowdsap" in ap_dict:
            mean_crowd = float(np.nanmean(ap_dict["crowdsap"]))
            ap_header_cards.append((f"CRW_{short_id}", mean_crowd, f"Mean CROWDSAP crowdedness for {ap_key}"))

        # Check for aperture mask
        if "aperture_mask" in ap_dict:
            mask_val = ap_dict["aperture_mask"]
            if isinstance(mask_val, (list, tuple)) and len(mask_val) > 0:
                mask_2d = np.asarray(mask_val[0], dtype=bool)
                ap_masks[ap_key] = mask_2d
                mean_npix = float(np.mean([np.sum(m) for m in mask_val]))
            elif isinstance(mask_val, np.ndarray):
                if mask_val.ndim == 2:
                    ap_masks[ap_key] = mask_val.astype(bool)
                    mean_npix = float(mask_val.sum())
                elif mask_val.ndim == 3:
                    ap_masks[ap_key] = mask_val[0].astype(bool)
                    mean_npix = float(np.mean(mask_val.sum(axis=(1, 2))))
                else:
                    mean_npix = float(mask_val.sum())
            else:
                mean_npix = 0.0
            ap_header_cards.append((f"PIX_{short_id}", mean_npix, f"Mean active pixels in {ap_key}"))

        # Compute precision metrics for aperture light curve
        if "flux_corr" in ap_dict and np.any(np.isfinite(ap_dict["flux_corr"])):
            target_flux = np.asarray(ap_dict["flux_corr"], dtype=np.float64)
        elif "flux" in ap_dict and np.any(np.isfinite(ap_dict["flux"])):
            target_flux = np.asarray(ap_dict["flux"], dtype=np.float64)
        else:
            target_flux = None

        if target_flux is not None:
            try:
                ap_metrics = compute_lightcurve_metrics(target_flux, time=time_arr)
                for m_key, prefix, desc in [
                    ("cdpp", "CDPP", f"CDPP for {short_id}"),
                    ("rms", "RMS", f"RMS scatter for {short_id}"),
                    ("p2p", "P2P", f"P2P scatter for {short_id}"),
                    ("mad", "MAD", f"MAD scatter for {short_id}"),
                ]:
                    fits_k = f"{prefix}_{short_id}"
                    val = float(ap_metrics[m_key])
                    comment = f"[ppm] {desc}"
                    pri_header[fits_k] = (val, comment)
                    ap_header_cards.append((fits_k, val, comment))
            except Exception as e:
                log.warning(f"Failed to compute precision metrics for {ap_key}: {e}")

    # 4. Construct LIGHTCURVE BinTableHDU
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = ("LIGHTCURVE", "Extension name")
    for card in ap_header_cards:
        table_hdu.header[card[0]] = (card[1], card[2])

    hdul = fits.HDUList([pri_hdu, table_hdu])

    # 5. Optional APERTURE ImageHDU for masks
    if ap_masks:
        mask_names = list(ap_masks.keys())
        stacked_masks = np.stack([ap_masks[k].astype(np.uint8) for k in mask_names], axis=0)
        mask_hdu = fits.ImageHDU(data=stacked_masks, name="APERTURE")
        for idx, k in enumerate(mask_names):
            mask_hdu.header[f"APNAME{idx}"] = (k, f"Aperture name for slice {idx}")
        hdul.append(mask_hdu)

    # 6. Write to disk or S3 if path specified
    if write_to_s3:
        bname = str(bucket_name).strip().strip("/")
        clean_path = str(output_path).strip()
        if clean_path.startswith("s3://"):
            clean_path = clean_path[len("s3://"):]

        if clean_path.startswith(f"{bname}/"):
            key = clean_path[len(f"{bname}/"):].lstrip("/")
        else:
            key = clean_path.lstrip("/")

        s3_key_path = f"{bname}/{key}" if key else bname

        if s3_fs is not None:
            fs = s3_fs
        else:
            if s3fs is None:
                raise ImportError(
                    "The `s3fs` package is required to write to S3. "
                    "Please install it (e.g. `pip install s3fs`)."
                )
            fs = s3fs.S3FileSystem(**(s3_kwargs or {}))

        if not overwrite and fs.exists(s3_key_path):
            raise FileExistsError(f"Target S3 file already exists and overwrite=False: s3://{s3_key_path}")

        buf = io.BytesIO()
        hdul.writeto(buf, checksum=True)
        buf.seek(0)
        with fs.open(s3_key_path, "wb") as f:
            f.write(buf.getvalue())
        log.info(f"Successfully wrote light curve FITS to s3://{s3_key_path}")

    elif output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        hdul.writeto(output_path, overwrite=overwrite, checksum=True)

    return hdul


def read_lightcurve_fits(
    filepath: Union[str, Any],
    s3_fs: Optional[Any] = None,
    s3_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Dict[str, Any]], fits.Header]:
    """
    Read a Roman light curve FITS file and reconstruct the nested light curve dictionary.
    Supports local filesystem paths, file-like objects, or S3 paths using `s3fs`.

    Parameters
    ----------
    filepath : str or file-like object
        Path to the FITS file, or an S3 key / URI.
    s3_fs : s3fs.S3FileSystem or fsspec filesystem, optional
        Pre-configured S3/fsspec filesystem instance.
    s3_kwargs : dict, optional
        Keyword arguments passed to `s3fs.S3FileSystem(...)` if `s3_fs` is None.
    **kwargs : Any
        Additional keyword arguments. Accepts `bucket_name` (str) if reading from S3
        and bucket is not included in URI.

    Returns
    -------
    lcs : dict
        Reconstructed light curve dictionary containing 'psf' and any aperture types.
    primary_header : astropy.io.fits.Header
        Header of the PRIMARY HDU containing metadata cards.
    """
    bucket_name = kwargs.pop("bucket_name", None)

    is_s3 = (
        s3_fs is not None
        or (bucket_name is not None)
        or (isinstance(filepath, str) and filepath.startswith("s3://"))
    )

    if is_s3:
        if s3_fs is not None:
            fs = s3_fs
        else:
            if s3fs is None:
                raise ImportError(
                    "The `s3fs` package is required to read from S3. "
                    "Please install it (e.g. `pip install s3fs`)."
                )
            fs = s3fs.S3FileSystem(**(s3_kwargs or {}))

        clean_path = str(filepath).strip()
        if clean_path.startswith("s3://"):
            clean_path = clean_path[len("s3://"):]

        if bucket_name:
            bname = str(bucket_name).strip().strip("/")
            if clean_path.startswith(f"{bname}/"):
                key = clean_path[len(f"{bname}/"):].lstrip("/")
            else:
                key = clean_path.lstrip("/")
            s3_path = f"{bname}/{key}" if key else bname
        else:
            s3_path = clean_path

        with fs.open(s3_path, "rb") as f:
            target_source = io.BytesIO(f.read())
    else:
        target_source = filepath

    with fits.open(target_source) as hdul:
        pri_header = hdul[0].header.copy()
        if "LIGHTCURVE" not in hdul:
            raise KeyError(f"Expected HDU 'LIGHTCURVE' not found in {filepath}.")

        table_data = hdul["LIGHTCURVE"].data
        col_names = [col.name for col in hdul["LIGHTCURVE"].columns]

        time_arr = table_data["TIME"]
        lcs: Dict[str, Dict[str, Any]] = {}

        # 1. PSF component
        if "PSF_FLUX" in col_names:
            lcs["psf"] = {
                "time": time_arr,
                "flux": table_data["PSF_FLUX"],
                "flux_err": table_data["PSF_FLUX_ERR"] if "PSF_FLUX_ERR" in col_names else None,
            }
            if "PSF_CHI2" in col_names:
                lcs["psf"]["chi2"] = table_data["PSF_CHI2"]
            if "PSF_QUALITY" in col_names:
                lcs["psf"]["quality"] = table_data["PSF_QUALITY"]
            if "PSF_DROW" in col_names:
                lcs["psf"]["drow"] = table_data["PSF_DROW"]
            if "PSF_DCOL" in col_names:
                lcs["psf"]["dcol"] = table_data["PSF_DCOL"]
            if "PSF_ROW" in col_names:
                lcs["psf"]["row"] = table_data["PSF_ROW"]
            if "PSF_COL" in col_names:
                lcs["psf"]["column"] = table_data["PSF_COL"]
            if "PSF_FIT_FLAG" in col_names:
                lcs["psf"]["fit_flag"] = table_data["PSF_FIT_FLAG"]

        # 2. Discover aperture prefixes from columns
        # Column format: AP_<SUFFIX>_<METRIC>
        aperture_prefixes = set()
        for c in col_names:
            if c.startswith("AP_") and c.endswith("_FLUX"):
                # e.g., AP_OPT_FLUX -> AP_OPT
                prefix = c[:-len("_FLUX")]
                aperture_prefixes.add(prefix)

        # 3. Read masks if present
        masks_by_name = {}
        if "APERTURE" in hdul:
            ap_hdu = hdul["APERTURE"]
            mask_data = ap_hdu.data
            for idx in range(mask_data.shape[0] if mask_data.ndim == 3 else 1):
                key_name = ap_hdu.header.get(f"APNAME{idx}", f"aperture_{idx}")
                slice_mask = mask_data[idx] if mask_data.ndim == 3 else mask_data
                masks_by_name[key_name] = slice_mask.astype(bool)

        # 4. Reconstruct aperture dictionaries
        for prefix in sorted(aperture_prefixes):
            # Convert AP_ISO / AP_OPT -> aperture_iso, AP_3PIX -> aperture_3pix
            suffix = prefix[len("AP_"):].lower()
            ap_key = "aperture_iso" if suffix in ("iso", "opt") else f"aperture_{suffix}"

            ap_dict: Dict[str, Any] = {
                "time": time_arr,
                "flux": table_data[f"{prefix}_FLUX"],
            }
            if f"{prefix}_FLUX_ERR" in col_names:
                ap_dict["flux_err"] = table_data[f"{prefix}_FLUX_ERR"]
            if f"{prefix}_FLUX_CORR" in col_names:
                ap_dict["flux_corr"] = table_data[f"{prefix}_FLUX_CORR"]
            if f"{prefix}_FLUX_ERR_CORR" in col_names:
                ap_dict["flux_err_corr"] = table_data[f"{prefix}_FLUX_ERR_CORR"]
            if f"{prefix}_BKG_FLUX" in col_names:
                ap_dict["bkg_flux"] = table_data[f"{prefix}_BKG_FLUX"]
            if f"{prefix}_BKG_FLUX_ERR" in col_names:
                ap_dict["bkg_flux_err"] = table_data[f"{prefix}_BKG_FLUX_ERR"]
            if f"{prefix}_FLFRCSAP" in col_names:
                ap_dict["flfrcsap"] = table_data[f"{prefix}_FLFRCSAP"]
            if f"{prefix}_CROWDSAP" in col_names:
                ap_dict["crowdsap"] = table_data[f"{prefix}_CROWDSAP"]
            if f"{prefix}_CENTROID_X" in col_names:
                ap_dict["centroid_x"] = table_data[f"{prefix}_CENTROID_X"]
            if f"{prefix}_CENTROID_Y" in col_names:
                ap_dict["centroid_y"] = table_data[f"{prefix}_CENTROID_Y"]
            if f"{prefix}_QUALITY" in col_names:
                ap_dict["quality"] = table_data[f"{prefix}_QUALITY"]

            if ap_key in masks_by_name:
                ap_dict["aperture_mask"] = masks_by_name[ap_key]
            elif ap_key == "aperture_iso" and "aperture_opt" in masks_by_name:
                ap_dict["aperture_mask"] = masks_by_name["aperture_opt"]

            lcs[ap_key] = ap_dict

    return lcs, pri_header
