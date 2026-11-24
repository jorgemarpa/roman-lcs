"""
Command-line script and extraction pipeline for Roman WFI stellar light curves.
Extracts multi-epoch cutouts, models PRF scenes, performs PRF and aperture
photometry, and saves standardized FITS light curves.
"""

import argparse
import glob
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from roman_cuts.com import make_cutout, query_mast_files
from roman_cuts.com.io import _open_asdf
from roman_epsf import load_epsf

from roman_lcs import LCExtractor, PhotutilsPSFAdapter
from roman_lcs.utils import convert_abmag_to_l2_flux

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roman_lc_extract")


class GWCSAdapter:
    """Adapts a GWCS object to provide .all_world2pix(coords, origin=0.0) -> (row, col)."""

    def __init__(self, gwcs):
        self._gwcs = gwcs

    def all_world2pix(self, coords, origin=0.0):
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        ra_arr, dec_arr = coords[:, 0], coords[:, 1]
        _coords = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")
        col_vals, row_vals = self._gwcs.world_to_pixel(_coords)
        return np.column_stack([row_vals, col_vals])

    def to_astropy_wcs(self):
        return WCS(self._gwcs.to_fits())


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Roman light curve extraction."""
    parser = argparse.ArgumentParser(
        description="Extract PRF and aperture light curves for a target from Roman WFI observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--ra", type=float, required=True, help="Target Right Ascension in degrees (ICRS).")
    parser.add_argument("--dec", type=float, required=True, help="Target Declination in degrees (ICRS).")
    parser.add_argument("--pid", type=int, required=True, help="Roman Program/Observation identification number (e.g. 1055).")
    parser.add_argument("--filter", type=str, required=True, help="Filter name (e.g. 'F146', case-insensitive).")
    parser.add_argument("--detector", type=str, required=True, help="Detector number or identifier (e.g. 2, '2', 'WFI02', 'SCA02').")

    # Optional extraction parameters
    parser.add_argument("--cutout-size", type=int, default=25, help="Cutout box size in native detector pixels.")
    parser.add_argument(
        "--aperture-radii",
        type=float,
        nargs="+",
        default=[3.0],
        help="One or more circular aperture radii in pixels.",
    )
    parser.add_argument(
        "--model-bkg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to model the spatial background signal.",
    )
    parser.add_argument(
        "--fit-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to iteratively fit sub-pixel centroid offsets using scene gradients.",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Dedicated path to source catalog file (CSV or Parquet).",
    )
    parser.add_argument(
        "--epsf",
        type=str,
        default=None,
        help="Dedicated path to photutils ePSF model FITS file.",
    )
    parser.add_argument(
        "--image-list",
        type=str,
        default=None,
        help="Path to text file containing paths of calibrated L2 ASDF images (one per line).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit path or filename for output FITS light curve file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Target folder where output light curves and figures will be saved.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of exposures to process (useful for tests).")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of worker processes for cutout extraction.")
    parser.add_argument("--plot", action="store_true", help="Generate and save diagnostic multi-panel figure.")

    return parser.parse_args(args)


def normalize_detector(detector_arg: Union[str, int]) -> Tuple[int, str]:
    """Normalize detector argument to integer index and string formatted names."""
    det_str = str(detector_arg).strip().upper()
    if det_str.startswith("WFI") or det_str.startswith("SCA"):
        det_int = int(det_str[-2:])
    else:
        det_int = int(det_str)
    return det_int, f"wfi{det_int:02d}"


def resolve_pid_dir(pid: int, data_dir: Optional[str] = None) -> str:
    """Resolve the project directory for a given PID from data_dir or environment defaults."""
    pid_str = f"pid{pid}"
    candidates = []
    if data_dir:
        norm = os.path.normpath(os.path.expanduser(data_dir))
        if os.path.basename(norm).lower() == pid_str:
            candidates.append(norm)
        else:
            candidates.append(os.path.join(norm, pid_str))
            candidates.append(norm)

    for env_var in ["ROMAN_DATA", "ROMAN_DATA_DIR"]:
        if os.environ.get(env_var):
            env_root = os.path.normpath(os.path.expanduser(os.environ[env_var]))
            if os.path.basename(env_root).lower() == pid_str:
                candidates.append(env_root)
            else:
                candidates.append(os.path.join(env_root, pid_str))
                candidates.append(env_root)

    candidates.extend([
        os.path.expanduser(f"~/Work/ROMAN/data/romcom/pid{pid}"),
        os.path.expanduser("~/Work/ROMAN/data/romcom"),
        os.path.expanduser(f"~/data/romcom/pid{pid}"),
        os.path.expanduser("~/data/romcom"),
        f"./pid{pid}",
        ".",
    ])

    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def find_l2_files(
    pid: int,
    detector_int: int,
    filter_name: str,
    data_dir: Optional[str] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    """Find local L2 calibrated ASDF images matching observation parameters or query MAST."""
    filter_clean = filter_name.strip().lower()
    candidates = []

    pid_dir = resolve_pid_dir(pid, data_dir=data_dir)
    search_dirs = [
        os.path.join(pid_dir, "l2"),
        pid_dir,
    ]
    if data_dir and os.path.isdir(data_dir) and data_dir not in search_dirs:
        search_dirs.append(data_dir)

    patterns = [
        f"r{pid:05}*_wfi{detector_int:02d}_{filter_clean}_cal.asdf",
        f"r{pid:05}*_sca{detector_int:02d}_{filter_clean}_cal.asdf",
        f"r{pid}*_wfi{detector_int:02d}_{filter_clean}_cal.asdf",
        f"r{pid}*_sca{detector_int:02d}_{filter_clean}_cal.asdf",
        f"*_wfi{detector_int:02d}_{filter_clean}_cal.asdf",
        f"*_sca{detector_int:02d}_{filter_clean}_cal.asdf",
    ]

    for root in search_dirs:
        if root and os.path.isdir(root):
            for pat in patterns:
                matches = glob.glob(os.path.join(root, pat))
                if matches:
                    candidates.extend(matches)
                matches_rec = glob.glob(os.path.join(root, "**", pat), recursive=True)
                if matches_rec:
                    candidates.extend(matches_rec)

    candidates = sorted(list(set(candidates)))

    if not candidates:
        log.info(f"No local L2 files found for PID {pid}, SCA {detector_int:02d}, filter {filter_name}. Querying MAST...")
        products_l2, _ = query_mast_files(
            program_id=pid,
            detector=detector_int,
            filter_name=filter_name.upper(),
            file_type="l2",
            execution_plan=1,
        )
        if len(products_l2) > 0:
            candidates = products_l2["filename"].tolist()

    if not candidates:
        raise FileNotFoundError(
            f"No L2 calibrated ASDF exposures found for PID {pid}, SCA {detector_int:02d}, filter {filter_name} in {pid_dir}."
        )

    if max_files is not None and max_files > 0:
        candidates = candidates[:max_files]

    log.info(f"Found {len(candidates)} L2 exposures to process.")
    return candidates


def find_epsf_path(
    pid: int,
    detector_int: int,
    filter_name: str,
    data_dir: Optional[str] = None,
) -> str:
    """Resolve path to photutils ePSF model FITS file within project directory."""
    filter_clean = filter_name.strip().lower()
    expected_fname = f"roman_epsf_pid{pid}_wfi{detector_int:02d}_{filter_clean}_global.fits"

    pid_dir = resolve_pid_dir(pid, data_dir=data_dir)
    search_dirs = [
        os.path.join(pid_dir, "epsf"),
        pid_dir,
    ]
    if data_dir and os.path.isdir(data_dir) and data_dir not in search_dirs:
        search_dirs.append(data_dir)

    # 1. Exact filename match
    for root in search_dirs:
        if root and os.path.isdir(root):
            p = os.path.join(root, expected_fname)
            if os.path.isfile(p):
                log.info(f"Found ePSF model at {p}")
                return p
            p_rec = glob.glob(os.path.join(root, "**", expected_fname), recursive=True)
            if p_rec:
                log.info(f"Found ePSF model at {p_rec[0]}")
                return p_rec[0]

    # 2. Relaxed glob matching
    patterns = [
        f"roman_epsf_pid{pid}_*wfi{detector_int:02d}*_{filter_clean}*.fits",
        f"roman_epsf_pid{pid}_*sca{detector_int:02d}*_{filter_clean}*.fits",
        f"*wfi{detector_int:02d}*_{filter_clean}*.fits",
        f"*sca{detector_int:02d}*_{filter_clean}*.fits",
        f"*wfi{detector_int:02d}*.fits",
        f"*sca{detector_int:02d}*.fits",
    ]
    for root in search_dirs:
        if root and os.path.isdir(root):
            for pat in patterns:
                matches = glob.glob(os.path.join(root, pat)) or glob.glob(os.path.join(root, "**", pat), recursive=True)
                if matches:
                    matches.sort()
                    log.info(f"Found ePSF model at {matches[0]}")
                    return matches[0]

    raise FileNotFoundError(
        f"Could not locate ePSF model '{expected_fname}' for PID {pid}, SCA {detector_int:02d}, filter {filter_name} in {pid_dir}."
    )


def find_catalog(
    pid: int,
    detector_int: int,
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load local source catalog matching PID and detector within project directory."""
    expected_name = f"{pid:05d}_sca{detector_int:02d}_clean_export.csv"
    pid_dir = resolve_pid_dir(pid, data_dir=data_dir)
    search_dirs = [
        os.path.join(pid_dir, "catalogs"),
        pid_dir,
    ]
    if data_dir and os.path.isdir(data_dir) and data_dir not in search_dirs:
        search_dirs.append(data_dir)

    # 1. Exact match
    for root in search_dirs:
        if root and os.path.isdir(root):
            p = os.path.join(root, expected_name)
            if os.path.isfile(p):
                log.info(f"Found catalog at {p}")
                return pd.read_csv(p), p
            p_rec = glob.glob(os.path.join(root, "**", expected_name), recursive=True)
            if p_rec:
                log.info(f"Found catalog at {p_rec[0]}")
                return pd.read_csv(p_rec[0]), p_rec[0]

    # 2. Relaxed matching for CSV or Parquet catalogs
    patterns = [
        f"*{pid:05d}*sca{detector_int:02d}*.csv",
        f"*{pid:05d}*wfi{detector_int:02d}*.csv",
        f"*{pid}*sca{detector_int:02d}*.csv",
        f"*{pid}*wfi{detector_int:02d}*.csv",
        f"*sca{detector_int:02d}*.csv",
        f"*wfi{detector_int:02d}*.csv",
        f"*sca{detector_int:02d}*.parquet",
        f"*wfi{detector_int:02d}*.parquet",
    ]
    for root in search_dirs:
        if root and os.path.isdir(root):
            for pat in patterns:
                matches = glob.glob(os.path.join(root, pat)) or glob.glob(os.path.join(root, "**", pat), recursive=True)
                if matches:
                    matches.sort()
                    match_path = matches[0]
                    log.info(f"Found catalog at {match_path}")
                    if match_path.endswith(".parquet"):
                        return pd.read_parquet(match_path), match_path
                    return pd.read_csv(match_path), match_path

    raise FileNotFoundError(
        f"Could not find catalog file '{expected_name}' for PID {pid}, SCA {detector_int:02d} in {pid_dir}."
    )


def plot_diagnostics(
    extractor: LCExtractor,
    target_idx: int,
    epsf_model: Any,
    output_png: str,
) -> None:
    """
    Generate and save a multi-panel diagnostic figure containing:
    1. Cutout reference image with catalog sources and target marked.
    2. ePSF model visualization.
    3. Three sample cadences of [Real Flux (Data), Model, Residuals].
    4. Light curve plots comparing PRF and aperture time series.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.2], hspace=0.35, wspace=0.3)

    # 1. Reference Cutout with Sources
    ax_cutout = fig.add_subplot(gs[0, 0])
    norm_cut = simple_norm(extractor.flux_3d[0], "asinh", percent=99.0)
    im0 = ax_cutout.imshow(extractor.flux_3d[0], origin="lower", cmap="viridis", norm=norm_cut)
    plt.colorbar(im0, ax=ax_cutout, fraction=0.046, pad=0.04, label="Flux [DN/s]")

    srow, scol = extractor.pixel_coordinates(frame_index=0)
    col_min = extractor.column[0].min()
    row_min = extractor.row[0].min()
    scol_plot = scol - col_min if scol.max() >= col_min else scol
    srow_plot = srow - row_min if srow.max() >= row_min else srow
    ax_cutout.scatter(scol_plot, srow_plot, marker=".", s=20, color="cyan", label="Sources")
    ax_cutout.scatter(scol_plot[target_idx], srow_plot[target_idx], marker="x", s=80, color="red", label="Target")
    ax_cutout.set_title("Reference Cutout (Epoch 0)")
    ax_cutout.set_xlabel("Pixel Column")
    ax_cutout.set_ylabel("Pixel Row")
    ax_cutout.legend(loc="upper right", fontsize=8)

    # 2. ePSF Model Profile
    ax_epsf = fig.add_subplot(gs[0, 1])
    norm_epsf = simple_norm(epsf_model.data, "log", percent=99.5)
    im_epsf = ax_epsf.imshow(epsf_model.data, origin="lower", cmap="magma", norm=norm_epsf)
    plt.colorbar(im_epsf, ax=ax_epsf, fraction=0.046, pad=0.04, label="ePSF Intensity")
    ax_epsf.set_title(f"ePSF Model ({epsf_model.oversampling[0]}x Oversampled)")
    ax_epsf.set_xlabel("Subpixel X")
    ax_epsf.set_ylabel("Subpixel Y")

    # 3. Model Scene on Epoch 0
    ax_scene = fig.add_subplot(gs[0, 2])
    im_scene = ax_scene.imshow(
        extractor.scene_model_cube[0].reshape(extractor.image_shape),
        origin="lower",
        cmap="viridis",
        norm=norm_cut,
    )
    plt.colorbar(im_scene, ax=ax_scene, fraction=0.046, pad=0.04, label="Flux [DN/s]")
    ax_scene.set_title("PRF Model Scene (Epoch 0)")
    ax_scene.set_xlabel("Pixel Column")
    ax_scene.set_ylabel("Pixel Row")

    # 4. Three Sample Cadences of [Data, Model, Residuals]
    # Pick 3 spread epochs
    nt = extractor.nt
    epochs = [0, nt // 2, max(0, nt - 1)]

    for col_idx, tdx in enumerate(epochs):
        ax_cad = fig.add_subplot(gs[1, col_idx])
        data_frame = extractor.flux_3d[tdx]
        model_frame = extractor.scene_model_cube[tdx].reshape(extractor.image_shape)
        residuals = data_frame - model_frame

        norm_res = simple_norm(residuals, "linear", percent=98.0)
        im_res = ax_cad.imshow(residuals, origin="lower", cmap="RdBu_r", norm=norm_res)
        plt.colorbar(im_res, ax=ax_cad, fraction=0.046, pad=0.04, label="Data - Model")
        ax_cad.set_title(f"Residuals Epoch {tdx} (MJD {extractor.time[tdx]:.2f})")
        ax_cad.set_xlabel("Pixel Column")
        ax_cad.set_ylabel("Pixel Row")

    # 5. Light Curves Time Series (Bottom Panel)
    ax_lc = fig.add_subplot(gs[2, :])
    t_lcs = extractor.lcs[target_idx]

    if "psf" in t_lcs:
        psf = t_lcs["psf"]
        psf_err = np.where(np.isfinite(psf["flux_err"]) & (psf["flux_err"] > 0), psf["flux_err"], 0.0)
        ax_lc.errorbar(
            psf["time"],
            psf["flux"],
            yerr=psf_err,
            fmt="o",
            ms=3.5,
            color="tab:blue",
            alpha=0.8,
            label="PRF Photometry",
        )

    colors = ["tab:orange", "tab:green", "tab:purple", "tab:red"]
    c_idx = 0
    seen_ap_dicts = []
    for k in sorted(t_lcs.keys()):
        if k.startswith("aperture_"):
            ap = t_lcs[k]
            if any(ap is prev for prev in seen_ap_dicts):
                continue
            seen_ap_dicts.append(ap)
            if k == "aperture_snr":
                label = "SNR Optimal Aperture"
            elif k == "aperture_iso":
                label = "Aperture Isophot"
            else:
                label = k.replace("aperture_", "Aperture ")
            ap_err = np.where(np.isfinite(ap["flux_err"]) & (ap["flux_err"] > 0), ap["flux_err"], 0.0)
            ax_lc.errorbar(
                ap["time"],
                ap["flux"],
                yerr=ap_err,
                fmt="s",
                ms=2.5,
                color=colors[c_idx % len(colors)],
                alpha=0.6,
                label=label,
            )
            c_idx += 1

    ax_lc.set_xlabel("Time [MJD]")
    ax_lc.set_ylabel("Flux [DN/s]")
    ax_lc.set_title(f"Target Light Curve Time Series (RA={extractor.sources.loc[target_idx, 'ra']:.5f}, Dec={extractor.sources.loc[target_idx, 'dec']:.5f})")
    ax_lc.grid(True, alpha=0.3)
    ax_lc.legend(loc="best", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved diagnostic figure to {output_png}")


def extract_lightcurve(
    ra: float,
    dec: float,
    pid: int,
    filter_name: str,
    detector: Union[str, int],
    cutout_size: int = 25,
    aperture_radii: Union[float, List[float]] = 3.0,
    model_bkg: bool = True,
    fit_offsets: bool = True,
    catalog: Optional[str] = None,
    epsf: Optional[str] = None,
    image_list: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_files: Optional[int] = None,
    n_workers: int = 8,
    plot: bool = False,
    to_s3: bool = False,
    **kwargs: Any,
) -> str:
    """
    Execute full light curve extraction for given target coordinates and observation metadata.
    """
    det_int, det_str = normalize_detector(detector)
    filter_upper = filter_name.strip().upper()
    filter_lower = filter_name.strip().lower()

    if isinstance(aperture_radii, (int, float)):
        aperture_radii = [float(aperture_radii)]

    log.info("=" * 60)
    log.info("Starting Roman Light Curve Extraction")
    log.info(f"Target Coordinates : RA={ra:.6f} deg, Dec={dec:.6f} deg")
    log.info(f"Observation Params : PID={pid}, Filter={filter_upper}, SCA={det_int:02d}")
    log.info(f"Cutout Dimension   : {cutout_size}x{cutout_size} pixels")
    log.info(f"Aperture Radii     : {aperture_radii} pixels")
    log.info(f"Fit Offsets        : {fit_offsets}")
    log.info(f"Model Background   : {model_bkg}")
    log.info("=" * 60)

    # 1. Locate L2 ASDF files
    if image_list is not None:
        if not os.path.isfile(image_list):
            raise FileNotFoundError(f"Image list file not found: {image_list}")
        with open(image_list, "r") as f:
            l2_files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not l2_files:
            raise ValueError(f"No valid image file paths found in {image_list}")
        if max_files is not None and max_files > 0:
            l2_files = l2_files[:max_files]
        log.info(f"Loaded {len(l2_files)} L2 exposures from image list file: {image_list}")
    else:
        l2_files = find_l2_files(
            pid=pid,
            detector_int=det_int,
            filter_name=filter_upper,
            max_files=max_files,
        )

    # 2. Extract photometric calibration metadata from reference exposure
    log.info(f"Reading photometric metadata from {os.path.basename(l2_files[0])}...")
    with _open_asdf(l2_files[0]) as af:
        phot_dict = af["roman"]["meta"]["photometry"].copy()

    # 3. Load catalog and convert magnitudes to L2 DN/s flux
    if catalog is not None:
        if not os.path.isfile(catalog):
            raise FileNotFoundError(f"Catalog file not found: {catalog}")
        resolved_cat_path = os.path.abspath(catalog)
        log.info(f"Loading catalog from {resolved_cat_path}")
        if resolved_cat_path.endswith(".parquet"):
            cat_df = pd.read_parquet(resolved_cat_path)
        else:
            cat_df = pd.read_csv(resolved_cat_path)
    else:
        cat_df, resolved_cat_path = find_catalog(
            pid=pid,
            detector_int=det_int,
        )

    mag_col = f"{filter_lower}_mag"
    flux_col = f"{filter_lower}_flux"
    if mag_col in cat_df.columns and flux_col not in cat_df.columns:
        cat_df[flux_col] = convert_abmag_to_l2_flux(cat_df[mag_col], meta=phot_dict)

    # 4. Extract cutouts using roman-cuts
    log.info(f"Extracting {cutout_size}x{cutout_size} cutouts across {len(l2_files)} exposures...")
    cutout_cubes, metadatas, gwcsss = make_cutout(
        l2_files,
        ra=ra,
        dec=dec,
        size=(cutout_size, cutout_size),
        lite=False,
        n_workers=n_workers,
    )

    cube = cutout_cubes[0]
    meta_df = metadatas[0]
    gwcs_list = gwcsss[0]

    # Convert mid_time to numerical JD / MJD
    meta_df["mid_time"] = np.array([x.jd for x in meta_df["mid_time"]])

    # Filter empty / out-of-bounds epochs
    keep = meta_df["in_bounds"].values
    cube = cube[:, keep]
    meta_df = meta_df[keep]
    gwcs_list = [item for item, kp in zip(gwcs_list, keep) if kp]

    nt = cube.shape[1]
    log.info(f"Retained {nt} valid in-bounds exposures.")
    if nt == 0:
        raise ValueError("Target coordinate is outside detector bounds for all provided exposures.")

    # 5. Load and adapt ePSF model
    if epsf is not None:
        if not os.path.isfile(epsf):
            raise FileNotFoundError(f"ePSF model file not found: {epsf}")
        resolved_epsf_path = os.path.abspath(epsf)
        log.info(f"Loading ePSF model from {resolved_epsf_path}")
    else:
        resolved_epsf_path = find_epsf_path(
            pid=pid,
            detector_int=det_int,
            filter_name=filter_upper,
        )
    epsf_model, _ = load_epsf(resolved_epsf_path)
    psf_adapter = PhotutilsPSFAdapter(epsf_model)

    # 6. Filter local catalog sources within cutout footprint
    ref_gwcs = gwcs_list[0]
    src_coords = SkyCoord(
        ra=cat_df["ra"].to_numpy() * u.deg,
        dec=cat_df["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    col_src, row_src = ref_gwcs.world_to_pixel(src_coords)
    margin = 2.0
    in_cutout = (
        (col_src >= -margin)
        & (col_src < cutout_size + margin)
        & (row_src >= -margin)
        & (row_src < cutout_size + margin)
    )
    if flux_col in cat_df.columns:
        in_cutout &= np.isfinite(cat_df[flux_col].to_numpy()) & (cat_df[flux_col].to_numpy() > 0)

    cutout_sources = cat_df[in_cutout].copy().reset_index(drop=True)
    if len(cutout_sources) == 0:
        raise ValueError("No catalog sources found within the cutout footprint.")

    # Locate target star index in cutout catalog
    target_dist = np.hypot(cutout_sources["ra"] - ra, cutout_sources["dec"] - dec)
    target_idx = int(np.argmin(target_dist))
    log.info(
        f"Located target star at index {target_idx} (separation: {np.degrees(target_dist[target_idx])*3600:.2f} arcsec)."
    )

    # 7. Assemble 3D arrays for LCExtractor
    flux_3d = cube[0].copy()
    flux_err_3d = cube[1].copy()

    col_3d = np.zeros((nt, cutout_size, cutout_size), dtype=np.float64)
    row_3d = np.zeros((nt, cutout_size, cutout_size), dtype=np.float64)
    ra_3d = np.zeros((nt, cutout_size, cutout_size), dtype=np.float64)
    dec_3d = np.zeros((nt, cutout_size, cutout_size), dtype=np.float64)

    cutout_col_grid, cutout_row_grid = np.meshgrid(
        np.arange(cutout_size, dtype=np.float64),
        np.arange(cutout_size, dtype=np.float64),
        indexing="xy",
    )

    for t in range(nt):
        row_t = meta_df.iloc[t]
        orig_y = row_t["input_position_original"][0] - row_t["input_position_cutout"][0]
        orig_x = row_t["input_position_original"][1] - row_t["input_position_cutout"][1]
        col_3d[t] = orig_x + cutout_col_grid
        row_3d[t] = orig_y + cutout_row_grid

        grid = gwcs_list[t].pixel_to_world(cutout_col_grid, cutout_row_grid)
        ra_3d[t], dec_3d[t] = grid.ra.deg, grid.dec.deg

    if "roman_source_id" in cutout_sources.columns:
        target_source_id = cutout_sources.loc[target_idx, "roman_source_id"]
    else:
        target_source_id = str(target_idx)

    adapted_wcss = [GWCSAdapter(g) for g in gwcs_list]
    metadata: Dict[str, Any] = {
        "SCA": det_int,
        "FILTER": filter_upper,
        "DETECTOR": f"WFI{det_int:02d}",
        "MISSION": "Roman",
        "PID": pid,
        "PROGRAM": pid,
        "CUTSIZE": cutout_size,
        "RA_OBJ": ra,
        "DEC_OBJ": dec,
        "TARGETID": target_source_id,
    }

    # 8. Instantiate LCExtractor
    extractor = LCExtractor(
        time=meta_df["mid_time"].to_numpy(),
        flux=flux_3d,
        flux_err=flux_err_3d,
        ra=ra_3d,
        dec=dec_3d,
        sources=cutout_sources,
        column=col_3d,
        row=row_3d,
        wcs=adapted_wcss,
        meta=metadata,
        sources_flux_column=flux_col if flux_col in cutout_sources.columns else "flux",
        sources_mag_column=mag_col if mag_col in cutout_sources.columns else "mag",
        dithered=True,
    )
    extractor.prf_model = psf_adapter

    # 9. Perform Photometry
    log.info("Fitting PRF and aperture photometry...")
    extractor.fit_prf_photometry(
        targets=[target_idx],
        model_bkg=model_bkg,
        bkg_nknots=3,
        fit_offsets=fit_offsets,
        do_aperture=True,
        aperture_radius=aperture_radii,
    )

    # 10. Save light curve to FITS file using LCExtractor.to_fits
    target_folder = output_dir if output_dir is not None else "."
    os.makedirs(target_folder, exist_ok=True)

    if output_path is None:
        fname = f"roman_wfi_pid{pid}_wfi{det_int:02d}_{filter_upper}_ra{ra:.5f}_dec{dec:.5f}_lc.fits"
        output_path = os.path.join(target_folder, fname)
    elif not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
        output_path = os.path.join(target_folder, output_path)

    extra_meta: Dict[str, Any] = {
        "EPSFFILE": os.path.basename(resolved_epsf_path),
        "EPSFPATH": resolved_epsf_path,
        "CATFILE": os.path.basename(resolved_cat_path),
        "CATPATH": resolved_cat_path,
    }

    extractor.to_fits(
        target=target_idx,
        path=output_path,
        overwrite=True,
        extra_meta=extra_meta,
        to_s3=to_s3,
        **kwargs,
    )
    log.info(f"Successfully saved light curve FITS to {output_path}")

    # 11. Diagnostic plot
    if plot:
        fname_base = os.path.splitext(os.path.basename(output_path))[0]
        plot_path = os.path.join(target_folder, f"{fname_base}_diag.png")
        plot_diagnostics(
            extractor=extractor,
            target_idx=target_idx,
            epsf_model=epsf_model,
            output_png=plot_path,
        )

    return output_path


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    try:
        extract_lightcurve(
            ra=args.ra,
            dec=args.dec,
            pid=args.pid,
            filter_name=args.filter,
            detector=args.detector,
            cutout_size=args.cutout_size,
            aperture_radii=args.aperture_radii,
            model_bkg=args.model_bkg,
            fit_offsets=args.fit_offsets,
            catalog=args.catalog,
            epsf=args.epsf,
            image_list=args.image_list,
            output_path=args.output,
            output_dir=args.output_dir,
            max_files=args.max_files,
            n_workers=args.n_workers,
            plot=args.plot,
        )
    except Exception as e:
        log.error(f"Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
