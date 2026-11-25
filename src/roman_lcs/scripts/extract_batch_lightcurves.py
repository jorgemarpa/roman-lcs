"""
Command-line script and parallel batch extraction pipeline for Roman WFI stellar light curves.

Processes a batch of targets simultaneously:
1. Skips targets whose light curve FITS file already exists (unless --overwrite is set).
2. Performs multi-target cutout slicing in one go using roman_cuts.com.make_cutout,
   reading each L2 ASDF exposure only once across all targets.
3. Loads the global ePSF model once in the parent process.
4. Distributes individual target PRF & aperture light curve extractions across a configurable
   number of single-threaded CPU worker processes.
5. Saves per-target FITS light curves, per-target execution logs, and selective diagnostic plots.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

# Enforce strict single-threaded execution per worker before importing numpy/scipy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from roman_cuts.com import make_cutout
from roman_cuts.com.io import _open_asdf
from roman_epsf import load_epsf

from roman_lcs import LCExtractor, PhotutilsPSFAdapter
from roman_lcs.scripts.extract_lightcurve import (
    GWCSAdapter,
    find_catalog,
    find_epsf_path,
    find_l2_files,
    normalize_detector,
    plot_diagnostics,
    resolve_pid_dir,
)
from roman_lcs.utils import convert_abmag_to_l2_flux

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("roman_lc_extract_batch")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for Roman batch light curve extraction."""
    parser = argparse.ArgumentParser(
        description="Extract PRF and aperture light curves for a batch of targets from Roman WFI observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Batch target file & observation identity
    parser.add_argument(
        "-f",
        "--targets-file",
        type=str,
        required=True,
        help="Path to target list text file (format: ra dec cutout_size plot_flag target_id).",
    )
    parser.add_argument("--pid", type=int, required=True, help="Roman Program/Observation identification number (e.g. 1055).")
    parser.add_argument("--filter", type=str, required=True, help="Filter name (e.g. 'F146', case-insensitive).")
    parser.add_argument("--detector", type=str, required=True, help="Detector number or identifier (e.g. 2, '2', 'WFI02', 'SCA02').")

    # Worker allocation
    parser.add_argument(
        "-j",
        "--n-extract-workers",
        type=int,
        default=8,
        help="Number of parallel single-CPU worker processes for LCExtractor fitting.",
    )
    parser.add_argument(
        "--n-cutout-workers",
        type=int,
        default=8,
        help="Number of concurrent I/O threads/workers for make_cutout.",
    )

    # Optional extraction parameters
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
        help="Text file containing list of L2 ASDF paths, one per line.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Project data directory (default: ROMAN_DATA_DIR env or /Users/jimartin/Work/ROMAN/data/romcom).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Target folder where light curves will be saved.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Output directory for per-target logs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=None,
        help="Output directory for diagnostic plots.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of L2 exposures to process (useful for quick verification).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Limit number of targets to extract from the batch file.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing light curve FITS files. By default, existing targets are skipped.",
    )

    return parser.parse_args(args)


def load_targets_file(targets_path: str, max_targets: Optional[int] = None) -> pd.DataFrame:
    """Read space-separated target list text file."""
    rows = []
    with open(targets_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ra = float(parts[0])
            dec = float(parts[1])
            cutout_size = int(parts[2]) if len(parts) >= 3 else 25
            plot_flag = int(parts[3]) if len(parts) >= 4 else 0
            target_id = parts[4] if len(parts) >= 5 else f"TARGET_{ra:.5f}_{dec:.5f}"
            rows.append({
                "ra": ra,
                "dec": dec,
                "cutout_size": cutout_size,
                "plot_flag": plot_flag,
                "target_id": target_id,
            })
            if max_targets is not None and len(rows) >= max_targets:
                break
    return pd.DataFrame(rows)


def _process_single_target_worker(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function executed in an isolated process on a single CPU core.
    Configures a per-target log file, runs LCExtractor, and outputs results.
    """
    t_start = time.time()
    target_id = bundle["target_id"]
    ra = bundle["ra"]
    dec = bundle["dec"]
    cutout_size = bundle["cutout_size"]
    plot_flag = bundle["plot_flag"]
    output_fits_path = bundle["output_fits_path"]
    log_file_path = bundle["log_file_path"]
    figure_path = bundle["figure_path"]

    # 1. Setup isolated file logging for this target
    os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
    target_logger = logging.getLogger(f"target_{target_id}")
    target_logger.setLevel(logging.INFO)
    target_logger.handlers.clear()
    fh = logging.FileHandler(log_file_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    target_logger.addHandler(fh)

    try:
        target_logger.info(f"Starting extraction for target {target_id} (RA={ra:.6f}, Dec={dec:.6f})")
        
        cube = bundle["cube"]
        meta_df = bundle["meta_df"]
        gwcs_list = bundle["gwcs_list"]
        cat_df = bundle["cat_df"]
        epsf_model = bundle["epsf_model"]
        flux_col = bundle["flux_col"]
        mag_col = bundle["mag_col"]
        model_bkg = bundle["model_bkg"]
        fit_offsets = bundle["fit_offsets"]
        aperture_radii = bundle["aperture_radii"]
        meta_base = bundle["meta_base"]

        # Filter empty / out-of-bounds exposures
        keep = meta_df["in_bounds"].values
        cube = cube[:, keep]
        meta_df = meta_df[keep].reset_index(drop=True)
        gwcs_list = [item for item, kp in zip(gwcs_list, keep) if kp]

        nt = cube.shape[1]
        target_logger.info(f"Retained {nt} valid in-bounds exposures.")
        if nt == 0:
            target_logger.warning("Target is out of bounds for all provided exposures. Skipping.")
            fh.close()
            return {"target_id": target_id, "status": "skipped_out_of_bounds", "nt": 0, "elapsed": time.time() - t_start}

        # Adapt ePSF model
        psf_adapter = PhotutilsPSFAdapter(epsf_model)

        # Filter catalog sources within cutout footprint
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
            target_logger.error("No catalog sources found within cutout footprint.")
            fh.close()
            return {"target_id": target_id, "status": "failed_no_sources", "nt": nt, "elapsed": time.time() - t_start}

        # Locate target star index in cutout catalog
        target_dist = np.hypot(cutout_sources["ra"] - ra, cutout_sources["dec"] - dec)
        target_idx = int(np.argmin(target_dist))
        target_logger.info(
            f"Target star index {target_idx} (separation: {np.degrees(target_dist[target_idx])*3600:.3f} arcsec), "
            f"{len(cutout_sources)} total catalog sources in cutout."
        )

        # Assemble 3D arrays
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

        for t_idx in range(nt):
            row_t = meta_df.iloc[t_idx]
            orig_y = row_t["input_position_original"][0] - row_t["input_position_cutout"][0]
            orig_x = row_t["input_position_original"][1] - row_t["input_position_cutout"][1]
            col_3d[t_idx] = orig_x + cutout_col_grid
            row_3d[t_idx] = orig_y + cutout_row_grid

            grid = gwcs_list[t_idx].pixel_to_world(cutout_col_grid, cutout_row_grid)
            ra_3d[t_idx], dec_3d[t_idx] = grid.ra.deg, grid.dec.deg

        adapted_wcss = [GWCSAdapter(g) for g in gwcs_list]
        metadata = dict(meta_base)
        metadata["CUTSIZE"] = cutout_size
        metadata["RA_OBJ"] = ra
        metadata["DEC_OBJ"] = dec
        metadata["TARGETID"] = target_id

        # Initialize LCExtractor
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

        # Run photometry
        target_logger.info("Fitting PRF and aperture photometry...")
        extractor.fit_prf_photometry(
            targets=[target_idx],
            model_bkg=model_bkg,
            bkg_nknots=3,
            fit_offsets=fit_offsets,
            do_aperture=True,
            aperture_radius=aperture_radii,
        )

        # Save light curve FITS
        os.makedirs(os.path.dirname(os.path.abspath(output_fits_path)), exist_ok=True)
        extractor.to_fits(
            target=target_idx,
            path=output_fits_path,
            overwrite=True,
        )
        target_logger.info(f"Successfully saved FITS light curve to {output_fits_path}")

        # Optional diagnostic plot
        if plot_flag:
            target_logger.info(f"Generating diagnostic figure at {figure_path}...")
            os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
            plot_diagnostics(
                extractor=extractor,
                target_idx=target_idx,
                epsf_model=epsf_model,
                output_png=figure_path,
            )
            target_logger.info("Diagnostic figure saved.")

        elapsed = time.time() - t_start
        target_logger.info(f"Completed target {target_id} in {elapsed:.2f}s.")
        fh.close()
        return {
            "target_id": target_id,
            "status": "success",
            "nt": nt,
            "elapsed": elapsed,
            "output_fits": output_fits_path,
        }

    except Exception as e:
        target_logger.exception(f"Error during extraction of {target_id}: {e}")
        fh.close()
        return {
            "target_id": target_id,
            "status": "failed",
            "error": str(e),
            "elapsed": time.time() - t_start,
        }


def extract_batch_lightcurves(
    targets_file: str,
    pid: int,
    filter_name: str,
    detector: Union[str, int],
    n_extract_workers: int = 8,
    n_cutout_workers: int = 8,
    aperture_radii: Union[float, List[float]] = 3.0,
    model_bkg: bool = True,
    fit_offsets: bool = True,
    catalog: Optional[str] = None,
    epsf: Optional[str] = None,
    image_list: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    figure_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    max_files: Optional[int] = None,
    max_targets: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Execute batch light curve extraction for a list of targets."""
    t0_batch = time.time()
    filter_upper = filter_name.strip().upper()
    det_int, _ = normalize_detector(detector)
    pid_dir = resolve_pid_dir(pid, data_dir=data_dir)

    # 1. Setup Output Directories
    base_out = output_dir if output_dir else os.path.join(pid_dir, "lightcurves")
    base_logs = log_dir if log_dir else os.path.join(base_out, "logs")
    base_figs = figure_dir if figure_dir else os.path.join(pid_dir, "figures")
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(base_logs, exist_ok=True)
    os.makedirs(base_figs, exist_ok=True)

    # 2. Load Targets File
    targets_df = load_targets_file(targets_file, max_targets=max_targets)
    n_total = len(targets_df)
    log.info(f"Loaded {n_total:,} targets from {targets_file}")
    if n_total == 0:
        log.warning("No targets to process in targets file. Exiting.")
        return

    # 3. Check Skip Policy (skip already extracted targets unless overwrite=True)
    pending_targets = []
    skipped_count = 0

    for _, row in targets_df.iterrows():
        tid = str(row["target_id"])
        expected_fits = os.path.join(base_out, f"{tid}_lc.fits")
        # Also check coordinate standard filename
        alt_fits = os.path.join(
            base_out,
            f"roman_wfi_pid{pid}_wfi{det_int:02d}_{filter_upper}_ra{row['ra']:.5f}_dec{row['dec']:.5f}_lc.fits",
        )
        if not overwrite and (
            (os.path.isfile(expected_fits) and os.path.getsize(expected_fits) > 0)
            or (os.path.isfile(alt_fits) and os.path.getsize(alt_fits) > 0)
        ):
            skipped_count += 1
        else:
            pending_targets.append(row.to_dict())

    log.info(
        f"Skip policy check: {skipped_count:,} targets already extracted, "
        f"{len(pending_targets):,} targets pending extraction."
    )
    if len(pending_targets) == 0:
        log.info("All targets have already been extracted. Nothing to do.")
        return

    # 4. Resolve L2 ASDF Exposures
    if image_list is not None and os.path.isfile(image_list):
        with open(image_list, "r") as f:
            l2_files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        log.info(f"Loaded {len(l2_files)} exposures from image list: {image_list}")
    else:
        l2_files = find_l2_files(
            pid=pid,
            detector_int=det_int,
            filter_name=filter_upper,
            data_dir=data_dir,
            max_files=max_files,
        )

    if not l2_files:
        raise FileNotFoundError(f"No L2 images found for PID {pid}, SCA {det_int:02d}, filter {filter_upper}.")

    # 5. Load Global ePSF Model Once
    if epsf is not None and os.path.isfile(epsf):
        epsf_path = os.path.abspath(epsf)
    else:
        epsf_path = find_epsf_path(
            pid=pid,
            detector_int=det_int,
            filter_name=filter_upper,
            data_dir=data_dir,
        )
    log.info(f"Loading global ePSF model from {epsf_path}...")
    epsf_model, _ = load_epsf(epsf_path)

    # 6. Load Source Catalog Once & Precalculate L2 Fluxes
    if catalog is not None and os.path.isfile(catalog):
        cat_df = pd.read_csv(catalog, low_memory=False)
        cat_path = catalog
    else:
        cat_df, cat_path = find_catalog(pid=pid, detector_int=det_int, data_dir=data_dir)
    log.info(f"Loaded source catalog with {len(cat_df):,} sources from {cat_path}")

    # Read reference photometric zero point
    with _open_asdf(l2_files[0]) as af:
        phot_meta = af["roman"]["meta"].get("photometry", {})
        phot_dict = {
            "conversion_megajanskys": phot_meta.get("conversion_megajanskys", 1.0),
            "pixelarea_steradians": phot_meta.get("pixelarea_steradians", 1.0),
        }

    filter_lower = filter_upper.lower()
    mag_col = f"{filter_lower}_mag"
    flux_col = f"{filter_lower}_flux"
    if mag_col in cat_df.columns and flux_col not in cat_df.columns:
        cat_df[flux_col] = convert_abmag_to_l2_flux(cat_df[mag_col], meta=phot_dict)

    # 7. Batch Cutout Slicing across all pending targets
    target_ras = [t["ra"] for t in pending_targets]
    target_decs = [t["dec"] for t in pending_targets]
    target_sizes = [int(t["cutout_size"]) for t in pending_targets]

    log.info(
        f"Extracting cutouts for {len(pending_targets):,} targets simultaneously "
        f"across {len(l2_files)} exposures using {n_cutout_workers} I/O workers..."
    )
    t0_cutouts = time.time()
    cutout_cubes, metadatas, gwcsss = make_cutout(
        l2_files,
        ra=target_ras,
        dec=target_decs,
        size=target_sizes,
        lite=False,
        n_workers=n_cutout_workers,
    )
    log.info(f"Completed batch cutouts in {time.time() - t0_cutouts:.2f}s.")

    # 8. Prepare Target Worker Bundles
    meta_base: Dict[str, Any] = {
        "SCA": det_int,
        "FILTER": filter_upper,
        "DETECTOR": f"WFI{det_int:02d}",
        "MISSION": "Roman",
        "PID": pid,
        "PROGRAM": pid,
    }

    worker_bundles = []
    for idx, t_info in enumerate(pending_targets):
        tid = str(t_info["target_id"])
        out_fits = os.path.join(base_out, f"{tid}_lc.fits")
        out_log = os.path.join(base_logs, f"{tid}.log")
        out_fig = os.path.join(base_figs, f"{tid}_diag.png")

        meta_df_t = metadatas[idx].copy()
        # Convert mid_time to numerical JD / MJD
        meta_df_t["mid_time"] = np.array([x.jd for x in meta_df_t["mid_time"]])

        bundle = {
            "target_id": tid,
            "ra": float(t_info["ra"]),
            "dec": float(t_info["dec"]),
            "cutout_size": int(t_info["cutout_size"]),
            "plot_flag": int(t_info["plot_flag"]),
            "cube": cutout_cubes[idx],
            "meta_df": meta_df_t,
            "gwcs_list": gwcsss[idx],
            "cat_df": cat_df,
            "epsf_model": epsf_model,
            "flux_col": flux_col,
            "mag_col": mag_col,
            "model_bkg": model_bkg,
            "fit_offsets": fit_offsets,
            "aperture_radii": aperture_radii,
            "meta_base": meta_base,
            "output_fits_path": out_fits,
            "log_file_path": out_log,
            "figure_path": out_fig,
        }
        worker_bundles.append(bundle)

    # 9. Parallel Single-CPU Worker Pool Dispatch
    log.info(
        f"Dispatching {len(worker_bundles):,} targets to {n_extract_workers} "
        f"single-CPU worker processes..."
    )
    t0_fit = time.time()
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=n_extract_workers) as executor:
        futures = {executor.submit(_process_single_target_worker, b): b["target_id"] for b in worker_bundles}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                res = future.result()
                if res.get("status") == "success":
                    success_count += 1
                elif res.get("status", "").startswith("skipped"):
                    log.warning(f"Target {tid}: {res.get('status')}")
                else:
                    fail_count += 1
                    log.error(f"Target {tid} extraction failed: {res.get('error', 'unknown error')}")
            except Exception as exc:
                fail_count += 1
                log.error(f"Target {tid} generated worker exception: {exc}")

    t_fit_elapsed = time.time() - t0_fit
    total_elapsed = time.time() - t0_batch
    log.info("=" * 60)
    log.info("BATCH EXTRACTION COMPLETE")
    log.info(f"Total targets in batch : {n_total:,}")
    log.info(f"Previously completed   : {skipped_count:,}")
    log.info(f"Extracted successfully : {success_count:,}")
    log.info(f"Failed extractions     : {fail_count:,}")
    log.info(f"Fitting stage time     : {t_fit_elapsed:.2f}s")
    log.info(f"Total pipeline time    : {total_elapsed:.2f}s")
    if success_count > 0:
        log.info(f"Throughput             : {success_count / t_fit_elapsed:.2f} targets/sec")
    log.info("=" * 60)


def main() -> None:
    args = parse_args()
    extract_batch_lightcurves(
        targets_file=args.targets_file,
        pid=args.pid,
        filter_name=args.filter,
        detector=args.detector,
        n_extract_workers=args.n_extract_workers,
        n_cutout_workers=args.n_cutout_workers,
        aperture_radii=args.aperture_radii,
        model_bkg=args.model_bkg,
        fit_offsets=args.fit_offsets,
        catalog=args.catalog,
        epsf=args.epsf,
        image_list=args.image_list,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        figure_dir=args.figure_dir,
        data_dir=args.data_dir,
        max_files=args.max_files,
        max_targets=args.max_targets,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
