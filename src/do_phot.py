import argparse
import logging
import os
import sqlite3
import sys
import warnings
from datetime import datetime
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd
from roman_cuts import RomanCuts
from threadpoolctl import threadpool_limits

# from mem_profile import mem_profile
from roman_lcs import RomanMachine
from roman_lcs.utils import clean_blends_in_catalog, to_fits

warnings.filterwarnings("ignore")

# set logging
log = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout,
)

PATH = "/Users/jimartin/Work/ROMAN/TRExS/simulations/dryrun_01"
# PATH = "/Volumes/JorgeMarpa-2T/trexs/dryrun_01/"

ZP = {
    "F087": 26.29818407774948,
    "F146": 27.577660642304814,
    "F213": 25.85726796291789,
}


# @profile
def do_target_photometry(
    target: int = 0,
    FIELD: int = 3,
    FILTER: str = "F146",
    SCA: int = 2,
    cutout_size: int = 32,
    mag_limit: float = 23.0,
    fit_blends: bool = False,
    nthreads: Optional[int] = None,
    blend_limit: Optional[float] = None,
):
    # get list of FITS file paths to load into Machine
    ff = sorted(
        glob(
            f"{PATH}/simulated_image_data/rimtimsim_WFI_lvl02_{FILTER}_SCA{SCA:02}_field{FIELD:02}_rampfitted_exposureno_*_sim.fits"
        )
    )
    # parPATH = "/Volumes/JorgeMarpa-2T/trexs/dryrun_01/"
    # ff = sorted(
    #     glob(
    #         f"{parPATH}/simulated_image_data/rimtimsim_WFI_lvl02_{FILTER}_SCA{SCA:02}_field{FIELD:02}_rampfitted_exposureno_*_sim.fits"
    #     )
    # )
    # ff.extend(ff)
    ff = np.unique(ff).tolist()
    if len(ff) < 0:
        log.error(f"No files found for Field {FIELD} Filter {FILTER} in folder {PATH}.")
        return 1
    log.info(f"Total files for Field {FIELD} Filter {FILTER} in folder: {len(ff)}.")

    try:
        with sqlite3.connect(
            f"{PATH}/metadata/TRExS_dryrun_01_MASTER_input_catalog_v1.1.db"
        ) as conn:
            query = f"sicbro_id == {target}"
            sources = pd.read_sql_query(
                f"SELECT * FROM Master_input_catalog WHERE {query}", conn
            )
    except Exception as e:
        log.error(f"Could not load source catalog for target {target}.")
        log.error(f"Error: {e}")
        return 2
    cutout_center = tuple(sources.loc[0, ["RA_DEG", "DEC_DEG"]].values.tolist())

    log.info(
        f"Targeting photometry to ID {target} RA {cutout_center[0]} Dec {cutout_center[1]}."
    )
    # get cutout origin
    rcube = RomanCuts(field=FIELD, sca=SCA, filter=FILTER, file_list=ff[:2])
    rcube.get_all_wcs()
    pix_center = rcube.wcss[0].all_world2pix(cutout_center[0], cutout_center[1], 0)
    cutout_origin = np.array(pix_center) - cutout_size / 2
    cutout_origin = np.round(cutout_origin).astype(int)

    # load surce catalogs in the cutout
    buffer = 4  # pixel buffer for catalog query
    with sqlite3.connect(
        f"{PATH}/metadata/TRExS_dryrun_01_MASTER_input_catalog_v1.1.db"
    ) as conn:
        query = (
            f"F146 <= {mag_limit} and "
            f"MEAN_XCOL >= {cutout_origin[0] - buffer} and MEAN_XCOL <= {cutout_origin[0] + cutout_size + buffer} and "
            f"MEAN_YCOL >= {cutout_origin[1] - buffer} and MEAN_YCOL <= {cutout_origin[1] + cutout_size + buffer}"
        )
        sources = pd.read_sql_query(
            f"SELECT * FROM Master_input_catalog WHERE {query}", conn
        ).reset_index(drop=True)
    if len(sources) == 0:
        log.error(f"No sources found in cutout for target {target}.")
        return 3

    # rename columns so Machine can read the right columns
    sources = sources.rename(
        columns={
            "RA_DEG": "ra",
            "DEC_DEG": "dec",
            "MEAN_XCOL": "column",
            "MEAN_YCOL": "row",
            f"{FILTER}_flux": "flux",
            f"{FILTER}_flux_err": "flux_err",
        }
    )
    log.info(f"Total sources Mag_{FILTER} <= {mag_limit} is {len(sources)}.")

    # remove highly contaminated sources, faint one from < blend_limit pairs
    if isinstance(blend_limit, float):
        sources = clean_blends_in_catalog(
            sources, blend_limit=blend_limit, filter=FILTER
        )

    log.info(f"Total sources Mag_{FILTER} <= {mag_limit} is {len(sources)}.")

    # we check for transiting hosts in the source list and skip if none
    # need to comment this when extracting all sources...
    # if len(sources.query("transitHost == 1")) < 1:
    #     log.info("No transit hosts in this cutout, exiting...")
    #     sys.exit()

    # find the target indices in the sources catalog
    target_idx = sources.query(f"sicbro_id == {target}").index
    if fit_blends:
        r = np.hypot(
            sources.ra.values - sources.ra.iloc[target_idx].item(),
            sources.dec.values - sources.dec.iloc[target_idx].item(),
        )
        sort_idx = np.argsort(r)
        targets_idx = sort_idx[r[sort_idx] * 3600 < 0.2].ravel()
    else:
        targets_idx = target_idx
    log.info(f"Target indices: {targets_idx.tolist()}")

    # start Machine object
    log.info("Create RomanMachine object")
    try:
        mac = RomanMachine.from_file(
            ff,
            sources=sources,
            sparse_dist_lim=2,
            sources_flux_column="flux",
            cutout_size=cutout_size,
            cutout_center=cutout_center,
        )
    except Exception as e:
        log.error("Could not start RomanMachine.")
        log.error(f"Error: {e}")
        return 6
    log.info(mac)
    # set contaminant limit at 21 to aliviate pixel mask
    mac.contaminant_flux_limit = 10 ** ((ZP[FILTER] - 21.0) / 2.5)

    # load the PRF model from file
    # for dryrun 01 the PSF is constant across the FoV, so we use the PSF we fitted
    # in the center of the CCD
    DATPATH = "/Users/jimartin/Work/ROMAN/TRExS/Roman-lcs/data"
    prf_fname = (
        f"{DATPATH}/prf_models/"
        f"roman_WFI_{mac.meta['READMODE']}_{mac.meta['FILTER']}"
        f"_{mac.meta['FIELD']}_{mac.meta['DETECTOR']}_shape_model_cad{0}"
        f"_center_v2.fits"
    )
    try:
        mac.load_shape_model(
            prf_fname, flux_cut_off=0.01, plot=False, source_flux_limit=5
        )
    except Exception as e:
        log.error(f"Could not load PRF model from {prf_fname}.")
        log.error(f"Error: {e}")
        return 4

    mac.quiet = False

    # we limit the number of threads used by BLAS library when doing matrix solving
    # so we can run parallel jobs without lowering performance too much
    try:
        with threadpool_limits(limits=nthreads, user_api="blas"):
            mac.fit_prf_photometry(targets=targets_idx.tolist(), model_bkg=True)
    except Exception as e:
        log.error("Error during PRF fitting")
        log.error(f"Error: {e}")
        return 5

    # save LCs to fits files
    for i, k in enumerate(targets_idx):
        metadata = mac.meta.copy()
        metadata["FILEVER"] = ("2.0", "File version")
        metadata["INSTRUME"] = "WFI"
        metadata["SICBROID"] = mac.sources["sicbro_id"].iloc[k]
        metadata["RADESYS"] = "ICRS"
        metadata["RA_OBJ"] = mac.sources["ra"].iloc[k]
        metadata["DEC_OBJ"] = mac.sources["dec"].iloc[k]
        metadata[f"{FILTER}MAG"] = (
            mac.sources[FILTER].iloc[k],
            "Input catalog magnitude",
        )
        metadata[f"{FILTER}FLX"] = (
            np.round(mac.sources["flux"].iloc[k], decimals=3),
            "Input catalog flux",
        )
        metadata["FITMODE"] = ("target", "fitting mode")

        # replace nans and negatives with 0
        quality = np.zeros(mac.nt, dtype=int)
        expnumber = mac.cadenceno
        flux = mac.targets_prf_flux[:, i]
        flux_err = mac.targets_prf_flux_err[:, i]

        # quality flag: nans from PRF fitting
        quality[~np.isfinite(flux)] += 1
        # quality flag: non invertible matrix
        quality[flux == -1e6] += 2
        # quality flag: negative fluxes
        neg_mask = np.isfinite(flux) & (flux < 0) & (flux != -1e6)
        quality[neg_mask] += 4
        # set negative fluxes to nan
        pos_mask = np.isfinite(flux) & (flux >= 0)
        flux[~pos_mask] = np.nan
        flux_err[~pos_mask] = np.nan

        data = {
            "time": mac.time,
            "flux": flux,
            "flux_err": flux_err,
            "cadenceno": expnumber,
            "quality": quality,
        }
        fid = f"{metadata['SICBROID']:08}"
        lc_dir = f"{DATPATH}/lcs_v2/{fid[:5]}"
        if not os.path.isdir(lc_dir):
            os.makedirs(lc_dir)

        fname = f"{lc_dir}/roman_wfi_{fid}_{metadata['FILTER']}_dryrun01_lc.fits"
        to_fits(data, path=fname, overwrite=True, **metadata)
    return 0


if __name__ == "__main__":
    # program flags
    parser = argparse.ArgumentParser(description="Do photometry using saved PRF model.")
    parser.add_argument(
        "--filter",
        dest="filter",
        type=str,
        default="F146",
        help="Filter.",
    )
    parser.add_argument(
        "--cutout-size",
        dest="cutout_size",
        type=int,
        default=256,
        help="Subimage size",
    )
    parser.add_argument(
        "--target",
        dest="target",
        # type=int,
        default=None,
        help="Object id when targeting the photometry fitting.",
    )
    parser.add_argument(
        "--blend-limit",
        dest="blend_limit",
        type=float,
        default=None,
        help="Lower distance limit to allow blends, fainter and closer sources will be removed from input catalog.",
    )
    parser.add_argument(
        "--fit-blends",
        dest="fit_blends",
        type=bool,
        default=False,
        help="Fit blended objects to 'target' within 0.2 arcseconds.",
    )
    parser.add_argument("--log", dest="log", default=0, help="Logging level")
    args = parser.parse_args()

    # set logging level
    try:
        args.log = int(args.log)
    except ValueError:
        args.log = str(args.log.upper())
    # log.addHandler(
    #     logging.FileHandler(
    #         f"{PACKAGEDIR}/logs/romanmachine_"
    #         f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.info"
    # )
    # )
    log.setLevel(args.log)

    log.info(args)

    exit = do_target_photometry(
        target=args.target,
        FIELD=3,
        SCA=2,
        FILTER=args.filter,
        cutout_size=args.cutout_size,
        mag_limit=24.0,
        fit_blends=args.fit_blends,
        blend_limit=args.blend_limit,
        nthreads=4,  # set to None to use all available threads
    )

    with open(f"../logs/photometry_{args.filter}_cutout.log", "a") as f:
        f.write(f"{datetime.now()} - Target {args.target} exit code: {exit}\n")
    log.info(f"Photometry done for target {args.target} with exit code {exit}.")
