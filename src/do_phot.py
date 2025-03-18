import argparse
import os
import sqlite3
import sys
from glob import glob
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from roman_lcs import RomanMachine
from roman_lcs.utils import to_fits

# from .build_prf import plot_image_and_mask

PATH = "/Users/jimartin/Work/ROMAN/TRExS/simulations/dryrun_01"
# PATH = "/Volumes/seagate_exhd/trexs/DryRun_01"

ZP = {'F087': 26.29818407774948,
'F146': 27.577660642304814,
'F213': 25.85726796291789}


def do_photometry(
    FIELD: int = 3,
    FILTER: str = "F146",
    cutout_size: int = 32,
    mag_limit: float = 23.0,
    plot: bool = False,
    cutout_origin: tuple = (0, 0),
    target: Optional[Union[int, str]] = None,
    nthreads: int = 3,
    use_gpu: bool = False,
):

    # get list of FITS file paths to load into Machine
    ff = sorted(
        glob(
            f"{PATH}/simulated_image_data/rimtimsim_WFI_lvl02_{FILTER}_SCA02_field{FIELD:02}_rampfitted_exposureno_*_sim.fits"
        )
    )
    if True:
        parPATH = "/Volumes/TRExS/dryrun01/"
        ffp = sorted(
            glob(
                f"{parPATH}/simulated_image_data/rimtimsim_WFI_lvl02_{FILTER}_SCA02_field{FIELD:02}_rampfitted_exposureno_*_sim.fits"
            )
        )
        ff.extend(ffp)
        ff = np.unique(ff).tolist()
    print(f"Total files for Field {FIELD} Filter {FILTER} in folder: {len(ff)}.")

    with sqlite3.connect(f"{PATH}/metadata/TRExS_dryrun_01_MASTER_input_catalog_v1.1.db") as conn:
        query = (
            f"F146 <= {mag_limit} and "
            f"MEAN_XCOL >= {cutout_origin[0]} and MEAN_XCOL <= {cutout_origin[0] + cutout_size} and "
            f"MEAN_YCOL >= {cutout_origin[1]} and MEAN_YCOL <= {cutout_origin[1] + cutout_size}"
        )
        sources = pd.read_sql_query(
            f"SELECT * FROM Master_input_catalog WHERE {query}", conn
        ).reset_index(drop=False)

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
    print(f"Total sources Mag_{FILTER} <= {mag_limit} is {len(sources)}.")

    # we check for transiting hosts in the source list and skip if none
    # need to comment this when extracting all sources...
    print(sources.transitHost.value_counts())
    if len(sources.query("transitHost == 1")) < 1:
        print("No transit hosts in this cutout, exiting...")
        sys.exit()

    # we check for missing light curves in the archive as listed in a file with the ids
    missing_ids = np.loadtxt(
        f"/Users/jimartin/Work/ROMAN/TRExS/Roman-lcs/src/{FILTER}_missing_ids.txt"
    )
    total_missing = np.isin(sources["sicbro_id"].values, missing_ids).sum()
    print(f"Missing {total_missing} targets in archive")
    if total_missing == 0:
        print("No missing transit targets in this cutout, exiting...")
        sys.exit()

    # start Machine object
    mac = RomanMachine.from_file(
        ff, 
        sources=sources,
        sparse_dist_lim=2, 
        sources_flux_column="flux",
        cutout_size=cutout_size,
        cutout_origin=cutout_origin,
    )
    print(mac)
    # set contaminant limit at 21 to aliviate pixel mask
    mac.contaminant_flux_limit = 10 ** ((ZP[FILTER] - 21.0)/2.5)

    # load the PRF model from file
    # for dryrun 01 the PSF is constant across the FoV, so we use the PSF we fitted
    # in the center of the CCD
    prf_origin = (1792, 1792)
    DATPATH = "/Users/jimartin/Work/ROMAN/TRExS/Roman-lcs/data"
    prf_fname = (
        f"{DATPATH}/prf_models/"
        f"roman_WFI_{mac.meta['READMODE']}_{mac.meta['FILTER']}"
        f"_{mac.meta['FIELD']}_{mac.meta['DETECTOR']}_shape_model_cad{0}"
        f"_row{prf_origin[0]}-col{prf_origin[1]}_size256.fits"
    )

    if plot:
        pdf_name = f"build_PRF_plots_{mac.meta['FIELD']}_{mac.meta['FILTER']}_xo{cutout_origin[0]}-yo{cutout_origin[1]}_s{cutout_size}_m{int(mag_limit)}.pdf"
        with PdfPages(f"{DATPATH}/figures/eval_phot/{pdf_name}") as pdf:
            FigureCanvasPdf(mac.plot_image(sources=True, frame_index=0)).print_figure(pdf)
            FigureCanvasPdf(
                mac.load_shape_model(
                    prf_fname, flux_cut_off=0.1, plot=True, source_flux_limit=10
                )
            ).print_figure(pdf)
            # FigureCanvasPdf(plot_image_and_mask(mac)).print_figure(pdf)
        print(f"Saving figures file to \n" f"{DATPATH}/figures/eval_phot/{pdf_name}")
    else:
        mac.load_shape_model(
            prf_fname, flux_cut_off=0.1, plot=False, source_flux_limit=10
        )

    mac.quiet = True

    # we limit the number of threads used by BLAS library when doing matrix solving
    # so we can run parallel jobs without lowering performance too much
    with threadpool_limits(limits=nthreads, user_api="blas"):
        # if asked, we fit for an specific target by narrowing the priors for background sources
        if isinstance(target, int):
            # we check target is in catalog
            if target in mac.sources["sicbro_id"].values:
                print(f"Fitting photometry targeted to {target}")
                prior_mu = mac.source_flux_estimates
                prior_sigma = (
                    np.ones(mac.nsources)
                    * 1
                    * np.abs(mac.source_flux_estimates) ** 0.5
                )
                prior_sigma[mac.sources["sicbro_id"].values == target] *= 5
                mac.fit_model(prior_mu=prior_mu, prior_sigma=prior_sigma)
            # if not in catalog fit all sources
            else:
                mac.fit_model()
        # or we fit for specific transit hosts in the catalog...
        elif target == "transits":
            transit_idx = mac.sources.query("transitHost == 1").index.values
            prior_mu = mac.source_flux_estimates
            prior_sigma = (
                np.ones(mac.nsources) * 1 * np.abs(mac.source_flux_estimates) ** 0.5
            )
            prior_sigma[transit_idx] *= 5
            mac.fit_model(prior_mu=prior_mu, prior_sigma=prior_sigma, use_gpu=use_gpu)
        # or we fit for all sources with no constrains
        else:
            mac.fit_model(use_gpu=use_gpu)

    # get PSF metrics
    mac.get_psf_metrics(npoints_per_pixel=0)
    psffrac = mac.source_psf_fraction / np.percentile(mac.source_psf_fraction, 75)
    psffrac[psffrac>=1] = 1

    # save LCs to fits files
    for k in tqdm(range(mac.nsources), total=mac.nsources, desc="Saving FITS"):
        metadata = mac.meta.copy()
        metadata["INSTRUME"] = "WFI"
        metadata["SICBROID"] = mac.sources["sicbro_id"].iloc[k]
        metadata["RADESYS"] = "ICRS"
        metadata["RA_OBJ"] = mac.sources["ra"].iloc[k]
        metadata["DEC_OBJ"] = mac.sources["dec"].iloc[k]
        metadata[f"{FILTER}MAG"] = mac.sources[FILTER].iloc[k]
        metadata[f"{FILTER}FLX"] = np.round(mac.sources["flux"].iloc[k], decimals=3)
        metadata["PSFFRAC"] = np.round(mac.source_psf_fraction[k], decimals=3)

        # corr_flux = (mac.ws[:, k] / np.nanmedian(mac.ws[:, k])) * np.random.normal(mac.sources["flux"].iloc[k], mac.sources["flux_err"].iloc[k])

        # replace nans and negatives with 0

        quality = np.zeros(mac.nt, dtype=int)
        cadenceno = np.arange(mac.nt, dtype=int)
        flux = mac.ws[:, k]
        flux_err = mac.werrs[:, k]

        quality[~np.isfinite(mac.ws[:, k])] += 1

        neg_mask = np.isfinite(flux) & (flux < 0)
        quality[neg_mask] += 2
        
        pos_mask = np.isfinite(flux) & (flux >= 0)
        flux[~pos_mask] = np.nan
        flux_err[~pos_mask] = np.nan

        data = {
            "time": mac.time,
            "flux": flux,
            "flux_err": flux_err,
            "cadenceno" : cadenceno,
            "quality": quality,
        }
        fid = f"{metadata['SICBROID']:08}"
        lc_dir = f"{DATPATH}/lcs_v2/{fid[:5]}"
        if not os.path.isdir(lc_dir):
            os.makedirs(lc_dir)

        fname = f"{lc_dir}/roman_wfi_{fid}_{metadata['FILTER']}_dryrun01_lc.fits"
        if mac.sources["transitHost"].iloc[k] == 1:
            fname = fname.replace("dryrun01", "f_dryrun01")
            print(fname)
        to_fits(data, path=fname, overwrite=True, **metadata)

    print("Done!")

if __name__ == "__main__":
    # program flags
    parser = argparse.ArgumentParser(
        description="Do photometry using saved PRF model."
    )
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
        "--row-org",
        dest="row0",
        type=int,
        default=0,
        help="Lower-left x-coordinate origin of cutout.",
    )
    parser.add_argument(
        "--col-org",
        dest="col0",
        type=int,
        default=0,
        help="Lower-left y-coordinate origin of cutout.",
    )
    parser.add_argument(
        "--target",
        dest="target",
        # type=int,
        default=None,
        help="Object id when targetting the photometry fitting.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot target light curve.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=False,
        help="Enable GPU.",
    )

    args = parser.parse_args()

    print(args)

    do_photometry(
        FILTER=args.filter,
        cutout_size=args.cutout_size,
        plot=args.plot,
        cutout_origin=(args.row0, args.col0),
        target=args.target,
        use_gpu=args.use_gpu,
    )
