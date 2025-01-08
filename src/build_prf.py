import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import simple_norm
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from roman_lcs import RomanMachine

PATH = "/Users/jimartin/Work/ROMAN/TRExS/simulations/dryrun_01"

ZP = {'F087': 26.29818407774948,
'F146': 27.577660642304814,
'F213': 25.85726796291789}


def plot_image_and_mask(mac):
    ROW, COL = mac.pixel_coordinates(mac.ref_frame)

    fig, ax = plt.subplots(
        1, 3, figsize=(15, 5), sharex=True, sharey=True, constrained_layout=True
    )

    ax[0].set_title("Flux")
    ax[0].scatter(
        mac.column,
        mac.row,
        c=mac.flux[mac.ref_frame], 
        vmin=10, 
        vmax=500,
        s=1,
        marker="s",
    )
    ax[0].scatter(COL, ROW, c="tab:red", marker=".", s=2)

    ax[1].set_title("Source Mask")
    ax[1].scatter(
        mac.column,
        mac.row,
        c=np.array(mac.source_mask.sum(axis=0)[0]), 
        vmin=0, 
        vmax=4,
        alpha=1,
        cmap="YlGnBu",
        s=1,
        marker="s",
    )
    ax[1].scatter(COL, ROW, c="tab:red", marker=".", s=2)

    ax[2].set_title("Uncontaminated Pixel Mask")
    ax[2].scatter(
        mac.column,
        mac.row,
        c=np.array(mac.uncontaminated_source_mask.sum(axis=0)[0]), 
        vmin=0, 
        vmax=1,
        alpha=1,
        cmap="Blues",
        s=1,
        marker="s",
    )
    ax[2].scatter(COL, ROW, c="tab:red", marker=".", s=2)

    ax[0].set_aspect("equal", adjustable="box")
    ax[1].set_aspect("equal", adjustable="box")
    ax[2].set_aspect("equal", adjustable="box")

    ax[0].set_ylabel("Pixel Row")
    ax[0].set_xlabel("Pixel Row")
    ax[1].set_xlabel("Pixel Row")
    ax[2].set_xlabel("Pixel Row")

    return fig


def build_prf(
    FIELD: int = 3,
    FILTER: str = "F146",
    cutout_size: int = 256,
    mag_limit: float=21.0,
    plot: bool = False,
    cutout_origin: tuple = (0, 0),
):


    ff = sorted(
        glob(
            f"{PATH}/imgs/{FILTER}/rimtimsim_WFI_lvl02_{FILTER}_SCA02_field{FIELD:02}_rampfitted_exposureno_*_sim.fits"
        )
    )
    print(f"Total files for Field {FIELD} Filter {FILTER} in folder: {len(ff)}.")

    catalog = pd.read_csv(
        f"{PATH}/TRExS_dryrun_01_MASTER_input_catalog_v1.1.txt", index_col=0
    )
    catalog["flux"] = 10 ** ((ZP[FILTER] - catalog[FILTER]) / 2.5)
    catalog["flux_err"] = np.sqrt(catalog["flux"])

    sources = catalog.query(
        f"F146 <= {mag_limit} and "
        f"MEAN_XCOL >= {cutout_origin[0]} and MEAN_XCOL <= {cutout_origin[0] + cutout_size} and "
        f"MEAN_YCOL >= {cutout_origin[1]} and MEAN_YCOL <= {cutout_origin[1] + cutout_size}"
    ).reset_index(drop=False)
    sources = sources.rename(
        columns={
            "RA_DEG": "ra",
            "DEC_DEG": "dec",
            "MEAN_XCOL": "column",
            "MEAN_YCOL": "row",
        }
    )
    print(f"Total sources Mag_{FILTER} <= {mag_limit} is {len(sources)}.")

    mac = RomanMachine.from_file(
        ff[:1], 
        sources=sources,
        sparse_dist_lim=2, 
        sources_flux_column="flux",
        cutout_size=cutout_size,
        cutout_origin=cutout_origin,
    )
    print(mac)
    mac.contaminant_flux_limit = 10 ** ((ZP[FILTER] - 21.0)/2.5)

    mac.rmin = 0.02
    mac.rmax = 0.8
    mac.cut_r = 0.2
    mac.n_r_knots = 9
    mac.n_phi_knots = 15

    OUTPATH = f"/Users/jimartin/Work/ROMAN/TRExS/Roman-lcs/data"

    if plot:
        pdf_name = f"build_PRF_plots_{mac.meta['FIELD']}_{mac.meta['FILTER']}_xo{cutout_origin[0]}-yo{cutout_origin[1]}_s{cutout_size}_m{int(mag_limit)}.pdf"
        with PdfPages(f"{OUTPATH}/figures/{pdf_name}") as pdf:
            FigureCanvasPdf(mac.plot_image(sources=True, frame_index=0)).print_figure(pdf)
            FigureCanvasPdf(mac._get_source_mask(source_flux_limit=10, plot=True, reference_frame=0, iterations=1)).print_figure(pdf)
            FigureCanvasPdf(plot_image_and_mask(mac)).print_figure(pdf)
            FigureCanvasPdf(
                mac.build_shape_model(
                    plot=True,
                    flux_cut_off=0.2,
                    frame_index=0,
                    bin_data=False,
                )
            ).print_figure(pdf)
        print(f"Saving figures file to \n"f"{OUTPATH}/figures/{pdf_name}")
    else:
        mac._get_source_mask(
            source_flux_limit=10, plot=True, reference_frame=0, iterations=1
        )

    fname = (
        f"{OUTPATH}/prf_models/roman_WFI_{mac.meta['READMODE']}_{mac.meta['FILTER']}"
        f"_{mac.meta['FIELD']}_{mac.meta['DETECTOR']}_shape_model_cad{0}"
        f"_xo{cutout_origin[0]}-yo{cutout_origin[1]}_s{cutout_size}.fits"
    )
    mac.save_shape_model(output=fname)
    print(f"Saving PRF file to \n{fname}")

    print("Done!")

if __name__ == "__main__":
    # program flags
    parser = argparse.ArgumentParser(
        description="Build PRF model for a given filter band."
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
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot target light curve.",
    )

    args = parser.parse_args()

    print(args)

    build_prf(
        FILTER=args.filter,
        cutout_size=args.cutout_size,
        plot=args.plot,
        cutout_origin=(args.row0, args.col0),
    )
