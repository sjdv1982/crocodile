import sys
import numpy as np
from crocodile.trinuc import trinuc_dtype, trinuc_roco_dtype


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


def discretize_grid(trinuc_array: np.ndarray, gridspacing: float):
    offsets = trinuc_array["offset"]
    rmsds = trinuc_array["rmsd"]
    offsets_disc = np.round(offsets / gridspacing) * gridspacing
    disc_error = ((offsets - offsets_disc) ** 2).sum(axis=1)
    msds = rmsds**2
    rmsds_disc = np.sqrt(msds + disc_error)

    inds = np.where(
        np.diff(trinuc_array["first_resid"], prepend=-1, append=-1).astype(bool)
    )[0]
    result = np.empty(len(inds) - 1, dtype=trinuc_roco_dtype)
    for n in range(len(inds) - 1):
        start, end = inds[n : n + 2]
        curr_rmsds_disc = rmsds_disc[start:end]
        best_ind = curr_rmsds_disc.argmin()
        best = trinuc_array[start:end][best_ind].copy()
        best["rmsd"] = curr_rmsds_disc[best_ind]
        best["offset"] = offsets_disc[start:end][best_ind]
        result[n] = best

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trinuc",
        help="A trinucleotide fitted array of conformers, translated rotaconformers or rotaconformers on a grid, in numpy format",
    )
    parser.add_argument(
        "gridspacing",
        default=np.sqrt(3) / 3,
        nargs="?",
        type=float,
        help="Grid spacing in Angstroms. Default: 1/3 * sqrt(3)",
    )
    parser.add_argument(
        "--output",
        help="Output file for resulting grid-discretized array. Only the lowest RMSD is kept",
        required=True,
    )

    args = parser.parse_args()

    trinuc_array_file = args.trinuc
    trinuc_array = np.load(trinuc_array_file)
    if trinuc_array.dtype != trinuc_roco_dtype:
        if trinuc_array.dtype == trinuc_dtype:
            err(
                f"'{trinuc_array_file}' does not contain rotaconformers, use fit_roco first"
            )
        else:
            err(
                f"'{trinuc_array_file}' does not contain a trinucleotide rotaconformer fitted array"
            )

    result = discretize_grid(trinuc_array, args.gridspacing)

    np.save(args.output, result)


if __name__ == "__main__":
    main()
