import sys
import numpy as np
from crocodile.trinuc import trinuc_dtype, trinuc_roco_dtype


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


def best_fit(trinuc_array: np.ndarray, nth_best=1):
    best_ind = np.diff(trinuc_array["first_resid"], prepend=-1).astype(bool)
    best_ind = np.where(best_ind)[0]
    sizes = np.diff(best_ind, append=len(trinuc_array))
    indices = best_ind + nth_best - 1
    mask = sizes < nth_best
    indices[mask] = 0
    result = trinuc_array[indices]
    result["first_resid"] = trinuc_array["first_resid"][best_ind]
    result["rmsd"][mask] = np.nan
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trinuc",
        help="A trinucleotide fitted array of conformers, translated rotaconformers or rotaconformers on a grid, in numpy format",
    )
    parser.add_argument(
        "nth_best",
        default=1,
        nargs="?",
        type=int,
        help="Print the nth best fitting conformer",
    )

    args = parser.parse_args()

    trinuc_array_file = args.trinuc
    trinuc_array = np.load(trinuc_array_file)
    if trinuc_array.dtype not in (trinuc_dtype, trinuc_roco_dtype):
        err(f"'{trinuc_array_file}' does not contain a trinucleotide fitted array")

    result = best_fit(trinuc_array, args.nth_best)

    for trinuc in result:
        print(trinuc["first_resid"], trinuc["rmsd"])


if __name__ == "__main__":
    main()
