# Author: Sjoerd de Vries, CNRS. Copyright 2023-2024.

import sys
import numpy as np

atom13_G = [4.66517696, -0.69231318, 0.03220869]
atom13_G[2] = 0  # impose base planarity
atom13_G = np.array(atom13_G)


def mutate_AtoG(input, output):
    """mutates A to G in output"""
    assert input.shape == (22, 3), input.shape
    assert output.shape == (23, 3), output.shape
    v1 = input[11] - input[8]
    v2 = input[13] - input[8]
    v3 = np.cross(v1, v2)
    v2a = np.cross(v3, v1)
    x = v1 / np.linalg.norm(v1)
    y = v2a / np.linalg.norm(v2a)
    z = v3 / np.linalg.norm(v3)
    mat = np.stack((x, y, z))

    output[:12] = input[:12]
    output[13:] = input[12:]
    atom13 = atom13_G.dot(mat) + input[8]
    output[12] = atom13


def mutate_AtoG_array(input, output):
    """mutates A to G in output, array-wise"""
    assert input.shape[-2:] == (22, 3), input.shape
    assert output.shape[-2:] == (23, 3), output.shape
    v1 = input[:, 11] - input[:, 8]
    v2 = input[:, 13] - input[:, 8]
    v3 = np.cross(v1, v2, axis=1)
    v2a = np.cross(v3, v1, axis=1)
    x = v1 / np.linalg.norm(v1, axis=1)[:, None]
    y = v2a / np.linalg.norm(v2a, axis=1)[:, None]
    z = v3 / np.linalg.norm(v3, axis=1)[:, None]
    mat = np.stack((x, y, z), axis=1)

    output[:, :12] = input[:, :12]
    output[:, 13:] = input[:, 12:]
    atom13 = np.einsum("i,jik->jk", atom13_G, mat) + input[:, 8]
    output[:, 12] = atom13


rna_sizes = {"A": 22, "C": 20, "G": 23, "U": 20}


def mutate(arr: np.ndarray, input_sequence: str, output_sequence: str) -> np.ndarray:
    """Mutate heavy-atom fragment or fragment library, from A/C to A/C/G/U.

    arr: Input fragment coordinates in .npy format.

    The dtype must be float, and the shape must be (X, 3) or (Y, X, 3).
    X is the number of heavy atoms in the fragment, which must match with the sequence:
    22 atoms for each A, 20 atoms for each C.
    This is what is generated by ATTRACT aareduce, using the OPLSX force field for RNA.

    input_sequence: Sequence of the input fragment(s). Must consist of A and/or C",
    output_sequence: Sequence of the output fragment(s). Must consist of A, C, G and/or U".

    Returns: Output fragment coordinates in .npy format.
    The shape will be the same as the input data, but with one extra atom for each G."""

    if len(input_sequence) != len(output_sequence):
        err("Input and output sequence are not of the same length")

    totsize = 0
    for char in input_sequence:
        if char not in ("A", "C"):
            err("Input sequence must contain only A and C")
        totsize += rna_sizes[char]

    output_totsize = 0
    for char in output_sequence:
        if char not in ("A", "C", "G", "U"):
            err("Output sequence must contain only A, C, G, U")
        output_totsize += rna_sizes[char]

    if arr.ndim not in (2, 3) or arr.shape[-1] != 3:
        err(f"Input data has the wrong shape: {arr.shape}")

    if arr.shape[-2] != totsize:
        err(
            f"Input data has the wrong number of atoms for the sequence: {arr.shape[-2]} instead of {totsize}"
        )
    if arr.ndim == 2:
        inp = arr[None, :, :]
    else:
        inp = arr

    outp = np.empty((inp.shape[0], output_totsize, 3), inp.dtype)

    offset = 0
    output_offset = 0

    for nuc1, nuc2 in zip(input_sequence, output_sequence):
        size = rna_sizes[nuc1]
        output_size = rna_sizes[nuc2]
        curr_inp = inp[:, offset : offset + size]
        curr_outp = outp[:, output_offset : output_offset + output_size]
        if nuc1 != nuc2:
            if (nuc1, nuc2) not in (("A", "G"), ("C", "U")):
                err(f"Cannot mutate {nuc1} to {nuc2}")
            if (nuc1, nuc2) == ("A", "G"):
                mutate_AtoG_array(curr_inp, curr_outp)
            elif (nuc1, nuc2) == ("C", "U"):
                curr_outp[:] = curr_inp  # equivalent atoms in the same order
        else:
            curr_outp[:] = curr_inp  # no change
        offset += size
        output_offset += output_size

    if arr.ndim == 2:
        outp = outp[0]
    return outp


def mutate_to_AC(
    arr: np.ndarray,
    input_sequence: str,
) -> tuple[np.ndarray, str]:
    """Mutate heavy-atom fragment or fragment library, from A/C/G/U to A/C.

    arr: Input fragment coordinates in .npy format.

    The dtype must be float, and the shape must be (X, 3) or (Y, X, 3).
    X is the number of heavy atoms in the fragment, which must match with the sequence:
    22 atoms for each A, 20 atoms for each C, 23 atoms for each G, 20 atoms for each U.
    This is what is generated by ATTRACT aareduce, using the OPLSX force field for RNA.

    input_sequence: Sequence of the input fragment(s). Must consist of A, C, G and/or U".

    Returns:
        - Output fragment coordinates in .npy format.
            The shape will be the same as the input data, but with one less atom for each G.
    """

    totsize = 0
    output_totsize = 0
    output_sequence = ""
    for char in input_sequence:
        if char not in ("A", "C", "G", "U"):
            err("Input sequence must contain only A, C, G, U")
        rna_size = rna_sizes[char]
        totsize += rna_size

        output_totsize += rna_size
        if char == "G":
            output_totsize -= 1
            char = "A"
        elif char == "U":
            char = "C"
        output_sequence = output_sequence + char

    if arr.ndim not in (2, 3) or arr.shape[-1] != 3:
        err(f"Input data has the wrong shape: {arr.shape}")

    if arr.shape[-2] != totsize:
        err(
            f"Input data has the wrong number of atoms for the sequence: {arr.shape[-2]} instead of {totsize}"
        )
    if arr.ndim == 2:
        inp = arr[None, :, :]
    else:
        inp = arr
    outp = np.empty((inp.shape[0], output_totsize, 3), inp.dtype)

    offset = 0
    output_offset = 0

    for nuc1, nuc2 in zip(input_sequence, output_sequence):
        size = rna_sizes[nuc1]
        output_size = rna_sizes[nuc2]
        for frag in range(len(inp)):
            curr_inp = inp[frag, offset : offset + size]
            curr_outp = outp[frag, output_offset : output_offset + output_size]
            if nuc1 != nuc2:
                if (nuc1, nuc2) not in (("G", "A"), ("U", "C")):
                    err(f"Cannot mutate {nuc1} to {nuc2}")
                if (nuc1, nuc2) == ("G", "A"):
                    curr_outp[:12] = curr_inp[:12]
                    curr_outp[12:] = curr_inp[13:]
                elif (nuc1, nuc2) == ("U", "C"):
                    curr_outp[:] = curr_inp  # equivalent atoms in the same order
            else:
                curr_outp[:] = curr_inp  # no change
        offset += size
        output_offset += output_size

    if arr.ndim == 2:
        outp = outp[0]
    return outp, output_sequence


if __name__ == "__main__":

    def err(*args, **kwargs):
        print(*args, **kwargs, file=sys.stderr)
        exit(1)

    import argparse

    parser = argparse.ArgumentParser(
        description="Mutate heavy-atom fragment or fragment library, from A/C to A/C/G/U"
    )
    parser.add_argument(
        "input_sequence",
        help="Sequence of the input fragment(s). Must consist of A and/or C",
    )
    parser.add_argument(
        "output_sequence",
        help="Sequence of the output fragment(s). Must consist of A, C, G and/or U",
    )
    parser.add_argument(
        "input_data",
        type=argparse.FileType("rb"),
        help="""Input fragment coordinates in .npy format.
The dtype must be float, and the shape must be (X, 3) or (Y, X, 3).
X is the number of heavy atoms in the fragment, which must match with the sequence:
22 atoms for each A, 20 atoms for each C.""",
    )
    parser.add_argument(
        "output_data",
        type=argparse.FileType("wb"),
        help="""Output fragment coordinates in .npy format.
The shape will be the same as the input data, but with one extra atom for each G.""",
    )

    args = parser.parse_args()
    input_sequence = args.input_sequence
    output_sequence = args.output_sequence
    arr = np.load(args.input_data)
    outp = mutate(arr, input_sequence, output_sequence)

    np.save(args.output_data, outp)
