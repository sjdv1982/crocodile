import sys, os
from hashlib import sha256
from pathlib import Path
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from library import config


def test_library():
    dinucleotide_libraries, dinucleotide_templates = config(verify_checksums=False)
    libf = dinucleotide_libraries["AA"]
    lib = libf.create("1b7f", nucleotide_mask=[True, False])
    assert lib.coordinates.shape == (9869, 22, 3)
    b = lib.coordinates.tobytes()
    checksum = sha256(b).digest().hex()
    assert (
        checksum == "5e3920f20783389a900472ad53fccefab7fd78de155c86da5fd23839886734c8"
    )


def test_library_reduce() -> None:
    dinucleotide_libraries, _ = config(verify_checksums=False)
    lib = dinucleotide_libraries["AA"].create("1b7f", nucleotide_mask=[True, False])
    reduced = lib.reduce()
    assert reduced.shape == (9869, 7, 3)

    # RA GP1 bead is exactly atom P.
    assert np.array_equal(reduced[:, 0], lib.coordinates[:, 0])

    # RA GS1 bead averages C3', C4', C5'.
    expected = lib.coordinates[:, [20, 5, 4]].mean(axis=1)
    assert np.allclose(reduced[:, 1], expected)


def test_library_reduce_custom_file() -> None:
    dinucleotide_libraries, _ = config(verify_checksums=False)
    lib = dinucleotide_libraries["AA"].create("1b7f", nucleotide_mask=[True, False])

    reduce_data = """\
ALA
1 N N
RA
99 IGNORED O1P
1 B1 P O1P
2 B2 C5' 0.5
"""
    with tempfile.TemporaryDirectory(prefix="reduce_test_") as tmpdir:
        reduce_path = Path(tmpdir) / "reduce.dat"
        reduce_path.write_text(reduce_data)
        reduced = lib.reduce(reduce_path)

    assert reduced.shape == (9869, 2, 3)
    assert np.allclose(reduced[:, 0], lib.coordinates[:, [0, 1]].mean(axis=1))
    assert np.array_equal(reduced[:, 1], lib.coordinates[:, 4])
