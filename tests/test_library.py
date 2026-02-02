import sys, os
from hashlib import sha256

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
