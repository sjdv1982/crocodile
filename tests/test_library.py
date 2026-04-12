import sys, os
from hashlib import sha256
from pathlib import Path
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from library import config, load_crmsds, _sha256sum, _verify_checksums


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


def test_load_crmsds_with_pdb_exclusion(tmp_path, monkeypatch) -> None:
    library_dir = tmp_path / "library"
    crmsd_dir = tmp_path / "crmsds"
    library_dir.mkdir()
    crmsd_dir.mkdir()

    np.save(library_dir / "dinuc-AA-0.5.npy", np.zeros((2, 1, 3)))
    np.save(library_dir / "dinuc-AC-0.5.npy", np.zeros((3, 1, 3)))
    (library_dir / "dinuc-AA-0.5-extension.origin.txt").write_text("keep\n1b7f\n")
    (library_dir / "dinuc-AC-0.5-extension.origin.txt").write_text("1b7f\nkeep\n")

    crmsds = np.arange(20, dtype=np.float64).reshape(4, 5)
    np.save(crmsd_dir / "crmsd_matrix_AAC.npy", crmsds)

    fraglib_yaml = tmp_path / "fraglib.yaml"
    fraglib_yaml.write_text(
        f"""\
CONFORMER_DIR: {library_dir}
ROTAMER_DIR: {library_dir}
CRMSD_DIR: {crmsd_dir}
fraglen: 2
conformers: $CONFORMER_DIR/dinuc-XX-0.5.npy
conformer_replacements: $CONFORMER_DIR/dinuc-XX-0.5-replacement.npy
conformer_replacement_origins: $CONFORMER_DIR/dinuc-XX-0.5-replacement.txt
conformer_extensions: $CONFORMER_DIR/dinuc-XX-0.5-extension.npy
conformer_extension_origins: $CONFORMER_DIR/dinuc-XX-0.5-extension.origin.txt
rotamers: $ROTAMER_DIR/dinuc-XX-0.5.npy
rotamers_indices: $ROTAMER_DIR/dinuc-XX-0.5.index.npy
rotamer_extensions: $ROTAMER_DIR/dinuc-XX-0.5-extension.npy
rotamer_extension_indices: $ROTAMER_DIR/dinuc-XX-0.5-extension.index.npy
crmsds: $CRMSD_DIR/crmsd_matrix_XXX.npy
"""
    )
    monkeypatch.setattr("library._FRAGLIB_YAML_PATH", fraglib_yaml)

    assert np.array_equal(load_crmsds("GG", "GU"), crmsds)

    excluded = load_crmsds("AA", "AC", "1B7F")
    assert np.isinf(excluded[3, :]).all()
    assert np.isinf(excluded[:, 3]).all()
    finite_mask = np.ones(excluded.shape, dtype=bool)
    finite_mask[3, :] = False
    finite_mask[:, 3] = False
    assert np.array_equal(excluded[finite_mask], crmsds[finite_mask])


def test_verify_checksums_includes_crmsds(tmp_path) -> None:
    fraglib_root = tmp_path / "fraglib"
    checksum_dir = fraglib_root / "crmsds"
    crmsd_dir = tmp_path / "crmsds"
    checksum_dir.mkdir(parents=True)
    crmsd_dir.mkdir()

    crmsd_path = crmsd_dir / "crmsd_matrix_AAC.npy"
    np.save(crmsd_path, np.arange(4, dtype=np.float64))
    checksum = _sha256sum(crmsd_path)
    (checksum_dir / "crmsd_matrix_AAC.npy.CHECKSUM").write_text(
        f"{checksum} crmsd_matrix_AAC.npy\n"
    )

    _verify_checksums(
        {
            "conformers": str(tmp_path / "conformers" / "dinuc-XX-0.5.npy"),
            "rotamers": str(tmp_path / "rotamers" / "dinuc-XX-0.5.npy"),
            "crmsds": str(crmsd_dir / "crmsd_matrix_XXX.npy"),
        },
        fraglib_root,
    )
