from __future__ import annotations

import argparse
from pathlib import Path
from math import sqrt
import sys
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from library import config
from offsets import get_discrete_offsets
from parse_pdb import parse_pdb
from poses import PoseStreamAccumulator
from rna_pdb import ppdb2nucseq


GRID_SPACING = sqrt(3) / 3
LATERAL_SLOPE = 1.78966

RINGS = {
    "T": ["C6", "C5", "C7", "N3", "O2", "O4"],
    "U": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "C": ["N1", "C2", "N3", "C4", "C5", "C6"],
    "G": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
}
for left, right in (
    ("DT", "T"),
    ("RU", "U"),
    ("DC", "C"),
    ("RC", "C"),
    ("DG", "G"),
    ("RG", "G"),
    ("DA", "A"),
    ("RA", "A"),
):
    RINGS[left] = RINGS[right]


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters (e.g. 1B7F)"
        )
    return code.upper()


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    allowed = set("ACGU")
    if any(ch not in allowed for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _dihedral_pair(values: Sequence[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--dihedral requires 2 floats: MIN MAX")
    try:
        a = float(values[0])
        b = float(values[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dihedral values must be floats") from exc
    return (a, b)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack",
        description="Create stacking-based initial poses from a protein and a dinucleotide library.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a full traceback on errors.",
    )

    parser.add_argument(
        "--protein",
        required=True,
        type=_existing_file,
        help="PDB file of the protein.",
    )
    parser.add_argument(
        "--resid",
        required=True,
        type=int,
        help="Residue ID (numeric) of the protein aromatic residue that stacks.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (e.g. AA, AC, GU).",
    )
    parser.add_argument(
        "--dihedral",
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        required=True,
        help="Minimum and maximum dihedral angle in degrees.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Maximum stacking angle (in degrees).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Distance vector slack in angstroms (default: 0.5).",
    )
    nuc_group = parser.add_mutually_exclusive_group(required=True)
    nuc_group.add_argument(
        "--first",
        action="store_true",
        help="First nucleotide stacks (exactly one of --first/--second required).",
    )
    nuc_group.add_argument(
        "--second",
        action="store_true",
        help="Second nucleotide stacks (exactly one of --first/--second required).",
    )
    parser.add_argument(
        "--output",
        nargs=2,
        metavar=("POSES_OUT", "OFFSETS_OUT"),
        required=True,
        help=(
            "Two output filenames: pose indices (e.g. poses.npy) and offset table "
            "(e.g. offsets.dat). Names are not defaulted."
        ),
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=0,
        help="Random seed for test options (default: 0).",
    )
    parser.add_argument(
        "--test-conformers",
        type=int,
        default=None,
        metavar="N",
        help="If set, reduce the conformer library to N elements selected at random.",
    )
    parser.add_argument(
        "--test-rotamers",
        type=int,
        default=None,
        metavar="M",
        help=(
            "If set, reduce the rotamer library to M elements per conformer. "
            "The same rotamers are selected for all conformers from among the first R rotamers, "
            "where R is the smallest rotamer list among all conformers."
        ),
    )
    parser.add_argument(
        "--pdb-exclude",
        nargs="+",
        default=[],
        type=_pdb_code,
        metavar="PDB",
        help="One or more PDB codes to exclude from the fragment library.",
    )
    return parser


def _as_coordinates(mol: np.ndarray) -> np.ndarray:
    return np.stack((mol["x"], mol["y"], mol["z"]), axis=-1)


def _calc_plane(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=0)
    _, _, wt = np.linalg.svd(centered.T.dot(centered))
    plane_directionless = wt[2] / np.linalg.norm(wt[2])
    plane0_directed = np.cross((centered[1] - centered[0]), (centered[2] - centered[1]))
    flip = np.sign(np.dot(plane_directionless, plane0_directed))
    return plane_directionless * flip


def _calc_planes(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=1)[:, None, :]
    covar = np.einsum("ijk,ijl->ikl", centered, centered)
    _, _, wt = np.linalg.svd(covar)
    directionless = wt[:, 2, :] / np.linalg.norm(wt[:, 2, :], axis=-1)[:, None]
    directed = np.cross(
        centered[:, 1, :] - centered[:, 0, :],
        centered[:, 2, :] - centered[:, 1, :],
        axis=1,
    )
    flip = np.sign(np.einsum("ij,ij->i", directionless, directed))
    return directionless * flip[:, None]


def _protein_ring_coordinates(protein_atoms: np.ndarray, resid: int) -> np.ndarray:
    mask_resid = protein_atoms["resid"] == resid
    if not np.any(mask_resid):
        raise ValueError(f"Residue {resid} was not found in protein")

    chains = np.unique(protein_atoms["chain"][mask_resid])
    if len(chains) != 1:
        chains_str = ", ".join(
            ch.decode(errors="ignore").strip() or "<blank>" for ch in chains
        )
        raise ValueError(
            f"Residue ID {resid} is ambiguous across chains ({chains_str}); select a unique residue ID"
        )

    resnames = np.unique(protein_atoms["resname"][mask_resid])
    if len(resnames) != 1:
        raise ValueError(f"Residue {resid} resolves to multiple residue names")
    resname = resnames[0].decode()
    if resname not in RINGS:
        raise ValueError(
            f"Residue {resid} ({resname}) is not aromatic or unsupported; expected one of {sorted(RINGS)}"
        )

    ring_names = [name.encode() for name in RINGS[resname]]
    mask_ring = mask_resid & np.isin(protein_atoms["name"], ring_names)
    if not np.any(mask_ring):
        raise ValueError(f"Could not find ring atoms for residue {resid} ({resname})")
    coordinates = _as_coordinates(protein_atoms[mask_ring])
    if len(coordinates) < 3:
        raise ValueError(f"Residue {resid} has too few ring atoms ({len(coordinates)})")
    return coordinates


def _select_library(
    sequence: str,
    first: bool,
    excluded_pdb_codes: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dinucleotide_libraries, dinucleotide_templates = config(verify_checksums=False)
    if sequence not in dinucleotide_libraries:
        raise ValueError(f"Unsupported sequence: {sequence}")

    factory = dinucleotide_libraries[sequence]
    template = dinucleotide_templates[sequence]
    nucleotide_mask = np.array([first, not first], dtype=bool)

    factory.load_rotaconformers()
    pdb_code_filter: str | list[str] | None = None
    if excluded_pdb_codes:
        pdb_code_filter = sorted(excluded_pdb_codes)
    library = factory.create(
        pdb_code=pdb_code_filter,
        nucleotide_mask=nucleotide_mask,
        only_base=True,
        with_rotaconformers=True,
    )
    if library.rotaconformers is None or library.rotaconformers_index is None:
        raise RuntimeError("Rotaconformers were not loaded")

    coordinates = library.coordinates
    conformer_mask = np.ones(len(coordinates), dtype=bool)
    if library.conformer_mask is not None:
        conformer_mask &= library.conformer_mask

    sequence_from_template, nucleotide_indices = ppdb2nucseq(
        template,
        rna=True,
        return_index=True,
    )
    if sequence_from_template != sequence:
        raise ValueError(
            f"Template sequence mismatch: expected {sequence}, got {sequence_from_template}"
        )

    nucleotide_index = 0 if first else 1
    atom_start, atom_end = nucleotide_indices[nucleotide_index]
    residue_name = template[atom_start]["resname"].decode()
    if residue_name not in RINGS:
        raise ValueError(f"Unsupported nucleobase residue {residue_name}")
    ring_names = [name.encode() for name in RINGS[residue_name]]

    ring_mask_full = np.zeros(len(template), dtype=bool)
    ring_mask_full[atom_start:atom_end] = np.isin(
        template[atom_start:atom_end]["name"],
        ring_names,
    )
    if library.atom_mask is not None:
        ring_mask = ring_mask_full[library.atom_mask]
    else:
        ring_mask = ring_mask_full
    if not np.any(ring_mask):
        raise ValueError(f"No nucleobase ring atoms selected for {sequence}")

    factory.unload_rotaconformers()
    result = (
        coordinates,
        conformer_mask,
        ring_mask,
        library.rotaconformers,
        library.rotaconformers_index,
    )
    del library
    return result


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


def _select_conformers(
    available_indices: np.ndarray, count: int | None, seed: int
) -> np.ndarray:
    if count is None or count >= len(available_indices):
        return available_indices.copy()
    if count <= 0:
        raise ValueError("--test-conformers must be positive")
    rng = np.random.default_rng(seed)
    return rng.choice(available_indices, size=count, replace=False)


def _dihedral_filter(
    angles: np.ndarray,
    dihedral: np.ndarray,
    max_angle_deg: float,
    dihedral_min_deg: float,
    dihedral_max_deg: float,
) -> np.ndarray:
    mask = angles <= max_angle_deg / 180.0 * np.pi
    if abs(dihedral_min_deg) > 0.0001 and abs(dihedral_max_deg) > 0.0001:
        dihedral_min = dihedral_min_deg / 180.0 * np.pi
        dihedral_max = dihedral_max_deg / 180.0 * np.pi
        if dihedral_max < dihedral_min:
            dihedral_mask = (dihedral <= dihedral_max) | (dihedral >= dihedral_min)
        else:
            dihedral_mask = (dihedral >= dihedral_min) & (dihedral <= dihedral_max)
        mask &= dihedral_mask
    return mask


def _write_output_files(
    packed: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    poses_path: Path,
    offsets_path: Path,
) -> None:
    if len(packed) > 1:
        raise ValueError(
            f"Result requires {len(packed)} pose chunks, but --output only accepts one poses/offsets pair"
        )

    if packed:
        poses, mean_offset, offsets = packed[0]
    else:
        poses = np.empty((0, 3), dtype=np.uint16)
        mean_offset = np.zeros((3,), dtype=np.int16)
        offsets = np.empty((0, 3), dtype=np.uint8)

    poses_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(poses_path), np.asarray(poses, dtype=np.uint16))

    mean_offset = np.asarray(mean_offset, dtype=np.int16)
    offsets = np.asarray(offsets, dtype=np.uint8)
    with offsets_path.open("wb") as handle:
        handle.write(mean_offset.tobytes(order="C"))
        handle.write(offsets.tobytes(order="C"))


def _run(args: argparse.Namespace) -> int:
    dihedral_min_deg, dihedral_max_deg = _dihedral_pair(args.dihedral)
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")

    print("[1/7] Loading protein structure...", file=sys.stderr)
    protein_atoms = parse_pdb(Path(args.protein).read_text())
    if np.any(protein_atoms["model"] == 1):
        protein_atoms = protein_atoms[protein_atoms["model"] == 1]

    protein_ring = _protein_ring_coordinates(protein_atoms, args.resid)
    protein_center = protein_ring.mean(axis=0)
    protein_plane = _calc_plane(protein_ring)
    protein_x = protein_ring[0] - protein_center
    protein_x /= np.linalg.norm(protein_x)
    protein_y = np.cross(protein_plane, protein_x)
    protein_y /= np.linalg.norm(protein_y)

    print("[2/7] Loading fragment library...", file=sys.stderr)
    coordinates, conformer_mask, ring_mask, rotaconformers, rotaconformers_index = (
        _select_library(
            sequence=args.sequence,
            first=args.first,
            excluded_pdb_codes=set(args.pdb_exclude),
        )
    )

    available = np.nonzero(conformer_mask)[0]
    if len(available) == 0:
        raise ValueError("No conformers available after exclusions")
    selected_conformers = _select_conformers(
        available, args.test_conformers, args.test_seed
    )
    print(
        f"[3/7] Filtering angle/dihedral for {len(selected_conformers)} conformers...",
        file=sys.stderr,
    )

    rng = np.random.default_rng(args.test_seed)
    selected_rotamer_positions: np.ndarray | None = None
    if args.test_rotamers is not None:
        if args.test_rotamers <= 0:
            raise ValueError("--test-rotamers must be positive")
        rotamer_counts = np.diff(rotaconformers_index, prepend=0)
        smallest_count = int(rotamer_counts[selected_conformers].min())
        if args.test_rotamers > smallest_count:
            raise ValueError(
                f"--test-rotamers={args.test_rotamers} is larger than the smallest selected rotamer list ({smallest_count})"
            )
        selected_rotamer_positions = rng.choice(
            smallest_count,
            size=args.test_rotamers,
            replace=False,
        )

    print("[4/7] Generating offset candidates...", file=sys.stderr)
    print("[5/7] Applying distance-vector filter...", file=sys.stderr)
    center_batch_size = 512 if args.test_rotamers is not None else 256
    offset_chunk_size = 2_000_000
    total_candidates = 0
    total_surviving = 0
    reverse_table: np.ndarray | None = None

    stream_acc = PoseStreamAccumulator(Path(args.output[0]), Path(args.output[1]))
    distance_pbar = tqdm(
        desc="Offset-distance filter",
        unit="cand",
        mininterval=2.0,
    )
    try:
        for conformer_index in tqdm(
            selected_conformers,
            desc="Conformer-angle filter",
            unit="conf",
            mininterval=2.0,
        ):
            ring_coordinates = coordinates[conformer_index, ring_mask]
            if len(ring_coordinates) < 3:
                continue

            rotamer_end = int(rotaconformers_index[conformer_index])
            rotamer_start = (
                0
                if conformer_index == 0
                else int(rotaconformers_index[conformer_index - 1])
            )
            rotamer_count = rotamer_end - rotamer_start
            if rotamer_count <= 0:
                continue

            if selected_rotamer_positions is None:
                local_rotamer_indices = np.arange(rotamer_count, dtype=np.int64)
            else:
                local_rotamer_indices = selected_rotamer_positions.astype(
                    np.int64, copy=False
                )
            global_rotamer_indices = rotamer_start + local_rotamer_indices

            rotamer_matrices = _rotamers_to_matrices(
                rotaconformers[global_rotamer_indices]
            )
            ring_rotated = np.einsum("ij,kjl->kil", ring_coordinates, rotamer_matrices)
            ring_planes = _calc_planes(ring_rotated)

            cross_norm = np.linalg.norm(
                np.cross(protein_plane[None, :], ring_planes), axis=1
            )
            cross_norm = np.minimum(cross_norm, 1.0)
            angles = np.arcsin(cross_norm)

            ring_centers = ring_rotated.mean(axis=1)
            ring_x = ring_rotated[:, 0, :] - ring_centers
            ring_x -= (
                np.einsum("ij,j->i", ring_x, protein_plane)[:, None]
                * protein_plane[None, :]
            )
            ring_x_norm = np.linalg.norm(ring_x, axis=1)
            degenerate = ring_x_norm < 0.0001
            safe_norm = np.where(degenerate, 1.0, ring_x_norm)
            normalized_ring_x = ring_x / safe_norm[:, None]
            dihedral_x = np.einsum("ij,j->i", normalized_ring_x, protein_x)
            dihedral_y = np.einsum("ij,j->i", normalized_ring_x, protein_y)
            dihedral = np.angle(dihedral_x + 1.0j * dihedral_y)
            dihedral[degenerate] = np.pi / 2

            angle_mask = _dihedral_filter(
                angles,
                dihedral,
                max_angle_deg=args.angle,
                dihedral_min_deg=dihedral_min_deg,
                dihedral_max_deg=dihedral_max_deg,
            )
            if not np.any(angle_mask):
                continue

            passing_rotamers_raw = local_rotamer_indices[angle_mask]
            if (
                len(passing_rotamers_raw)
                and passing_rotamers_raw.max() > np.iinfo(np.uint16).max
            ):
                raise ValueError(
                    "Rotamer index exceeds uint16 range; use --test-rotamers to constrain per-conformer rotamers"
                )
            if conformer_index > np.iinfo(np.uint16).max:
                raise ValueError("Conformer index exceeds uint16 range")

            passing_rotamers = passing_rotamers_raw.astype(np.uint16, copy=False)
            passing_centers = ring_centers[angle_mask].astype(np.float32, copy=False)
            if len(passing_centers) == 0:
                continue

            for batch_start in range(0, len(passing_centers), center_batch_size):
                batch_end = min(batch_start + center_batch_size, len(passing_centers))
                centers_batch = passing_centers[batch_start:batch_end]
                rot_batch = passing_rotamers[batch_start:batch_end]

                displacement_world = protein_center[None, :] - centers_batch
                displacement_grid = displacement_world / GRID_SPACING
                radius_grid = np.full(
                    (len(displacement_grid),),
                    (5.0 + args.margin) / GRID_SPACING,
                    dtype=np.float32,
                )
                disp_indices, p_indices, rounded, reverse_map = get_discrete_offsets(
                    displacement_grid,
                    radius_grid,
                )
                if len(disp_indices) == 0:
                    continue

                total_candidates += len(disp_indices)
                distance_pbar.update(len(disp_indices))

                if reverse_table is None:
                    reverse_table = np.empty(
                        (max(reverse_map.keys()) + 1, 3),
                        dtype=np.int16,
                    )
                    for p_index, xyz in reverse_map.items():
                        reverse_table[p_index] = xyz

                for offset_start in range(0, len(disp_indices), offset_chunk_size):
                    offset_end = min(offset_start + offset_chunk_size, len(disp_indices))
                    disp_chunk = disp_indices[offset_start:offset_end]
                    p_chunk = p_indices[offset_start:offset_end]

                    offset_grid = rounded[disp_chunk].astype(np.int16) + reverse_table[p_chunk]
                    center_vec = (
                        offset_grid.astype(np.float32) * GRID_SPACING
                        - displacement_world[disp_chunk]
                    )

                    center_z = np.abs(center_vec[:, 2])
                    center_dis = np.linalg.norm(center_vec, axis=1)
                    axial = np.abs(center_vec.dot(protein_plane))
                    axial = np.maximum(0.0, np.minimum(center_dis - 0.001, axial))
                    lateral = np.sqrt(
                        np.maximum(0.0, center_dis * center_dis - axial * axial)
                    )
                    center_dis_corrected = center_dis - lateral / LATERAL_SLOPE

                    mask_a = (center_z >= 2.3 - args.margin) & (
                        center_z <= 4.5 + args.margin
                    )
                    mask_b = (center_dis_corrected >= 2.3 - args.margin) & (
                        center_dis_corrected <= 3.8 + args.margin
                    )
                    mask_c = center_dis <= 5.0 + args.margin
                    keep_mask = mask_a & mask_b & mask_c
                    if not np.any(keep_mask):
                        continue

                    kept_disp = disp_chunk[keep_mask]
                    kept_p = p_chunk[keep_mask]
                    total_surviving += len(kept_disp)

                    translations = (
                        rounded[kept_disp].astype(np.int16)
                        + reverse_table[kept_p]
                    )
                    conformers_chunk = np.full(
                        len(kept_disp),
                        conformer_index,
                        dtype=np.uint16,
                    )
                    rotamers_chunk = rot_batch[kept_disp]
                    stream_acc.add_chunk(
                        conformers_chunk,
                        rotamers_chunk,
                        translations,
                    )
        distance_pbar.close()
        del rotaconformers, rotaconformers_index

        print(
            f"[6/7] Candidate offset assignments: {total_candidates}",
            file=sys.stderr,
        )
        print(
            f"[6/7] Surviving offset assignments: {total_surviving}",
            file=sys.stderr,
        )
        print("[7/7] Packing and writing poses...", file=sys.stderr)
        written = stream_acc.finish()
        if len(written) > 1:
            print(
                f"[7/7] Output split into {len(written)} chunks (poses-*.npy, offsets-*.dat).",
                file=sys.stderr,
            )

        print(f"{stream_acc.total_poses} total solutions", file=sys.stderr)
        return 0
    finally:
        if hasattr(distance_pbar, "disable") and not distance_pbar.disable:
            distance_pbar.close()
        stream_acc.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(_run(args))
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
