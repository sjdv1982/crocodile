#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
from poses import (
    discover_pose_pairs,
    load_offset_table,
    open_pose_array,
    pose_array_length,
    pose_index_from_name,
)

GRID_SPACING = sqrt(3) / 3
MAX_MODELS = 9999

_ATOM_FORMAT_STRING = (
    "%s%5i %-4s%c%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s\n"
)


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _dinucleotide_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if len(seq) != 2:
        raise argparse.ArgumentTypeError("--sequence must be a dinucleotide (length 2)")
    if any(ch not in {"A", "C", "G", "U"} for ch in seq):
        raise argparse.ArgumentTypeError("--sequence must contain only A/C/G/U")
    return seq


def _pdb_code(code: str) -> str:
    code = code.strip()
    if len(code) != 4 or not code[0].isdigit() or not code[1:].isalnum():
        raise argparse.ArgumentTypeError(
            "PDB codes must be 4 chars: one digit + 3 alphanumeric characters (e.g. 1B7F)"
        )
    return code.upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="write_poses_pdb",
        description=(
            "Decode crocodile pose files and write the resulting coordinates as a "
            "multi-model PDB."
        ),
    )
    parser.add_argument(
        "--pose-dir",
        type=_existing_dir,
        required=True,
        help="Directory containing poses-<index>.npy(.zst) and offsets-<index>.dat",
    )
    parser.add_argument(
        "--first-index",
        type=_positive_int,
        help="First inclusive pose index (counting from 1)",
    )
    parser.add_argument(
        "--last-index",
        type=_positive_int,
        help="Last inclusive pose index (counting from 1)",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (AA/AC/.../UU)",
    )
    parser.add_argument("--output", required=True, help="Output PDB path")
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Enable fraglib checksum verification when loading the library config",
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


def _rotamers_to_matrices(rotamers: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if rotamers.ndim == 3 and rotamers.shape[1:] == (3, 3):
        return rotamers.astype(np.float32, copy=False)
    if rotamers.ndim == 2 and rotamers.shape[1] == 3:
        return Rotation.from_rotvec(rotamers).as_matrix().astype(np.float32, copy=False)
    raise ValueError(
        "Unsupported rotamer representation; expected shape [N,3] rotvec or [N,3,3] matrix"
    )


def _select_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for pose_path, offsets_path in discover_pose_pairs(args.pose_dir):
        index = pose_index_from_name(pose_path.name)
        if index is None:
            continue
        pairs.append((pose_path, offsets_path))

    if not pairs:
        raise FileNotFoundError("No pose/offsets file pairs selected")
    return pairs


def _count_total_poses(pairs: list[tuple[Path, Path]]) -> int:
    total = 0
    for pose_path, _ in pairs:
        total += pose_array_length(pose_path)
    return total


def _load_library(
    sequence: str,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
):
    from library import config
    from parse_pdb import atomic_dtype

    libraries, _ = config(verify_checksums=verify_checksums)
    if sequence not in libraries:
        raise ValueError(f"Sequence {sequence} not available in fragment library")
    libf = libraries[sequence]
    libf.load_rotaconformers()
    pdb_code_filter: str | list[str] | None = None
    if excluded_pdb_codes:
        pdb_code_filter = sorted(excluded_pdb_codes)
    lib = libf.create(pdb_code_filter, with_rotaconformers=True)
    if lib.rotaconformers is None or lib.rotaconformers_index is None:
        raise RuntimeError("Rotaconformers were not loaded")
    if lib.template.dtype != atomic_dtype:
        raise TypeError("Library template has unexpected dtype")
    if len(lib.template) != lib.coordinates.shape[1]:
        raise ValueError("Template atom count does not match conformer coordinates")
    return lib


def _write_pdb_atom(atom) -> str:
    name = atom["name"].decode()
    if not name.startswith("H"):
        name = " " + name
    occupancy = float(atom["occupancy"])
    if occupancy > 100:
        occupancy = 100
    if occupancy < 0:
        occupancy = 0
    args = (
        "ATOM  " if atom["hetero"].decode().strip() == "" else "HETATM",
        int(atom["index"]) % 100000,
        name,
        atom["altloc"].decode(),
        atom["resname"].decode(),
        atom["chain"].decode()[0],
        int(atom["resid"]),
        atom["icode"].decode(),
        float(atom["x"]),
        float(atom["y"]),
        float(atom["z"]),
        occupancy,
        min(float(atom["bfactor"]), 100.0),
        atom["segid"].decode(),
        atom["element"].decode(),
        "",
    )
    return _ATOM_FORMAT_STRING % args


def _write_model(handle, atoms: np.ndarray, model_index: int) -> None:
    handle.write(f"MODEL {model_index}\n")
    current_id: tuple[bytes, int] | None = None
    for atom in atoms:
        new_id = (bytes(atom["chain"]), int(atom["resid"]))
        if current_id is not None and new_id != current_id:
            if new_id[0] != current_id[0] or new_id[1] != current_id[1] + 1:
                handle.write("TER\n")
        handle.write(_write_pdb_atom(atom))
        current_id = new_id
    handle.write("ENDMDL\n")


def _transform_pose_atoms(
    template: np.ndarray,
    coordinates: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    model_index: int,
) -> np.ndarray:
    transformed = coordinates @ rotation
    transformed += translation[None, :]
    atoms = template.copy()
    atoms["model"] = model_index
    atoms["index"] = np.arange(1, len(atoms) + 1, dtype=np.uint32)
    atoms["x"] = transformed[:, 0]
    atoms["y"] = transformed[:, 1]
    atoms["z"] = transformed[:, 2]
    return atoms


def write_pose_pairs_to_pdb(
    pairs: list[tuple[Path, Path]],
    sequence: str,
    output_path: Path,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
    first_index: int | None,
    last_index: int | None,
) -> int:
    total_poses = _count_total_poses(pairs)
    if first_index is not None or last_index is not None:
        if last_index is None or first_index is None or first_index > last_index:
            raise ValueError("--first-index must be <= --last-index")

        first_index = min(first_index, total_poses)
        last_index = min(last_index, total_poses)
        total_poses = last_index - first_index + 1
    if total_poses > MAX_MODELS:
        raise ValueError(f"Refusing to write more than {MAX_MODELS} models to PDB")

    lib = _load_library(
        sequence,
        verify_checksums=verify_checksums,
        excluded_pdb_codes=excluded_pdb_codes,
    )
    template = lib.template
    coordinates = lib.coordinates.astype(np.float32, copy=False)

    rotamer_cache: dict[int, np.ndarray] = {}
    model_index = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        all_poses = []
        for poses_path, offsets_path in pairs:
            poses = np.asarray(open_pose_array(poses_path), dtype=np.uint16)
            if poses.ndim != 2 or poses.shape[1] != 3:
                raise ValueError(
                    f"Invalid pose array shape in {poses_path}: {poses.shape}"
                )
            offset_table = load_offset_table(offsets_path)

            for pose_number, (conf_u16, rot_u16, offset_u16) in enumerate(poses):
                conf = int(conf_u16)
                rot = int(rot_u16)
                offset_index = int(offset_u16)

                if conf >= len(coordinates):
                    raise ValueError(
                        f"Pose {pose_number} in {poses_path}: conformer index {conf} out of range"
                    )
                if offset_index >= len(offset_table):
                    raise ValueError(
                        f"Pose {pose_number} in {poses_path}: offset index {offset_index} out of range"
                    )

                rotations = rotamer_cache.get(conf)
                if rotations is None:
                    rotations = _rotamers_to_matrices(lib.get_rotamers(conf))
                    rotamer_cache[conf] = rotations
                if rot >= len(rotations):
                    raise ValueError(
                        f"Pose {pose_number} in {poses_path}: rotamer index {rot} out of range "
                        f"for conformer {conf} (n={len(rotations)})"
                    )

                translation = (
                    offset_table[offset_index].astype(np.float32, copy=False)
                    * GRID_SPACING
                )
                all_poses.append((conf, rotations[rot], translation))

        if first_index is not None:
            assert last_index is not None
            all_poses = all_poses[first_index - 1 : last_index]
        for conf, rotation, translation in all_poses:
            atoms = _transform_pose_atoms(
                template=template,
                coordinates=coordinates[conf],
                rotation=rotation,
                translation=translation,
                model_index=model_index,
            )
            _write_model(handle, atoms, model_index)
            model_index += 1

    assert model_index - 1 == total_poses
    return total_poses


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pairs = _select_pairs(args)
    first_index = args.first_index
    last_index = args.last_index
    total = write_pose_pairs_to_pdb(
        pairs=pairs,
        sequence=args.sequence,
        output_path=Path(args.output),
        verify_checksums=args.verify_checksums,
        excluded_pdb_codes=set(args.pdb_exclude),
        first_index=first_index,
        last_index=last_index,
    )
    print(f"Wrote {total} models to {args.output}")


if __name__ == "__main__":
    main()
