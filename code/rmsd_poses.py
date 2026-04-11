#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
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


def _existing_dir(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _existing_pdb_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    if p.suffix.lower() != ".pdb":
        raise argparse.ArgumentTypeError(f"reference must be a .pdb file: {path}")
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
        prog="rmsd_poses",
        description=(
            "Decode crocodile pose files, compare them to a reference PDB, and "
            "write one RMSD per pose."
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
        help="First inclusive shard index when using --pose-dir",
    )
    parser.add_argument(
        "--last-index",
        type=_positive_int,
        help="Last inclusive shard index when using --pose-dir",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=_dinucleotide_sequence,
        help="Dinucleotide sequence (AA/AC/.../UU)",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=_existing_pdb_file,
        help="Reference PDB file whose topology must match the library template",
    )
    parser.add_argument("--output", required=True, help="Output RMSD path")
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
    first_index = args.first_index
    last_index = args.last_index
    if first_index is not None and last_index is not None and first_index > last_index:
        raise ValueError("--first-index must be <= --last-index")

    pairs: list[tuple[Path, Path]] = []
    for pose_path, offsets_path in discover_pose_pairs(args.pose_dir):
        index = pose_index_from_name(pose_path.name)
        if index is None:
            continue
        if first_index is not None and index < first_index:
            continue
        if last_index is not None and index > last_index:
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


def _load_reference(reference_path: Path, template: np.ndarray) -> np.ndarray:
    from parse_pdb import parse_pdb

    reference = parse_pdb(reference_path.read_text())
    if len(reference) == 0:
        raise ValueError("Reference PDB contains no atoms")
    model_ids = np.unique(reference["model"])
    if len(model_ids) > 1:
        reference = reference[reference["model"] == model_ids[0]]
        if len(reference) == 0:
            raise ValueError("Reference PDB first model contains no atoms")

    if reference.shape != template.shape:
        raise ValueError(
            f"Reference/template shape mismatch: {reference.shape} vs {template.shape}"
        )

    def _normalize_resnames(values: np.ndarray) -> np.ndarray:
        names = np.char.strip(values.astype("U"))
        normalized = names.copy()
        mask = np.char.startswith(names, "R") & (np.char.str_len(names) == 2)
        normalized[mask] = np.char.lstrip(normalized[mask], "R")
        return normalized

    def _normalize_resids(values: np.ndarray) -> np.ndarray:
        resid = values.astype(np.int64, copy=False)
        return resid - int(resid[0])

    for field in template.dtype.names:
        if field in {"x", "y", "z", "occupancy", "segid"}:
            continue
        if field == "chain":
            continue
        if field == "resname":
            equal = np.array_equal(
                _normalize_resnames(reference[field]),
                _normalize_resnames(template[field]),
            )
        elif field == "resid":
            equal = np.array_equal(
                _normalize_resids(reference[field]),
                _normalize_resids(template[field]),
            )
        else:
            equal = np.array_equal(reference[field], template[field])
        if not equal:
            raise ValueError(
                f"Reference topology does not match template in field '{field}'"
            )

    return np.stack((reference["x"], reference["y"], reference["z"]), axis=-1).astype(
        np.float32,
        copy=False,
    )


def _detect_output_mode(path: Path) -> tuple[str, bool]:
    lower = path.name.lower()
    compression = False
    base = lower
    for suffix in (".zstd", ".zst"):
        if base.endswith(suffix):
            compression = True
            base = base[: -len(suffix)]
            break
    file_format = "npy" if base.endswith(".npy") else "text"
    return file_format, compression


def _require_zstandard(action: str):
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(f"zstandard is required to {action} zstd-compressed output") from exc
    return zstd


def _compress_file(source_path: Path, target_path: Path) -> None:
    zstd = _require_zstandard("write")
    with tempfile.NamedTemporaryFile(
        prefix=f"{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
        delete=False,
    ) as handle:
        tmp_target_path = Path(handle.name)
    try:
        with source_path.open("rb") as src, tmp_target_path.open("wb") as dst:
            zstd.ZstdCompressor().copy_stream(src, dst)
        tmp_target_path.replace(target_path)
    finally:
        if tmp_target_path.exists():
            tmp_target_path.unlink()


def _write_rmsd_output(path: Path, rmsd: np.ndarray) -> None:
    file_format, compression = _detect_output_mode(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not compression:
        if file_format == "npy":
            np.save(path, rmsd)
        else:
            np.savetxt(path, rmsd, fmt="%.6f")
        return

    suffix = ".npy" if file_format == "npy" else ".txt"
    with tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}.",
        suffix=suffix,
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        if file_format == "npy":
            np.save(temp_path, rmsd)
        else:
            np.savetxt(temp_path, rmsd, fmt="%.6f")
        _compress_file(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def compute_rmsd(
    pairs: list[tuple[Path, Path]],
    sequence: str,
    reference_path: Path,
    verify_checksums: bool,
    excluded_pdb_codes: set[str],
) -> np.ndarray:
    total_poses = _count_total_poses(pairs)
    lib = _load_library(
        sequence,
        verify_checksums=verify_checksums,
        excluded_pdb_codes=excluded_pdb_codes,
    )
    template = lib.template
    coordinates = lib.coordinates.astype(np.float32, copy=False)
    reference_coords = _load_reference(reference_path, template)
    natoms = len(reference_coords)

    rmsd = np.empty((total_poses,), dtype=np.float32)
    rotamer_cache: dict[int, np.ndarray] = {}
    output_offset = 0
    pose_batch_size = 4096

    for poses_path, offsets_path in pairs:
        poses = np.asarray(open_pose_array(poses_path), dtype=np.uint16)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"Invalid pose array shape in {poses_path}: {poses.shape}")

        if len(poses) == 0:
            continue

        offset_table = load_offset_table(offsets_path)

        conf_ids = poses[:, 0]
        rot_ids = poses[:, 1]
        offset_ids = poses[:, 2].astype(np.int64, copy=False)
        if offset_ids.size and int(offset_ids.max()) >= len(offset_table):
            raise ValueError(f"Pose offset index out of range in {poses_path}")
        translations = (
            offset_table[offset_ids].astype(np.float32, copy=False) * GRID_SPACING
        )

        conf_order = np.argsort(conf_ids, kind="stable")
        sorted_conf = conf_ids[conf_order]
        boundaries = np.flatnonzero(np.diff(sorted_conf)) + 1
        starts = np.concatenate((np.array([0], dtype=np.intp), boundaries))
        stops = np.concatenate((boundaries, np.array([len(conf_order)], dtype=np.intp)))

        local_rmsd = np.empty((len(poses),), dtype=np.float32)
        for start, stop in zip(starts, stops):
            rows = conf_order[start:stop]
            conf = int(sorted_conf[start])
            if conf >= len(coordinates):
                raise ValueError(
                    f"Conformer index {conf} out of range in {poses_path}"
                )

            rotations = rotamer_cache.get(conf)
            if rotations is None:
                rotations = _rotamers_to_matrices(lib.get_rotamers(conf))
                rotamer_cache[conf] = rotations

            for batch_start in range(0, len(rows), pose_batch_size):
                batch_rows = rows[batch_start : batch_start + pose_batch_size]
                batch_rot = rot_ids[batch_rows].astype(np.int64, copy=False)
                if batch_rot.size and int(batch_rot.max()) >= len(rotations):
                    bad_pos = int(np.flatnonzero(batch_rot >= len(rotations))[0])
                    bad_row = int(batch_rows[bad_pos])
                    bad_rot = int(batch_rot[bad_pos])
                    raise ValueError(
                        f"Pose {bad_row} in {poses_path}: rotamer index {bad_rot} out of range "
                        f"for conformer {conf} (n={len(rotations)})"
                    )

                transformed = np.einsum(
                    "aj,njk->nak",
                    coordinates[conf],
                    rotations[batch_rot],
                )
                transformed += translations[batch_rows, None, :]
                dif = transformed - reference_coords[None, :, :]
                local_rmsd[batch_rows] = np.sqrt(
                    np.einsum("nij,nij->n", dif, dif) / natoms
                )

        rmsd[output_offset : output_offset + len(poses)] = local_rmsd
        output_offset += len(poses)

    assert output_offset == total_poses
    return rmsd


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pairs = _select_pairs(args)
    rmsd = compute_rmsd(
        pairs=pairs,
        sequence=args.sequence,
        reference_path=Path(args.reference),
        verify_checksums=args.verify_checksums,
        excluded_pdb_codes=set(args.pdb_exclude),
    )
    output_path = Path(args.output)
    _write_rmsd_output(output_path, rmsd)
    print(f"Wrote {len(rmsd)} RMSD values to {output_path}")


if __name__ == "__main__":
    main()
