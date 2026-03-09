#!/usr/bin/env python3
"""Convert rotvec/world pose DOFs to a legacy ATTRACT .dat file.

Input DOFs are assumed to be:
- columns 0-2: rotvec
- columns 3-5: world-frame translation

Output .dat:
- ligand line uses Euler angles
- translations are ATTRACT pivot-centered translations
- the pivot is the fixed ligand pivot defined by conformer 1 / index 0
- ligand ensemble indices are written on the ligand line when provided
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import sys

_UTIL_DIR = Path(__file__).resolve().parents[2] / "attract-jax" / "util"
if str(_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_UTIL_DIR))
from mat4_to_dat import rotmat_to_euler_attr


def _existing_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _load_pivot(ligand_ensemble: str | None, ligand_pdb: str | None) -> np.ndarray:
    if ligand_ensemble:
        coords = np.load(ligand_ensemble).astype(np.float64)
        if coords.ndim == 2:
            coords = coords[None, ...]
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"Unsupported ligand ensemble shape: {coords.shape}")
        return np.asarray(coords[0].mean(axis=0), dtype=np.float64)
    if ligand_pdb:
        atoms = []
        with open(ligand_pdb) as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    atoms.append(
                        (
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        )
                    )
        if not atoms:
            raise ValueError(f"No atom coordinates found in {ligand_pdb}")
        return np.asarray(atoms, dtype=np.float64).mean(axis=0)
    raise ValueError("One of --ligand-ensemble or --ligand-pdb is required")


def _load_dat_conformers(
    conformers0_path: str | None,
    conformers1_path: str | None,
    nposes: int,
) -> np.ndarray:
    if conformers1_path:
        conf1 = np.load(conformers1_path).astype(np.int32).reshape(-1)
        if len(conf1) != nposes:
            raise ValueError(
                f"--input-conformers1 length mismatch: expected {nposes}, got {len(conf1)}"
            )
        if conf1.size and conf1.min() < 1:
            raise ValueError("--input-conformers1 must contain 1-based indices")
        return conf1
    if conformers0_path:
        conf0 = np.load(conformers0_path).astype(np.int32).reshape(-1)
        if len(conf0) != nposes:
            raise ValueError(
                f"--input-conformers0 length mismatch: expected {nposes}, got {len(conf0)}"
            )
        if conf0.size and conf0.min() < 0:
            raise ValueError("--input-conformers0 must contain 0-based indices")
        return conf0 + 1
    return np.ones((nposes,), dtype=np.int32)


def convert_to_legacy_dat(
    rotvec_npy: str,
    output_dat: str,
    ligand_ensemble: str | None = None,
    ligand_pdb: str | None = None,
    conformers0_npy: str | None = None,
    conformers1_npy: str | None = None,
) -> None:
    dofs = np.load(rotvec_npy).astype(np.float64)
    if dofs.ndim != 2 or dofs.shape[1] != 6:
        raise ValueError(f"Expected Nx6 DOFs in {rotvec_npy}, got {dofs.shape}")
    pivot = _load_pivot(ligand_ensemble, ligand_pdb)
    conformers1 = _load_dat_conformers(conformers0_npy, conformers1_npy, len(dofs))

    with open(output_dat, "w") as f:
        f.write("#pivot auto\n")
        f.write("#centered receptor: false\n")
        f.write("#centered ligands: false\n")
        for i, dof in enumerate(dofs, start=1):
            conf1 = int(conformers1[i - 1])
            rot_col = Rotation.from_rotvec(dof[:3]).as_matrix()
            tx, ty, tz = dof[3:6] - pivot + rot_col @ pivot
            phi, ssi, rot = rotmat_to_euler_attr(rot_col)
            f.write(f"#{i}\n")
            f.write("           0           0           0           0           0           0\n")
            f.write(
                f"{conf1:12d} {phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                f"{tx:24.16f} {ty:24.16f} {tz:24.16f}\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-rotvec", required=True, type=_existing_file)
    ap.add_argument("--output-dat", required=True)
    ap.add_argument("--input-conformers0", type=_existing_file, default=None)
    ap.add_argument("--input-conformers1", type=_existing_file, default=None)
    ap.add_argument("--ligand-ensemble", type=_existing_file, default=None)
    ap.add_argument("--ligand-pdb", type=_existing_file, default=None)
    args = ap.parse_args()

    if bool(args.ligand_ensemble) == bool(args.ligand_pdb):
        raise SystemExit("Exactly one of --ligand-ensemble or --ligand-pdb is required")

    convert_to_legacy_dat(
        rotvec_npy=args.input_rotvec,
        output_dat=args.output_dat,
        ligand_ensemble=args.ligand_ensemble,
        ligand_pdb=args.ligand_pdb,
        conformers0_npy=args.input_conformers0,
        conformers1_npy=args.input_conformers1,
    )
    print(args.output_dat)


if __name__ == "__main__":
    main()
