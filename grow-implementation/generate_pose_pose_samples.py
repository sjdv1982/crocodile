#!/usr/bin/env python3
from __future__ import annotations

"""
Test set for grow (pose-pose superposition) implementation.
Create 200 samples, using a random seed of 0.
For each sample:
- Choose two random sequences AB and BC, with A,B,C being "A" or "C"
- Choose a random direction "forward" or "backward".
- Load the conformer library for AB (or BC if "backward") and choose a random conformer D
- Load the rotaconformer index library for AB and choose a random rotaconformer E among the rotaconformers of conformer D
- Choose a random cRMSD threshold F between 0.5 and 1. This is done by choosing a random fraction between 0 and 1, multiplying by (1 - 0.5) and adding 0.5.
  To bias F towards lower values, before multiplying, take the square of the random fraction.
- Choose a overlap RMSD value G between F and 1.3 . This is done by choosing a random fraction between 0 and 1, multiplying by (1.3 - F) and adding F.
  To bias G towards lower values, before multiplying, take the fourth power of the random fraction.
"""
import csv
import gc
import sys
from pathlib import Path

import numpy as np


N_SAMPLES = 200
SEED = 0
BASES = ("A", "C")
DIRECTIONS = ("forward", "backward")
OUTPUT = Path(__file__).with_name("pose_pose_samples.csv")


def _library_directory():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "code"))

    from library import (
        LibraryDirectory,
        _fraglib_config,
        _require_int,
        _require_str,
        _resolve_templates_dir,
    )

    config_data = _fraglib_config()
    fraglib_root = repo_root / "fraglib"
    templates_dir = _resolve_templates_dir(config_data, fraglib_root)
    directory = LibraryDirectory(
        fraglen=_require_int(config_data, "fraglen"),
        filepattern=_require_str(config_data, "conformers"),
        replacement_filepattern=_require_str(config_data, "conformer_replacements"),
        replacement_origin_filepattern=_require_str(
            config_data, "conformer_replacement_origins"
        ),
        extension_filepattern=_require_str(config_data, "conformer_extensions"),
        extension_origin_filepattern=_require_str(
            config_data, "conformer_extension_origins"
        ),
        rotaconformers_filepattern=_require_str(config_data, "rotamers"),
        rotaconformers_index_filepattern=_require_str(config_data, "rotamers_indices"),
        rotaconformers_extension_filepattern=_require_str(
            config_data, "rotamer_extensions"
        ),
        rotaconformers_extension_index_filepattern=_require_str(
            config_data, "rotamer_extension_indices"
        ),
    )
    return directory, templates_dir


def _load_library(directory: object, templates_dir: Path, ab: str):
    template = np.load(str(templates_dir / f"{ab}-ppdb.npy"))
    factory = directory.load(
        sequence=ab,
        template=template,
        with_extension=True,
        with_replacement=True,
        with_rotaconformers=True,
    )
    factory.load_rotaconformers()
    return factory, factory.create(None, with_rotaconformers=True)


def _local_rotamer_count(library: object, conformer: int) -> int:
    rotaconformers_index = library.rotaconformers_index
    if rotaconformers_index is None:
        raise RuntimeError("Rotaconformer index library was not loaded")
    end = int(rotaconformers_index[conformer])
    start = 0 if conformer == 0 else int(rotaconformers_index[conformer - 1])
    return end - start


def main() -> None:
    rng = np.random.default_rng(SEED)
    directory, templates_dir = _library_directory()

    rows = []
    for sample_id in range(N_SAMPLES):
        a, b, c = rng.choice(BASES, size=3)
        ab = str(a + b)
        bc = str(b + c)
        direction = str(rng.choice(DIRECTIONS))
        crmsd_fraction = float(rng.random()) ** 2
        crmsd_threshold = 0.5 + crmsd_fraction * (1.0 - 0.5)

        overlap_fraction = float(rng.random()) ** 4
        overlap_rmsd = crmsd_threshold + overlap_fraction * (1.3 - crmsd_threshold)

        rows.append(
            {
                "sample_id": sample_id,
                "ab": ab,
                "bc": bc,
                "direction": direction,
                "library": ab if direction == "forward" else bc,
                "conformer": None,
                "rotaconformer": None,
                "crmsd_threshold": f"{crmsd_threshold:.4f}",
                "overlap_rmsd": f"{overlap_rmsd:.4f}",
            }
        )

    for ab in ("AA", "AC", "CA", "CC"):
        factory, library = _load_library(directory, templates_dir, ab)
        try:
            for row in rows:
                if row["library"] != ab:
                    continue
                conformer = int(rng.integers(len(library.coordinates)))
                rotamer_count = _local_rotamer_count(library, conformer)
                if rotamer_count <= 0:
                    raise RuntimeError(
                        f"Conformer {conformer} in {ab} has no rotaconformers"
                    )
                row["conformer"] = conformer
                row["rotaconformer"] = int(rng.integers(rotamer_count))
        finally:
            del library
            factory.unload_rotaconformers()
            del factory
            gc.collect()

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
