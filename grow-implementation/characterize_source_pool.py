#!/usr/bin/env python3
"""Data characterization for grow pool-to-pool implementation."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from poses import open_pose_array, load_offset_table, discover_pose_pairs  # noqa: E402


def load_pool(pool_dir: Path) -> np.ndarray:
    rows = []
    for pose_path, offsets_path in discover_pose_pairs(pool_dir):
        poses = np.asarray(open_pose_array(pose_path), dtype=np.uint16)
        offsets = load_offset_table(offsets_path)
        conf = poses[:, 0].astype(np.int32)
        rot = poses[:, 1].astype(np.int32)
        off_idx = poses[:, 2].astype(np.int64)
        translations = offsets[off_idx].astype(np.int16)
        rows.append(
            np.column_stack(
                (conf, rot, translations.astype(np.int32))
            ).astype(np.int32)
        )
    if not rows:
        return np.zeros((0, 5), dtype=np.int32)
    return np.concatenate(rows, axis=0)


def pct(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def characterize_pool(name: str, pool_dir: Path) -> dict:
    table = load_pool(pool_dir)
    n = len(table)
    conf = table[:, 0]
    rot = table[:, 1]
    trans = table[:, 2:5].astype(np.int16)

    unique_conf = np.unique(conf)
    cr_pack = conf.astype(np.uint64) * (np.iinfo(np.uint16).max + 1) + rot.astype(
        np.uint64
    )
    unique_rc, counts_rc = np.unique(cr_pack, return_counts=True)

    conf_sort = np.argsort(conf, kind="stable")
    conf_sorted = conf[conf_sort]
    rot_sorted = rot[conf_sort]
    boundaries = np.flatnonzero(np.diff(conf_sorted)) + 1
    group_starts = np.concatenate(([0], boundaries))
    group_stops = np.concatenate((boundaries, [n]))
    poses_per_conf = (group_stops - group_starts).astype(np.int64)

    unique_rot_per_conf = []
    for s, e in zip(group_starts, group_stops):
        unique_rot_per_conf.append(len(np.unique(rot_sorted[s:e])))
    unique_rot_per_conf = np.array(unique_rot_per_conf, dtype=np.int64)

    poses_per_rc = counts_rc.astype(np.int64)
    trans_ptp = trans.max(axis=0).astype(np.int32) - trans.min(axis=0).astype(np.int32)

    result = {
        "name": name,
        "n_poses": int(n),
        "n_unique_conformers": int(len(unique_conf)),
        "n_unique_rotaconformers": int(len(unique_rc)),
        "poses_per_conformer": {
            "p50": pct(poses_per_conf, 50),
            "p90": pct(poses_per_conf, 90),
            "p99": pct(poses_per_conf, 99),
            "max": int(poses_per_conf.max()),
            "min": int(poses_per_conf.min()),
            "mean": float(poses_per_conf.mean()),
        },
        "unique_rotamers_per_conformer": {
            "p50": pct(unique_rot_per_conf, 50),
            "p90": pct(unique_rot_per_conf, 90),
            "p99": pct(unique_rot_per_conf, 99),
            "max": int(unique_rot_per_conf.max()),
            "min": int(unique_rot_per_conf.min()),
            "mean": float(unique_rot_per_conf.mean()),
        },
        "poses_per_rotaconformer": {
            "p50": pct(poses_per_rc, 50),
            "p90": pct(poses_per_rc, 90),
            "p99": pct(poses_per_rc, 99),
            "max": int(poses_per_rc.max()),
            "min": int(poses_per_rc.min()),
            "mean": float(poses_per_rc.mean()),
        },
        "translation_range_int16": {
            "x": (int(trans[:, 0].min()), int(trans[:, 0].max()), int(trans_ptp[0])),
            "y": (int(trans[:, 1].min()), int(trans[:, 1].max()), int(trans_ptp[1])),
            "z": (int(trans[:, 2].min()), int(trans[:, 2].max()), int(trans_ptp[2])),
        },
        "unique_conformers": unique_conf,
    }
    return result


def characterize_crmsd(
    unique_conformers_by_pool: dict[str, np.ndarray],
    ab: str,
    bc: str,
    direction: str,
    crmsd_thresh: float,
) -> dict:
    from library import load_crmsds

    crmsds = load_crmsds(ab, bc, pdb_code="1B7F")
    if direction == "forward":
        mat = crmsds < crmsd_thresh
        get_row = lambda P: mat[P, :]
    else:
        mat = crmsds < crmsd_thresh
        get_row = lambda P: mat[:, P]

    out = {
        "ab": ab,
        "bc": bc,
        "direction": direction,
        "crmsd_thresh": crmsd_thresh,
        "crmsd_shape": tuple(int(v) for v in crmsds.shape),
    }
    for pool_name, unique_conf in unique_conformers_by_pool.items():
        targets_per_source = []
        all_targets = set()
        for P in unique_conf.tolist():
            row = get_row(int(P))
            targets = np.where(row)[0]
            targets_per_source.append(len(targets))
            all_targets.update(int(t) for t in targets)
        tps = np.array(targets_per_source, dtype=np.int64)

        # pivot T -> sources: count how many source conformers per target
        target_to_sources: dict[int, list[int]] = {}
        for P in unique_conf.tolist():
            row = get_row(int(P))
            for T in np.where(row)[0].tolist():
                target_to_sources.setdefault(int(T), []).append(int(P))
        sources_per_target = np.array(
            [len(v) for v in target_to_sources.values()], dtype=np.int64
        )
        out[pool_name] = {
            "n_source_conformers_in_pool": int(len(unique_conf)),
            "n_target_conformers_hit": int(len(all_targets)),
            "target_fraction_of_library": (
                float(len(all_targets)) / crmsds.shape[1]
                if direction == "forward"
                else float(len(all_targets)) / crmsds.shape[0]
            ),
            "targets_per_source": {
                "p50": pct(tps, 50),
                "p90": pct(tps, 90),
                "p99": pct(tps, 99),
                "max": int(tps.max()) if tps.size else 0,
                "min": int(tps.min()) if tps.size else 0,
                "mean": float(tps.mean()) if tps.size else 0.0,
                "total": int(tps.sum()),
            },
            "sources_per_target": {
                "p50": pct(sources_per_target, 50),
                "p90": pct(sources_per_target, 90),
                "p99": pct(sources_per_target, 99),
                "max": int(sources_per_target.max()) if sources_per_target.size else 0,
                "min": int(sources_per_target.min()) if sources_per_target.size else 0,
                "mean": float(sources_per_target.mean())
                if sources_per_target.size
                else 0.0,
                "total": int(sources_per_target.sum()),
            },
        }
    return out


def print_table(row: dict) -> None:
    print(f"\n=== {row['name']} ===")
    print(f"  poses                    : {row['n_poses']:,}")
    print(f"  unique conformers        : {row['n_unique_conformers']:,}")
    print(f"  unique rotaconformers    : {row['n_unique_rotaconformers']:,}")
    for key in (
        "poses_per_conformer",
        "unique_rotamers_per_conformer",
        "poses_per_rotaconformer",
    ):
        s = row[key]
        print(
            f"  {key:<28}: mean={s['mean']:.1f}  p50={s['p50']:.0f}  p90={s['p90']:.0f}  p99={s['p99']:.0f}  max={s['max']}"
        )
    tr = row["translation_range_int16"]
    print(
        f"  translation range        : x={tr['x']}  y={tr['y']}  z={tr['z']}"
    )


def print_crmsd(row: dict) -> None:
    print(
        f"\n=== cRMSD pivot  ab={row['ab']} bc={row['bc']} dir={row['direction']} thresh={row['crmsd_thresh']} ==="
    )
    print(f"  crmsds matrix shape      : {row['crmsd_shape']}")
    for k, v in row.items():
        if not isinstance(v, dict):
            continue
        if k in ("targets_per_source", "sources_per_target"):
            continue
        print(f"\n  pool: {k}")
        print(
            f"    source confs in pool   : {v['n_source_conformers_in_pool']:,}"
        )
        print(
            f"    target confs hit       : {v['n_target_conformers_hit']:,}  "
            f"({100*v['target_fraction_of_library']:.1f}% of target lib)"
        )
        tps = v["targets_per_source"]
        print(
            f"    targets per source     : mean={tps['mean']:.1f}  p50={tps['p50']:.0f}  p90={tps['p90']:.0f}  p99={tps['p99']:.0f}  max={tps['max']}  total pairs={tps['total']:,}"
        )
        spt = v["sources_per_target"]
        print(
            f"    sources per target     : mean={spt['mean']:.1f}  p50={spt['p50']:.0f}  p90={spt['p90']:.0f}  p99={spt['p99']:.0f}  max={spt['max']}  total pairs={spt['total']:,}"
        )


if __name__ == "__main__":
    pool_root = REPO_ROOT / "tests" / "1b7f"
    pools = {
        "frag4-fwd-ene-filtered": pool_root / "frag4-fwd-ene-filtered",
        "frag4-bwd-ene-filtered": pool_root / "frag4-bwd-ene-filtered",
        "1b7f-frag4-scored-and-merged": pool_root / "1b7f-frag4-scored-and-merged",
    }
    rows = {}
    for name, pdir in pools.items():
        rows[name] = characterize_pool(name, pdir)
        print_table(rows[name])

    unique_by_pool = {n: r["unique_conformers"] for n, r in rows.items()}
    # frag4 -> frag5: ab=GU (frag4), bc=UU (frag5), source_mask=(False,True) overlap,
    # direction forward, cRMSD threshold from constraints.json pair frag4->frag5 = 0.277
    print("\n\n# cRMSD pivot — frag4(GU) → frag5(UU), forward, cRMSD<0.277")
    crmsd_row_fwd = characterize_crmsd(
        unique_by_pool, ab="GU", bc="UU", direction="forward", crmsd_thresh=0.277
    )
    print_crmsd(crmsd_row_fwd)

    # frag4 <- frag3: ab=UG (frag3), bc=GU (frag4), backward, cRMSD for frag3->frag4 = 0.381
    print("\n\n# cRMSD pivot — frag3(UG) → frag4(GU), backward from frag4, cRMSD<0.381")
    crmsd_row_bwd = characterize_crmsd(
        unique_by_pool, ab="UG", bc="GU", direction="backward", crmsd_thresh=0.381
    )
    print_crmsd(crmsd_row_bwd)
