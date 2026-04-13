# grow (pool → pool): implementation plan

## Goal

A new script that, given a **source pose pool** for fragment N, produces a
**target pose pool** for fragment N+1. Unifies:

- **(1)** [crocodile-OLD/crocodile/main/grow.py](../crocodile-OLD/crocodile/main/grow.py) — defines *what* grow does: cRMSD pre-filter + ovRMSD pose-pose filter, pool in / pool out.
- **(2)** [code/stack.py](../code/stack.py) — new data/IO architecture: `code/library.py` + `code/poses.py` + `PoseStreamAccumulator`, argparse-driven CLI.
- **(3)** [grow-implementation/trace_pose_pose_superposition.py](trace_pose_pose_superposition.py) — new trace-SD algorithm: conformer-SD + rotaconformer-SD (via trace identity) + grid-discretization-SD + grid-translation-SD, with precomputed translation sets and boundary rechecks.
- **(4)** [crocodile-lab/test6.py](../crocodile-lab/test6.py) — scale trick: bucket by the *non-varying* conformer, precompute its rotamer matrix stack once, build `sqq_T = (S @ rot_qq).reshape(9, nRot)`, then inner-loop SD for all rotamer pairs as a single `(|PP|, nRot)` trace block.

## Inputs / outputs

**Inputs**
- `--source-poses POSE_DIR` — directory with `poses-*.npy(.zst)` + `offsets-*.dat` shards (same format as stack.py output; see test fixtures in `tests/1b7f/{1b7f-frag4-scored-and-merged,frag4-fwd-ene-filtered,frag4-bwd-ene-filtered}`).
- `--source-sequence` / `--target-sequence` — dinucleotide sequences.
- `--direction {forward, backward}` — which nucleotide is the overlap.
- `--crmsd FLOAT` — single cRMSD threshold for conformer pre-filter.
- `--ov-rmsd FLOAT` — single ovRMSD threshold for full pose-pose filter.
- `--output POSE_DIR` — output directory (must not exist).
- `--pdb-exclude`, `--max-poses-per-chunk`, `--test-conformers`, `--test-rotamers`, `--test-seed`, `--debug` — same semantics as stack.py.

**Output**
- Target pose pool written via `PoseStreamAccumulator` (zstd chunks), each pose = `(target_conformer u16, target_rotamer u16, translation_int16[3])`.
- Summary line (counts + timings) to stderr.

Single sequence, single direction, single cRMSD, single ovRMSD per run.

## Loop order (IMPORTANT — target outer, source inner)

The outer loop iterates **target conformers T** (in the target library),
*not* source conformers. This matches Codex's recommendation and is
reinforced by the measured data (see [GROW_POOL_CHARACTERIZATION.md](GROW_POOL_CHARACTERIZATION.md)). Rationale:

- Target-rotamer block `rot_qq[T]` is loaded and converted to matrices **once per T**. Target-outer ⇒ ~3.7k–8k loads across the measured pools. Source-outer would do one load per (P,T) pair — **181k–972k loads**, i.e. ~50–150× more rotamer-matrix work.
- The grid-discretization helper `means_Q = mean0_T @ rot_qq` (shape `(nRot_T, 3)`) is shared across **every** source pose that hits T. Measured sources-per-target is mean 8–64, p99 up to 1030.
- The scale trick `sqq_T = einsum("ij,jkn->ikn", S, rot_qq_3x3n).reshape(9, nRot_T)` fuses a per-(P,T) pair `S` with the shared `rot_qq_3x3n`; keeping `rot_qq_3x3n` hot outside the source-conformer loop is what makes this cheap.
- Output is naturally grouped by target conformer → `PoseStreamAccumulator` chunks are locality-friendly.
- `load_rotaconformers()` / `unload_rotaconformers()` are called at most twice per run (once for source, once for target).

Source-conformer outer would invert all four advantages and is rejected. **The measured data makes this more lopsided than test6.py suggested**: test6 projected ~200 unique source conformers, but real pools have 3k–8k, which multiplies source-outer's overhead proportionally.

## Reconciliation table

| Source | What we keep | What we drop |
|---|---|---|
| (1) grow.py `_grow_from_fragment` | Problem shape, `cRMSD`/`ovRMSD` thresholds, pool-in/pool-out contract, idea of source-conformer enumeration and cRMSD masking | `TaskList`, `CandidatePool`, `_load_membership`, `rotamember/*`, `prototype_clusters`, `crocodile_library_config.dinucleotide_libraries`, `_grow_from_anchor` (separate command) |
| (2) stack.py | `argparse` layout, `_run(args)` with `[n/N]` banners, `config()` → `factory.create()` → `load_rotaconformers()`, `PoseStreamAccumulator` usage, `--pdb-exclude`, `--test-*` flags, output dir must-not-exist check | Stacking geometry (`stack_geometry`, ring filters), offset/distance filter logic |
| (3) trace_pose_pose_superposition.py | `TranslationSets`, `_precompute_translation_sets`, `_select_translation_set_index`, `_rotamers_to_matrices`, trace-based rotaconformer SD, boundary rechecks (`TRACE_RMSD_BOUNDARY_TOLERANCE`), grid-discretization + translation sweep, output-writing structure | `Case` dataclass (single reference pose), `pose_pose_samples.csv` plumbing, `_compute_reference_poses` (pool replaces that) |
| (4) test6.py | Flattened `sqq_T` precompute, `rot_pp_flat @ sqq_T` trace block, per-conformer rotamer matrix stacks, rotamer/translation grouping by conformer | `VERIFY` block, brute-force checks, random selection |

## Phase 0 — data characterization (DONE)

Script: [characterize_source_pool.py](characterize_source_pool.py). Results: [GROW_POOL_CHARACTERIZATION.md](GROW_POOL_CHARACTERIZATION.md).

**Key numbers (measured on `tests/1b7f/`):**

| metric                                | fwd pool    | bwd pool      | merged    |
| ------------------------------------- | ----------: | ------------: | --------: |
| poses                                 | 190,697     | 1,752,299     | 18,301    |
| unique source confs                   | 3,397       | 7,521         | 351       |
| unique rot/conf (mean/max)            | 17.8 / 195  | 51.4 / 278    | 9.7 / 71  |
| poses/rotaconformer (mean/max)        | 3.1 / 41    | 4.5 / 117     | 5.4 / 41  |
| translation ptp per axis              | ≤ 30        | ≤ 43          | ≤ 16      |
| sources/target (mean) — forward grow  | 25.8        | 63.8          | 8.1       |
| total (P,T) pairs — forward grow      | 181k        | 506k          | 30k       |
| total (P,T) pairs — backward grow     | 451k        | 972k          | 26k       |

**Findings that changed the plan:**

1. Source conformer diversity is **10–30× higher than test6.py projected** (3–8k unique confs, not ~200). Phase 5 must use numpy group-by, not per-conformer Python loops.
2. Translations per rotaconformer is **3–5, not 10**. Phase 6's per-instance inner loop is small enough that we skip sub-batching and use a single broadcast expansion instead.
3. Translation ranges per axis ≤ 43 — well inside `PoseStreamAccumulator`'s 255-per-axis chunk constraint, so no extra canonical-center bucketing is needed.
4. Sources-per-target mean 8–64 matches test6's ~20 guess within a small factor → the target-outer inner loop is sized as expected.
5. Dev-loop fixture: **1b7f-frag4-scored-and-merged** (18k poses, fastest). Mid-scale: **frag4-fwd-ene-filtered** (190k). Stress: **frag4-bwd-ene-filtered** (1.75M).

**Scale budget (heaviest case — bwd pool, backward grow):**

- 972k (P,T) pairs × ~3M flops each ≈ 3 TFLOP of dense matmul.
- Memory for per-source-conformer caches (rot_pp_flat, centered_P, P_trace, mean_P_rotated) at 7,521 confs × ~200 unique PP × 9×4 B ≈ 54 MB. Plus ~10 MB for the reordered pool table. Negligible.

## Phase 1 — CLI + scaffolding

- Copy argparse shape from [stack.py:72-186](../code/stack.py#L72-L186). Add `--source-poses`, `--source-sequence`, `--target-sequence`, `--direction`, `--crmsd`, `--ov-rmsd`, keep `--output` / `--pdb-exclude` / `--test-*` / `--debug` / `--max-poses-per-chunk`.
- `_run(args)` with `[1/8] … [8/8]` progress banners.
- `main(argv)` wrapper identical in shape to stack.py's, including the debug-vs-error branch.

## Phase 2 — source pool loader

- Iterate shard indices in `--source-poses`. For each shard: `open_pose_array(poses-<i>.npy(.zst))` + `load_offset_table(offsets-<i>.dat)`.
- Decode into a single contiguous table
  ```
  source_table: (N, 5) — [source_conformer u16, source_rotamer u16, tx i16, ty i16, tz i16]
  ```
  N is the full pool size. ~10 B/row ⇒ 10 MB for 1 M poses. In memory, not streamed.
- **Reorder**: stable-sort by `source_conformer`, then by `source_rotamer`. Record
  - `conformer_boundaries`: `np.searchsorted(sorted_conf, unique_source_confs)` for O(1) slicing per conformer.
  - per-conformer `unique_rotamers` + `rotamer_boundaries` inside that slice (also sorted).
  - per `(conformer, rotamer)` instance list (translations only).
- The reorder is the "significant reordering" you flagged. No streaming reader is used beyond the per-shard decode.

## Phase 3 — libraries via new architecture

- Mirror [trace_pose_pose_superposition.py:111-169](trace_pose_pose_superposition.py#L111-L169): bootstrap `sys.path`, build `LibraryDirectory`, `_load_library(directory, templates_dir, sequence, mask)`.
- `direction == "forward"` → `source_mask = (False, True)`, `target_mask = (True, False)`; `"backward"` → swapped.
- Load **source** library first with `with_rotaconformers=True` (we need to resolve source rotamer vectors per source conformer). Decode them into `(3, 3)` matrices and cache per source conformer as they're needed.
- After source precompute is done (Phase 5), `factory.unload_rotaconformers()` on the source library and free its matrices **before** loading the target library, unless `source_sequence == target_sequence` in which case we reuse.

## Phase 4 — cRMSD pivot

- `from library import load_crmsds; crmsds = load_crmsds(ab, bc)` — match the forward/backward orientation used in [trace_pose_pose_superposition.py:351-354](trace_pose_pose_superposition.py#L351-L354). Forward uses rows, backward uses columns.
- `crmsd_ok = crmsds < args.crmsd` restricted to rows/cols for source conformers present in the pool.
- Build two structures:
  - `source_to_targets: {P → np.ndarray of T}` — used for bookkeeping only.
  - `target_to_sources: {T → np.ndarray of P}` — **this is the pivot that drives the outer loop.** Built once from `crmsd_ok.T` (or `.T` depending on direction).
- `target_conformer_list = sorted(target_to_sources.keys())`.

## Phase 5 — per-source-conformer precomputes

For every source conformer P actually present in the pool. **Critical**: with 3k–8k unique source conformers (measured, 10–30× more than test6 projected), per-conformer Python-level bookkeeping is expensive. Everything here must be numpy group-by driven from the sorted pool table built in Phase 2.

Approach:

- Start from the Phase-2 pool sorted by `(source_conformer, source_rotamer)` with per-conformer slice boundaries `(conf_start[P], conf_stop[P])`.
- **Single pass** over unique source conformers using numpy slicing (no Python loop over instances):
  - `coords_P = source_lib.coordinates[P]` (float32), `mean_P0 = coords_P.mean(0)`, `centered_P = coords_P - mean_P0`, `P_trace = einsum("ij,ij->", centered_P, centered_P)`.
  - Slice the sorted pool for this conformer, call `np.unique(rotamer_slice, return_inverse=True, return_counts=True)` → `unique_PP[P]`, `pp_local_index_per_instance[P]`, `counts_per_unique_PP[P]`.
  - `rot_pp_sel[P] = Rotation.from_rotvec(get_rotamers(P)[unique_PP[P]]).as_matrix().astype(float32)` → `(n_unique_PP, 3, 3)`.
  - `rot_pp_flat[P] = rot_pp_sel[P].reshape(n_unique_PP, 9)` — this is what feeds the trace block.
  - `mean_P_rotated[P] = einsum("j,njk->nk", mean_P0, rot_pp_sel[P])` → `(n_unique_PP, 3)`. Centered mean after rotation; combined with per-instance translation at inner-loop time.
  - Keep the per-instance translations as a contiguous `(n_instances_for_P, 3)` int16 slice plus the `pp_local_index_per_instance[P]` mapping back to `unique_PP[P]`.
- Drop `coords_P` / `rot_pp_sel[P]`; keep `centered_P`, `P_trace`, `rot_pp_flat[P]`, `mean_P_rotated[P]`, `unique_PP[P]`, `counts_per_unique_PP[P]`, and the per-instance translation slice.
- Once all P are precomputed (and before loading target library), call `source_factory.unload_rotaconformers()`.

Memory budget (measured, stress case = bwd pool):

- 7,521 source conformers × max 278 unique PP × 9 × 4 B ≈ **54 MB** for `rot_pp_flat`.
- Per-conformer `centered_P` + `mean_P_rotated`: <10 MB total.
- The sorted pool table: 1.75M poses × 10 B ≈ **18 MB**.
- Total Phase-5 cache footprint: under 100 MB.

## Phase 6 — main loop (target outer, source inner)

Load the **target** library with rotaconformers. Precompute `translation_sets = _precompute_translation_sets()` (verbatim from trace impl). Precompute `total_overlap_sd = natoms * ovRMSD**2` and `max_grid_discretization_sd = natoms * translation_sets.max_grid_discretization_sd_per_atom`.

Pseudocode:

```python
writer = PoseStreamAccumulator(outdir, zstd=True, max_poses_per_chunk=...)

for T in target_conformer_list:                    # OUTER: target conformer
    coords_T = target_lib.coordinates[T].astype(f32)
    mean_T0 = coords_T.mean(0)
    centered_T = coords_T - mean_T0
    T_trace = einsum("ij,ij->", centered_T, centered_T)

    rot_qq = as_matrices(target_lib.get_rotamers(T))          # (nRot_T, 3, 3)
    rot_qq_3x3n = ascontiguous(transpose(rot_qq, (1, 2, 0)))  # (3, 3, nRot_T)
    means_Q = mean_T0 @ rot_qq                                # (nRot_T, 3)

    for P in target_to_sources[T]:                  # INNER 1: source conformer
        S = centered_P[P].T @ centered_T                     # (3, 3)
        sqq_T = einsum("ij,jkn->ikn", S, rot_qq_3x3n).reshape(9, nRot_T)

        PP_flat = rot_pp_flat[P]                             # (n_unique_PP, 9)
        PP_mean = mean_P_rotated[P]                          # (n_unique_PP, 3)

        # Rotaconformer SD via trace identity — one dense matmul per (P,T)
        trace_block = PP_flat @ sqq_T                        # (n_unique_PP, nRot_T)
        rc_sd = P_trace[P] + T_trace - 2.0 * trace_block     # (n_unique_PP, nRot_T)
        rc_sd = maximum(rc_sd, 0.0)

        pp_keep_row, qq_keep_col = nonzero(rc_sd < total_overlap_sd)
        if pp_keep_row.size == 0: continue

        rc_kept = rc_sd[pp_keep_row, qq_keep_col]            # (K,)

        # Vectorized "expand over instances": replicate each surviving (pp, qq)
        # pair into len(instances_of_pp) rows.
        inst_off = instance_offsets_for_P[pp_keep_row]       # (K, 2): start, stop
        inst_counts = inst_off[:, 1] - inst_off[:, 0]        # (K,)
        inst_total = inst_counts.sum()
        if inst_total == 0: continue

        repeat_idx = repeat(arange(K), inst_counts)          # (inst_total,)
        pp_rows_exp = pp_keep_row[repeat_idx]                # (inst_total,)
        qq_cols_exp = qq_keep_col[repeat_idx]                # (inst_total,)
        rc_exp      = rc_kept    [repeat_idx]                # (inst_total,)

        # Gather per-instance translations
        inst_translations = instance_translations_for_P[     # (inst_total, 3) int16
            cumulative_offset_map[repeat_idx] + within_group_index
        ]

        world_mean = (
            PP_mean[pp_rows_exp]
            + inst_translations.astype(f32) * GRID_SPACING   # (inst_total, 3)
        )
        continuous_translation = world_mean - means_Q[qq_cols_exp]
        best_grid = rint(continuous_translation / GRID_SPACING).astype(i32)
        delta = best_grid.astype(f32) * GRID_SPACING - continuous_translation
        grid_disc_sd = natoms * einsum("ij,ij->i", delta, delta)

        keep_gd = nonzero(grid_disc_sd + rc_exp < total_overlap_sd)
        if keep_gd.size == 0: continue

        # remaining2/3/4 and translation-set sweep + boundary recheck
        # — structurally identical to trace_pose_pose_superposition.py:427-549,
        #   but operating on the expanded (inst_total_kept,) axis instead of
        #   a per-rotamer-batch (rotamer_block,) axis.
        remaining2 = total_overlap_sd - grid_disc_sd[keep_gd]
        remaining3 = remaining2 - rc_exp[keep_gd]
        remaining4 = total_overlap_sd - rc_exp[keep_gd]
        set_idx = _select_translation_set_index(translation_sets, remaining3, natoms)

        # group by set_idx, apply offsets, compute translation_sd, boundary recheck,
        # then for each kept point:
        #   absolute_translation_int16 = inst_translations[kept] - best_grid[kept] + offset
        #   (inverted sign convention because best_grid goes FROM source TO target;
        #    the absolute target translation is source.trans + offset_from_source_to_target)
        writer.add_chunk(
            conformer_indices = T_as_uint16_array_of_len_kept,
            rotamer_indices   = qq_cols_exp[kept_after_all_filters].astype(u16),
            translations      = absolute_int16_translations,
        )

writer.finish()
```

Key points:

- `sqq_T` is per-(P,T); computed once per source conformer, not per source pose or per rotamer batch.
- **No PP sub-batching.** Measured `n_unique_PP ≤ 278` per source conformer — a single `PP_flat @ sqq_T` matmul is the right shape. The `ROTAMER_BATCH_SIZE = 4096` split from trace impl would be inert at this scale, and skipping it simplifies the code.
- **No per-(pp,qq) Python loop over instances.** With measured 3–5 instances per rotaconformer, sub-batching instances is pure overhead. Expand once via `np.repeat(..., inst_counts)` and vectorize the translation-set sweep across the flat `inst_total` axis.
- `rc_sd` depends only on (P, PP, T, QQ), not on translations → rotaconformer prune happens *before* instance expansion, keeping the expanded array short.
- The boundary recheck logic from [trace_pose_pose_superposition.py:434-549](trace_pose_pose_superposition.py#L434-L549) transfers, adapted to operate on the flat `(inst_total_kept,)` axis. The two tolerances (`RMSD_BOUNDARY_TOLERANCE`, `TRACE_RMSD_BOUNDARY_TOLERANCE`) stay.
- Output: every accepted (T, QQ, absolute translation) row goes through `writer.add_chunk` in target-conformer order → good locality for `PoseStreamAccumulator` chunking.

## Phase 7 — validation

- **Pool-of-one equivalence**: add a `--single-pose` debug mode that accepts one source pose (conformer + rotaconformer + zero translation) and run it on the `grow-implementation/pose_pose_samples.csv` cases. Diff the resulting pose dir against `trace_pose_dirs_*` within boundary tolerance. This is the golden test — if we break trace-impl equivalence we abort.
- **Scale smoke test**: run on each of the three `tests/1b7f/` pools with `--test-conformers` / `--test-rotamers` to keep wall time bounded, then compare counts against (1)'s output if available, or sanity-check against a small brute force on a handful of target conformers.
- **Determinism**: running twice produces byte-identical pose files.

## Constants + tolerances (reuse verbatim from trace impl)

```
GRID_SPACING = sqrt(3) / 3
MAX_OVERLAP_RMSD = 1.3
ROTAMER_BATCH_SIZE = 4096
RMSD_BOUNDARY_TOLERANCE = 2e-6
TRACE_RMSD_BOUNDARY_TOLERANCE = 1e-5
```

## Open questions (resolved)

1. **Pose reader**: `open_pose_array` + `load_offset_table` exist in `code/poses.py` ([convert_poses.py:149](../code/convert_poses.py#L149)). The "significant reordering" is done in Phase 2 in memory — no streaming reader is added to `poses.py`.
2. **Pool format**: guaranteed `poses-*.npy[.zst]` + `offsets-*.dat`. Three concrete test pools in `tests/1b7f/`.
3. **cRMSD direction**: single direction per run, matches trace impl's `forward`/`backward` row/column choice.
4. **Threshold shape**: single `cRMSD` + single `ovRMSD` scalar, no per-pair constraint map.

## Dropped (won't be in the new script)

- Everything under `crocodile-OLD/crocodile/main/` — `TaskList`, `CandidatePool`, `tensorlib`, `_load_membership`, prototype clustering, `crocodile_library_config`, anchor mode.
- `Case` dataclass plumbing from trace impl.
- `pose_pose_samples.csv` — only used as a validation fixture in Phase 7, not for production runs.
