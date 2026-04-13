# Phase 0 — source pool characterization (results)

Data from `tests/1b7f/{frag4-fwd-ene-filtered, frag4-bwd-ene-filtered, 1b7f-frag4-scored-and-merged}`, decoded via `open_pose_array` + `load_offset_table`. cRMSD matrices loaded via `library.load_crmsds(pdb_code="1B7F")`. Thresholds taken from `tests/1b7f/data/constraints.json` pairs. Script: [characterize_source_pool.py](characterize_source_pool.py).

## Pool sizes

| pool | poses | unique confs | unique rotaconformers | poses/conf (mean/p90/max) | unique rot/conf (mean/p90/max) | poses/rotaconformer (mean/p90/max) | translation ptp (x,y,z) |
|---|---:|---:|---:|---|---|---|---|
| frag4-fwd-ene-filtered | 190,697 | 3,397 | 60,607 | 56 / 144 / 1397 | 17.8 / 50 / 195 | 3.1 / 7 / 41 | 16, 26, 30 |
| frag4-bwd-ene-filtered | 1,752,299 | 7,521 | 386,885 | 233 / 658 / 3719 | 51.4 / 122 / 278 | 4.5 / 11 / 117 | 33, 43, 30 |
| 1b7f-frag4-scored-and-merged | 18,301 | 351 | 3,415 | 52 / 144 / 753 | 9.7 / 25 / 71 | 5.4 / 13 / 41 | 13, 16, 16 |

## cRMSD pivot — forward grow frag4(GU) → frag5(UU), cRMSD<0.277

Target library: `crmsds.shape=(7740, 9013)`, so target-side = 9013 conformers.

| pool | src confs | target confs hit | targets/src (mean/p90/max) | **sources/target (mean/p90/max)** | total (P,T) pairs |
|---|---:|---:|---|---|---:|
| frag4-fwd-ene-filtered | 3,397 | 7,038 (78.1%) | 53.5 / 132 / 908 | **25.8 / 93 / 163** | 181,813 |
| frag4-bwd-ene-filtered | 7,521 | 7,943 (88.1%) | 67.3 / 173 / 908 | **63.8 / 241 / 488** | 506,373 |
| 1b7f-frag4-scored-and-merged | 351 | 3,709 (41.2%) | 86.0 / 211 / 804 | **8.1 / 22 / 36** | 30,184 |

## cRMSD pivot — backward grow frag3(UG) ← frag4(GU), cRMSD<0.381

`crmsds.shape=(8440, 7740)`. Looser threshold ⇒ higher fan-out.

| pool | src confs | target confs hit | targets/src (mean/p90/max) | sources/target (mean/p90/max) | total (P,T) pairs |
|---|---:|---:|---|---|---:|
| frag4-fwd-ene-filtered | 3,397 | 7,357 (87.2%) | 132.8 / 511 / 758 | 61.3 / 199 / 554 | 451,051 |
| frag4-bwd-ene-filtered | 7,521 | 7,951 (94.2%) | 129.3 / 498 / 758 | 122.3 / 374 / 1139 | 972,458 |
| 1b7f-frag4-scored-and-merged | 351 | 3,984 (47.2%) | 73.8 / 167 / 714 | 6.5 / 17 / 42 | 25,906 |

## How this compares to test6.py's projection

test6.py projected (lines 34-46):
- ~200 source conformers × ~50 rotamers × ~10 translations ⇒ 100k poses
- ~10k target conformers, ~10% cRMSD fan-out ⇒ ~1000 targets per source
- final pivot: ~10k targets × **~20 sources-per-target**

Measured:

| quantity | test6 guess | fwd pool | bwd pool | merged |
|---|---:|---:|---:|---:|
| unique source confs | 200 | 3,397 | 7,521 | 351 |
| unique rotamers/conf | 50 | 17.8 (mean) | 51.4 (mean) ✓ | 9.7 |
| translations/rotaconformer | 10 | 3.1 | 4.5 | 5.4 |
| total source poses | 100k | 190k | 1.75M | 18k |
| target confs hit | ~10k | 7,038 | 7,943 | 3,709 |
| **sources per target** (the key pivot number) | **~20** | **25.8** | **63.8** | 8.1 |

Observations:
- **test6 under-estimated source-conformer diversity by 10–30×.** Real pools have thousands of distinct source conformers, not ~200. This *hurts* source-outer proportionally — it means source-outer would reload target-library data on many more (P,T) pairs.
- **Per-source-conformer rotamer counts match bwd pool closely** (51 vs 50). fwd pool is leaner (18 mean, 50 p90).
- **Translations per rotaconformer are lower than guessed** (3–5 vs 10). The per-instance inner expansion is cheaper than Phase 6 planned for.
- **Sources-per-target — the quantity that drives the inner loop of the target-outer design — matches test6's order-of-magnitude guess**: 8-64 mean depending on pool, max ≤1139 in the heaviest case. Test6's 20 is right in the middle.

## Does this line up with my loop-order choice?

**Yes. Target-outer is the right call, and the data reinforces it.** The argument:

- Target library rotamer state (`rot_qq[T]`, matrices, `rot_qq_3x3n`) is loaded per target conformer. Target-outer ⇒ ~3.7k–8k loads. Source-outer ⇒ 181k–972k loads (one per (P,T) pair). **~50–150× saving.**
- `means_Q = mean0_T @ rot_qq` is the per-target grid-discretization lookup. It's reused for every source pose in the T-bucket — on average 8–64 source conformers × each conformer's ~3–5 poses/rotaconformer × ~18–51 unique rotamers. Amortized across thousands of rotaconformer-instance trials per T.
- `sqq_T = einsum("ij,jkn->ikn", S, rot_qq_3x3n)` is per (P,T) pair — **not** shared across sources within a T. But the input `rot_qq_3x3n` being hot is what makes the einsum fast; that's target-outer's gift.
- The cost of per-instance translation expansion is **lower than expected** (3–5 instances per surviving rotaconformer). Phase 6's pseudocode still works but the "expand over instances" inner block is smaller than I'd planned. I will simplify it (no need to sub-batch).
- Scale: the heaviest case is `frag4-bwd-ene-filtered → backward grow` = 972k (P,T) pairs. At ~3M flops per pair (trace block + sqq_T build) that's ~3 TFLOP of dense matmul — minutes, not hours.

## Surprises / adjustments to the plan

1. **Source conformer diversity is much higher than test6 suggested.** Phase 5's per-source-conformer precompute has to build caches for thousands of conformers, not hundreds. Still cheap (< 100 MB for the largest pool: 7521 confs × ~200 unique PP × 9×4 B = 54 MB for rot_pp_flat, plus centered_P ~ negligible), but the bookkeeping must be efficient (no per-conformer Python overhead — use numpy group-by on the sorted table).
2. **Per-instance expansion is cheap enough to inline.** Phase 6's inner `for (pp_row, qq_col)` loop can be rewritten as a single vectorized step: after rotaconformer pruning, broadcast per-instance translations against kept `(pp, qq)` pairs, then run the existing translation-set sweep. Keeps the code closer to trace impl.
3. **Translation ranges are tight** (ptp ≤ 43 per axis). Well inside `PoseStreamAccumulator`'s ≤255 per-axis chunk constraint. No extra canonical-center bucketing needed.
4. **Large p99/max fan-out in targets_per_source (e.g. max 908 target conformers for a single source)** means the source-outer memory working-set would also be big. Target-outer has no such issue because each target iteration only sees its own `sources_per_target` list (max 1139 in the extreme bwd case — still fine).
5. **1b7f-frag4-scored-and-merged** is small enough (351 conformers, 18k poses, 30k (P,T) pairs for forward grow) to be our fast dev-loop fixture. **frag4-fwd-ene-filtered** is the mid-scale realistic test. **frag4-bwd-ene-filtered** is the stress test.

## Next step

Proceed to Phase 1 (CLI scaffolding). No changes to the loop-order decision. One small refinement to Phase 6: don't split the inner instance loop — keep the vectorized `(PP × QQ)` block, then vectorize-expand to instances via broadcasting.
