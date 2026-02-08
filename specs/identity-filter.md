Identical-pose filtering in Julia (current implementation in `code/filter_identical_poses.jl`).

## Input Model
- A pose pool is a directory of pairs: `poses-X.npy` or `poses-X.npy.zst` with `offsets-X.dat`.
- Poses are `uint16[*,3]`: `[conformer, rotamer, offset_index]`.
- Offsets are center-encoded: `center::Int16[3]` plus per-entry relative `Int8[3]`.

## Outputs
- Unique intersection pose pool `K` as `poses-*.npy.zst` + `offsets-*.dat`.
- `ind-A-*.npy`: two columns `[A_global_index, K_unique_index]`.
- `ind-B-*.npy`: two columns `[B_global_index, K_unique_index]`.
- Index dtype is `uint32` when safe, otherwise `uint64`.

## Canonical Centering
- Canonical centers are multiples of 256 on each axis.
- Each absolute offset belongs to exactly one canonical center.
- Canonical pair blocks can be reused in place; non-canonical pair blocks are repartitioned by canonical center.

## Full Run Algorithm
1. Discover both inputs and pick largest as `A`, smallest as `B` by total on-disk pose-file bytes.
2. Global conf-rot prefilter (mandatory):
   - Build shared `(conf,rot)` bitset from raw `A` and `B`.
3. Canonicalization/indexing for `A` and `B`:
   - Canonical-center pairs are referenced directly.
   - Non-canonical pairs are rebased per pose to canonical center and written to compressed per-center temp buckets.
   - The shared conf-rot mask is applied while building canonical buckets.
4. Build unique `K` and lookup:
   - For each center present in both sets, load center keys from both sides.
   - Sort+unique keys on each side, then intersect.
   - Important: this is set-style unique intersection per center key, not multiplicity-preserving.
   - Write resulting unique keys to `K` (`poses-*.npy.zst`, `offsets-*.dat`) and write per-center lookup records `(key -> uid)` as compressed files.
5. Map stage:
   - `map-a`: map every original `A` occurrence that resolves to a lookup key into `ind-A`.
   - `map-b`: same for `B` into `ind-B`.
   - This is where duplicates in original sets are preserved (multiple global indices can map to one `K_unique_index`).
6. Cleanup temp under `<output>/_tmp` unless `--keep-temp`.

## Reuse Mode
- With `--reuse-unique-k-dir <dir>`:
1. Skip steps 2-4 above.
2. Rebuild per-center lookup from existing unique-K pose files in `<dir>`.
3. Build a map conf-rot mask from lookup files.
4. Map `A` and `B` directly against lookup via absolute `Int16` offsets and conf-rot prefilter.

## CLI Defaults (current)
- `--threshold-k 2000000`
- `--threshold-m 200000`
- `--max-poses-per-chunk 4294967295`
- `--dedup-buckets 256` (legacy/unused in active path)
- `--phase1-zstd-threads` defaults to `CROCODILE_ZSTD_THREADS` (or `0` if unset).
- `--map-zstd-threads` defaults to `1`.
- `--map-workers 0` means auto no-HT equivalent worker count.
- `--max-poses-prefilter 5000000000` (`0` disables)
- `--max-poses-final 50000000` (`0` disables)

## Notes
- `.npy.zst` data is decompressed in memory; no decompressed `.npy` temp files are written.
- Temp key/index buckets and lookup intermediates are written compressed (`.zst`) in the active path.
- Optional runtime env knobs still used by active code:
  - `CROCODILE_ZSTD_THREADS`
  - `CROCODILE_ZSTD_SINGLE_THREAD_BELOW_BYTES`
  - `CROCODILE_PRINT_SIZE_STATS`
- Unique-K pass2 inner parallelization thresholds are fixed in code:
  - min rows per center: `2_000_000`
  - pass2 chunk rows: `1_000_000`
- Panic guards:
  - after canonicalization, abort if post-confrot rows exceed `--max-poses-prefilter`
  - before map-a/map-b, abort if unique-K rows exceed `--max-poses-final`
