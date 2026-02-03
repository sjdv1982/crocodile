# Stage A — Simulated Poses for Pose Reader/Writer

**Summary**
Implement a Stage A test helper that simulates a realistic set of poses using the existing fragment libraries and utilities (`library.py`, `superimpose.py`, `offsets.py`). The simulation will generate rotamer selections, compatible conformer pairs, and discretized translations (offsets) for 20×1×10×10 combinations, without enforcing non-empty offset sets. No round-trip tests or file I/O are included in Stage A; those will be introduced in Stage B.

## Scope
- Add a new test module (e.g. `tests/test_poses.py`) that performs the Stage A simulation.
- Reuse existing library loading, superposition, and offsets machinery.
- No pose reader/writer yet; no file format I/O.

## Interfaces and Data Flow
- **Library loading**: use `library.config(verify_checksums=False)` to obtain `dinucleotide_libraries`.
- **AA selection**:
  - `libf_aa = dinucleotide_libraries["AA"]`
  - `libf_aa.load_rotaconformers()`
  - `lib_aa = libf_aa.create(pdb_code=None, nucleotide_mask=[False, True], with_rotaconformers=True)`
- **AC selection**:
  - after AA rotamer selection, call `libf_aa.unload_rotaconformers()` to reduce memory
  - `libf_ac = dinucleotide_libraries["AC"]`
  - `libf_ac.load_rotaconformers()`
  - `lib_ac = libf_ac.create(pdb_code=None, nucleotide_mask=[True, False], with_rotaconformers=True)`

- **Conformer/rotamer selection**:
  - Randomly select 20 distinct conformers from `lib_aa.coordinates`.
  - For each conformer:
    - Randomly sample rotamers (with replacement) until one is found whose COM is within 3 Å of the reference rotamer (the rotamer chosen for the first conformer). Store rotation matrix and COM.
  - For each selected AA conformer:
    - Determine compatible AC conformers: RMSD < 0.5 Å after superposition using `superimpose`.
    - Randomly sample 10 compatible conformers (with replacement if fewer than 10).
    - For each compatible AC conformer:
      - Randomly sample 10 rotamers; accept if COM within 3 Å of the first chosen AC rotamer for that conformer. Store rotation matrices and COMs.

- **COM definition**
  - COM of rotated coordinates for the selected nucleotide only (via `nucleotide_mask`).
  - Apply rotamer rotation to the conformer coordinates prior to COM: `coords_rot = coords.dot(rotmat)`.

- **Displacements and radii**
  - For each combination (20×1×10×10):
    - `displacement = com_aa - com_ac`
    - `radius = uniform(-0.2, 1.2)`
    - **Grid scaling**: divide both `displacement` and `radius` by `(sqrt(3)/3)`.
  - Call `offsets.get_discrete_offsets(displacements_scaled, radii_scaled)`.
  - Allow empty offset sets; no further checks or assertions on non-emptiness.

- **Determinism**
  - Set `np.random.seed(<fixed seed>)` to make the simulation reproducible.

## Implementation Details
- New file: `tests/test_poses.py`
- Helper functions inside the test module:
  - `pick_rotamer_with_com_constraint(rotamers, coords, ref_com, max_dist=3.0)`
  - `compatible_conformers(coords_a, coords_b, rmsd_cutoff=0.5)`
- Use `superimpose` (single-pair) for RMSD; performance is not a constraint.

## Public API / Interface Changes
- None. Only adds a test module.

## Tests and Scenarios
- `tests/test_poses.py` runs the simulation end-to-end without assertions on size or distribution, and without I/O.
- Assertions:
  - Correct shapes for selected rotamer matrices and COM arrays.
  - The number of generated displacement vectors equals `20 * 1 * 10 * 10`.

## Assumptions and Defaults
- `~/.crocodile/fraglib.yaml` exists and is valid on the target machine.
- Atom counts are already verified to match across AA/AC selections.
- Empty offset sets are allowed; no clamp on negative radii.
- Stage B will add round‑trip checks and pose reader/writer I/O.

# Stage B — Pose Pack/Unpack Round‑Trip (Deduped Offsets)

**Summary**
Implement `pack_all_poses` and `unpack_poses` in `code/poses.py`, then extend `tests/test_poses.py` to perform a full round‑trip check. Offsets are deduplicated and stored as an offset table plus indices in `poses.npy`. The round‑trip test reconstructs translations via lookup and compares to Stage A’s gathered translations.

## Scope
- Add `code/poses.py` with packing/unpacking helpers for the pose format.
- Extend `tests/test_poses.py` to use these helpers and verify round‑trip correctness.
- Implement full split logic per architecture constraints.

## Interfaces and Data Flow
- **Inputs** to `pack_all_poses`:
  - `conformer_indices`: array of pose conformer indices (expanded per translation).
  - `rotamer_indices`: array of pose rotamer indices (expanded per translation).
  - `offsets_tuple`: the 4‑tuple `(disp_indices, p_indices, rounded, reverse_map)` from `get_discrete_offsets`.

- **Outputs** from `pack_all_poses`:
  - list of `(poses, mean_offset, offsets)` tuples.
    - `poses`: `N x 3 uint16` (conformer, rotamer, offset index).
    - `mean_offset`: `int16[3]`.
    - `offsets`: `K x 3 uint8` values representing `int8` deltas.

- **Inputs** to `unpack_poses`:
  - A single `(poses, mean_offset, offsets)` tuple.

- **Outputs** from `unpack_poses`:
  - `conformer_indices`, `rotamer_indices`, `offset_indices`, `offset_table`.
  - `offset_table` is `int16[K,3]` translations (deduped).

## Deduped Offsets
- Use `expand_discrete_offsets` + `gather_discrete_offsets` to generate translations in gathered order.
- Deduplicate translations with `np.unique(axis=0, return_inverse=True)`.
- Store `offset_indices` (inverse mapping) in `poses.npy` third column.
- Store the unique offsets as `offset_table`.

## Split Logic (Architecture Constraints)
- If **offset spread** in any axis exceeds `255`, split into a new chunk.
- If **unique offsets** exceed `2**16`, split into a new chunk.
- If **poses** exceed `2**32`, split into a new chunk.
- Each chunk gets its own `(poses, mean_offset, offsets)` tuple.

## Round‑Trip Test
- Modify Stage A simulation to track chosen **conformer and rotamer indices** per displacement.
- Expand these indices using `tinds` from `gather_discrete_offsets`, so inputs align with gathered translation order.
- Call `pack_all_poses` and assert the list length is 1 for this test.
- Call `unpack_poses`, reconstruct translations via `offset_table[offset_indices]`.
- Assertions:
  - `conformer_indices` and `rotamer_indices` match the packed inputs.
  - reconstructed translations match the gathered translations (`tdata`).

## Assumptions and Defaults
- Pose indices correspond to the **AC conformer/rotamer** (conf2/rot2) for this Stage B test.
- Inputs to `pack_all_poses` are already aligned to the gathered translation order.
- Deduped offsets are acceptable for round‑trip verification via reconstruction.
