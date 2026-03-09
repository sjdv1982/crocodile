#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DAT="${1:-${INPUT_DAT:-poses-1.dat}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${INPUT_DAT%.dat}.legacy}"
OUTPUT_SCORE="${OUTPUT_SCORE:-${OUTPUT_PREFIX}.score}"
OUTPUT_ENE="${OUTPUT_ENE:-${OUTPUT_PREFIX}.ene}"
OUTPUT_ENE_NPY="${OUTPUT_ENE_NPY:-${OUTPUT_PREFIX}.ene.npy}"

ATTRACTDIR="${ATTRACTDIR:-${ROOT}/attract/bin}"
PARFILE="${PARFILE:-${ATTRACTDIR}/../attract.par}"
RECEPTOR_PDB="${RECEPTOR_PDB:-1b7f_dom2-aar.pdb}"
LIGAND_PDB="${LIGAND_PDB:-UG.pdb}"
LIGAND_ENS_LIST="${LIGAND_ENS_LIST:-}"
GRID="${GRID:-1b7f_dom2-aar.grid}"
PARALS="${PARALS:---np 4 --chunks 4}"

if [[ ! -f "${INPUT_DAT}" ]]; then
  echo "Missing input DAT: ${INPUT_DAT}" >&2
  exit 1
fi
if [[ ! -f "${ATTRACTDIR}/attract" ]]; then
  echo "Missing ATTRACT binary: ${ATTRACTDIR}/attract" >&2
  exit 1
fi
if [[ ! -f "${PARFILE}" ]]; then
  echo "Missing ATTRACT parameter file: ${PARFILE}" >&2
  exit 1
fi
if [[ ! -f "${RECEPTOR_PDB}" ]]; then
  echo "Missing receptor PDB: ${RECEPTOR_PDB}" >&2
  exit 1
fi
if [[ ! -f "${LIGAND_PDB}" ]]; then
  echo "Missing ligand PDB: ${LIGAND_PDB}" >&2
  exit 1
fi
if [[ ! -f "${GRID}" ]]; then
  echo "Missing grid file: ${GRID}" >&2
  exit 1
fi
if [[ -n "${LIGAND_ENS_LIST}" && ! -f "${LIGAND_ENS_LIST}" ]]; then
  echo "Missing ligand ensemble list: ${LIGAND_ENS_LIST}" >&2
  exit 1
fi

parals_arr=()
if [[ -n "${PARALS}" ]]; then
  # shellcheck disable=SC2206
  parals_arr=(${PARALS})
fi

ens_args=()
if [[ -n "${LIGAND_ENS_LIST}" ]]; then
  ens_args=(--ens 1 "${LIGAND_ENS_LIST}")
fi

echo "Legacy rescoring ${INPUT_DAT} -> ${OUTPUT_SCORE}" >&2
python "${ATTRACTDIR}/../protocols/attract.py" \
  "${INPUT_DAT}" \
  "${PARFILE}" \
  "${RECEPTOR_PDB}" \
  "${LIGAND_PDB}" \
  --fix-receptor \
  "${ens_args[@]}" \
  --grid 1 "${GRID}" \
  "${parals_arr[@]}" \
  --score \
  --output "${OUTPUT_SCORE}"

python - "${OUTPUT_SCORE}" "${OUTPUT_ENE}" "${OUTPUT_ENE_NPY}" <<'PY'
import re
import sys
import numpy as np

score_path, ene_txt, ene_npy = sys.argv[1:]
energy_re = re.compile(r"^##\s*Energy:\s*([-+0-9.eE]+)")
energies = []
with open(score_path) as handle:
    for line in handle:
        match = energy_re.match(line)
        if match:
            energies.append(float(match.group(1)))

arr = np.asarray(energies, dtype=np.float64)
np.savetxt(ene_txt, arr, fmt="%.3f")
np.save(ene_npy, arr)
PY

echo "Done. Wrote ${OUTPUT_SCORE}, ${OUTPUT_ENE}, and ${OUTPUT_ENE_NPY}" >&2
