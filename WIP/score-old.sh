#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JAX_ENV="${JAX_ENV:-jax}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(conda run --no-capture-output -n "${JAX_ENV}" python)
fi

POSES="${POSES:-poses-1.npy}"
OFFSETS="${OFFSETS:-offsets-1.dat}"
SEQUENCE="${SEQUENCE:-UG}"

GRID="${GRID:-1b7f_dom2-aar.grid}"
ATTRACT_PAR_NPZ="${ATTRACT_PAR_NPZ:-${ROOT}/attract-jax/attract-par.npz}"
RECEPTOR_ENS_LIST="${RECEPTOR_ENS_LIST:-}"
RECEPTOR_PDB="${RECEPTOR_PDB:-}"
RECEPTOR_COORDINATES="${RECEPTOR_COORDINATES:-1b7f_dom2-aar-ocordinates.npy}"
RECEPTOR_ATOMTYPES="${RECEPTOR_ATOMTYPES:-1b7f_dom2-aar-atomtypes.npy}"
RECEPTOR_CHARGES="${RECEPTOR_CHARGES:-}"

LIGAND_ENSEMBLE="${LIGAND_ENSEMBLE:-fraglib-UG-ex1b7f.npy}"
LIGAND_CONFORMERS="${LIGAND_CONFORMERS:-}"
LIGAND_ATOMTYPES="${LIGAND_ATOMTYPES:-UG-atomtypes.npy}"
LIGAND_CHARGES="${LIGAND_CHARGES:-}"
LIGAND_PDB="${LIGAND_PDB:-}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-poses-1}"
ORACLE="${ORACLE:-jax}"
NB_KERNEL="${NB_KERNEL:-nonbon8}"
SCORE_MODE="${SCORE_MODE:-}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-}"
POOL_CONFORMERS="${POOL_CONFORMERS:-0}"

CONVERT_SCRIPT="${ROOT}/crocodile/code/convert_poses.py"
CONVERT_TO_LEGACY_SCRIPT="${ROOT}/crocodile/code/convert_poses_to_legacy.py"
MINFOR_PY="${ROOT}/attract-jax/util/minfor.py"

if [[ ! -f "${POSES}" ]]; then
  echo "Missing poses file: ${POSES}" >&2
  exit 1
fi
if [[ ! -f "${OFFSETS}" ]]; then
  echo "Missing offsets file: ${OFFSETS}" >&2
  exit 1
fi
if [[ ! -f "${GRID}" ]]; then
  echo "Missing grid file: ${GRID}" >&2
  exit 1
fi
if [[ ! -f "${ATTRACT_PAR_NPZ}" ]]; then
  echo "Missing attract-par NPZ: ${ATTRACT_PAR_NPZ}" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT INT TERM

tmp_prefix="${tmpdir}/poses"
tmp_rotvec="${tmp_prefix}.rotvec.npy"
tmp_conformers0="${tmp_prefix}.conformers.npy"
tmp_conformers1="${tmp_prefix}.conformers1.npy"
tmp_score="${tmpdir}/score.out"
out_dat="${OUTPUT_PREFIX}.dat"

if [[ -n "${LIGAND_ENSEMBLE}" && -f "${LIGAND_ENSEMBLE}" ]]; then
  ligand_pose_source_kind="ensemble"
  ligand_pose_source_path="${LIGAND_ENSEMBLE}"
elif [[ -n "${LIGAND_PDB}" ]]; then
  ligand_pose_source_kind="pdb"
  ligand_pose_source_path="${LIGAND_PDB}"
else
  echo "Set LIGAND_ENSEMBLE or LIGAND_PDB" >&2
  exit 1
fi

echo "Converting poses to rotvec DOFs..." >&2
"${PYTHON_CMD[@]}" "${CONVERT_SCRIPT}" \
  --poses "${POSES}" \
  --offsets "${OFFSETS}" \
  --sequence "${SEQUENCE}" \
  --output-prefix "${tmp_prefix}"

echo "Writing ATTRACT .dat..." >&2
convert_legacy_cmd=(
  "${PYTHON_CMD[@]}" "${CONVERT_TO_LEGACY_SCRIPT}"
  --input-rotvec "${tmp_rotvec}"
  --output-dat "${out_dat}"
)
if [[ -n "${LIGAND_CONFORMERS}" ]]; then
  convert_legacy_cmd+=(--input-conformers1 "${LIGAND_CONFORMERS}")
else
  convert_legacy_cmd+=(--input-conformers0 "${tmp_conformers0}")
fi
if [[ "${ligand_pose_source_kind}" == "ensemble" ]]; then
  convert_legacy_cmd+=(--ligand-ensemble "${ligand_pose_source_path}")
else
  convert_legacy_cmd+=(--ligand-pdb "${ligand_pose_source_path}")
fi
"${convert_legacy_cmd[@]}"

if [[ -z "${RECEPTOR_ENS_LIST}" ]] && ! { [[ -n "${RECEPTOR_COORDINATES}" ]] && [[ -f "${RECEPTOR_COORDINATES}" ]]; }; then
  if [[ -z "${RECEPTOR_PDB}" && -f "1b7f_dom2-aar.pdb" ]]; then
    RECEPTOR_PDB="1b7f_dom2-aar.pdb"
  fi
  if [[ -z "${RECEPTOR_PDB}" ]]; then
    echo "Set RECEPTOR_ENS_LIST or RECEPTOR_PDB" >&2
    exit 1
  fi
fi

cmd=(
  "${PYTHON_CMD[@]}" "${MINFOR_PY}" "${out_dat}"
  --score
  --energy-only
  --oracle "${ORACLE}"
  --grid "${GRID}"
  --attract-par-npz "${ATTRACT_PAR_NPZ}"
  --nb-kernel "${NB_KERNEL}"
)

if [[ -n "${SCORE_MODE}" ]]; then
  cmd+=(--score-mode "${SCORE_MODE}")
fi
if [[ -n "${SCORE_BATCH_SIZE}" ]]; then
  cmd+=(--score-batch-size "${SCORE_BATCH_SIZE}")
fi
if [[ "${POOL_CONFORMERS}" == "1" ]]; then
  cmd+=(--pool-conformers)
fi

if [[ -n "${RECEPTOR_COORDINATES}" && -f "${RECEPTOR_COORDINATES}" ]]; then
  cmd+=(--receptor-coordinates "${RECEPTOR_COORDINATES}")
  cmd+=(--receptor-atomtypes "${RECEPTOR_ATOMTYPES}")
  if [[ -n "${RECEPTOR_CHARGES}" ]]; then
    cmd+=(--receptor-charges "${RECEPTOR_CHARGES}")
  fi
elif [[ -n "${RECEPTOR_ENS_LIST}" ]]; then
  cmd+=(--receptor-ens-list "${RECEPTOR_ENS_LIST}")
else
  cmd+=(--receptor-pdb "${RECEPTOR_PDB}")
fi

if [[ -n "${LIGAND_ENSEMBLE}" && -f "${LIGAND_ENSEMBLE}" ]]; then
  cmd+=(--ligand-ensemble "${LIGAND_ENSEMBLE}")
  cmd+=(--ligand-atomtypes "${LIGAND_ATOMTYPES}")
  if [[ -n "${LIGAND_CHARGES}" ]]; then
    cmd+=(--ligand-charges "${LIGAND_CHARGES}")
  fi
else
  if [[ -z "${LIGAND_PDB}" ]]; then
    echo "Set LIGAND_PDB when ligand ensemble files are not available" >&2
    exit 1
  fi
  cmd+=(--ligand-pdb "${LIGAND_PDB}")
fi

echo "Scoring with minfor.py (.dat)..." >&2
t_score_start=$(date +%s%N)
"${cmd[@]}" > "${tmp_score}"
t_score_end=$(date +%s%N)
t_score_ms=$(( (t_score_end - t_score_start) / 1000000 ))
echo "minfor.py (.dat score) finished in ${t_score_ms} ms" >&2

"${PYTHON_CMD[@]}" - "${tmp_score}" "${OUTPUT_PREFIX}.ene" "${OUTPUT_PREFIX}.ene.npy" <<'PY'
import re
import sys

import numpy as np

score_path, ene_txt, ene_npy = sys.argv[1:]
energy_re = re.compile(r"^\s*Energy:\s*([-+0-9.eE]+)\s*$")
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
