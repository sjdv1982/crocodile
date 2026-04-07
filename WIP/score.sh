set -euo pipefail

pose_dir=$1
first_index=$2  # e.g. 1 for poses-1.npy.zst
last_index=$3   # e.g. 1 for poses-1.npy.zst
sequence=$4  # e.g. UG
receptor_pdb=$5  # reduced, e.g. 1b7f_dom2-aar.pdb
ligand_ensemble=$6 # e.g. fraglib-UG-ex1b7f.npy
ligand_atomtypes=$7  #e.g. UG-atomtypes.npy
nb_kernel=$8 # compiled or jax
output_file=${9:-energies.npy}  # output .npy with energies

SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

CONVERT_SCRIPT=""

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT INT TERM

if (( first_index > last_index )); then
  echo "first_index must be <= last_index" >&2
  exit 1
fi

tmp_prefix=poses-${first_index}
if (( first_index != last_index )); then
  tmp_prefix+="-${last_index}"
fi
tmp_rotvec="${tmp_prefix}.rotvec.npy"
tmp_conformers="${tmp_prefix}.conformers.npy"
tmp_score="${tmpdir}/score.out"

# --- Step 1: convert poses → rotvec DOFs ---
echo "Converting poses to rotvec DOFs..." >&2
t_convert_start=$(date +%s%N)
convert_cmd=(
  python code/convert_poses.py
  --pose-dir "${pose_dir}"
  --first-index "${first_index}"
  --last-index "${last_index}"
  --sequence "${sequence}"
  --output-prefix "${tmp_prefix}"
)
"${convert_cmd[@]}"
t_convert_end=$(date +%s%N)
t_convert_ms=$(( (t_convert_end - t_convert_start) / 1000000 ))
echo "convert_poses.py finished in ${t_convert_ms} ms" >&2

# --- Step 2: score with minfor.py --input-rotvec ---
cmd=(
  python -u attract-jax/util/minfor.py
  --input-rotvec "${tmp_rotvec}"
  --input-conformers "${tmp_conformers}"
  --input-world-centered
  --score
  --energy-only
  --oracle jax
  --attract-par-npz attract-jax/attract-par.npz
  --nb-kernel "${nb_kernel}"
  --output-npy "${output_file}"
)

cmd+=(--score-mode bulk)
if [[ -n "${SCORE_BATCH_SIZE}" ]]; then
  cmd+=(--score-batch-size "${SCORE_BATCH_SIZE}")
fi

cmd+=(--receptor-pdb "${receptor_pdb}")
cmd+=(--ligand-ensemble "${ligand_ensemble}")
cmd+=(--ligand-atomtypes "${ligand_atomtypes}")

echo "Scoring with minfor.py (rotvec)..." >&2
t_score_start=$(date +%s%N)
env XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}" "${cmd[@]}" > "${tmp_score}"
t_score_end=$(date +%s%N)
t_score_ms=$(( (t_score_end - t_score_start) / 1000000 ))
echo "minfor.py (score) finished in ${t_score_ms} ms" >&2
