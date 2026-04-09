set -euo pipefail

pose_dir=$1
first_index=$2
last_index=$3
sequence=$4  # e.g. UG
receptor_pdb=$5  # reduced, e.g. 1b7f_dom2-aar.pdb
ligand_ensemble=$6 # e.g. fraglib-UG-ex1b7f.npy
ligand_atomtypes=$7  #e.g. UG-atomtypes.npy
nb_kernel=$8 # compiled or jax
remote_jobdir=$9

test ! -e "$remote_jobdir"


MINFOR="attract-jax/util/minfor.py"
CONVERT_POSES="code/convert_poses.py"
seamless-run -vvv -y --dry --write-remote-job "$remote_jobdir" \
  -i "${CONVERT_POSES}" \
  -I "${CONVERT_POSES}.DEPS.txt" \
  -i "${MINFOR}" \
  -I "${MINFOR}.DEPS.txt" \
  -i attract-jax/attract-par.npz \
  --conda jax \
  score.sh "$pose_dir" "$first_index" "$last_index" "$sequence" "$receptor_pdb" "$ligand_ensemble" "$ligand_atomtypes" "$nb_kernel"
