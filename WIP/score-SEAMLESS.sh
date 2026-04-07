set -euo pipefail

pose_dir=$1  
pose_dir_index=$2  # e.g. 1 for poses-1.npy.zst
sequence=$3  # e.g. UG
receptor_pdb=$4  # reduced, e.g. 1b7f_dom2-aar.pdb
ligand_ensemble=$5 # e.g. fraglib-UG-ex1b7f.npy
ligand_atomtypes=$6  #e.g. UG-atomtypes.npy
nb_kernel=$7 # compiled or jax
remote_jobdir=$8  

test ! -e "$remote_jobdir"


MINFOR="attract-jax/util/minfor.py"
CONVERT_POSES="code/convert_poses.py"
seamless-run -vvv -y --dry --write-remote-job $remote_jobdir \
  -i ${CONVERT_POSES} \
  -I ${CONVERT_POSES}.DEPS.txt \
  -i ${MINFOR} \
  -I ${MINFOR}.DEPS.txt \
  -i attract-jax/attract-par.npz \
  --conda jax \
  score.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
