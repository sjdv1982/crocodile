set -u -e

currdir=`python3 -c 'import os,sys;print(os.path.dirname(os.path.realpath(sys.argv[1])))' $0`

fragment=$(head -n 1 $1/FRAG)
seq=$(python -c 'import json, sys; j=json.load(open("constraints.json")); print(j["frag" + sys.argv[1]]["sequence"])' $fragment)
cd $1
rec=$2  #e.g. 1b7f_dom1


tmp1=$(mktemp --suffix=.npy)  #conformers
tmp2=$(mktemp --suffix=.npy)  #atomtypes
tmp3=$(mktemp --suffix=.npy)  #energy

trap cleanup 1 2 3 6 15

function cleanup()
{
  rm -f $tmp1 $tmp2 $tmp3
  exit
}

python -c '''
import sys, numpy as np
poses0 = np.load(sys.argv[1])
poses = np.empty((len(poses0), 4, 4))
poses[:, 3, 3] = 1
poses[:, :3, :3] = poses0["rotation"]
poses[:, 3, :3] = poses0["offset"]
np.save(sys.argv[2], poses)
np.save(sys.argv[3], poses0["conformer"] + 1)
''' poses.npy $tmp1 $tmp2

python3 $currdir/score-attract-jax.py $tmp1 \
    score-data/${rec}r-coor.npy score-data/lib-$seq-reduced.npy \
    --conformers $tmp2 \
    --atrec score-data/${rec}r-atomtypes.npy \
    --atlig templates/reduced/$seq-template-atomtypes.npy \
    --grid score-data/${rec}r.grid \
    --output $tmp3

python -c '''
import sys, numpy as np; a = np.load(sys.argv[1])
np.savetxt(sys.argv[2], a, fmt="%.3f")
''' $tmp3 poses-${rec}.ene

rm -f $tmp1 $tmp2 $tmp3
