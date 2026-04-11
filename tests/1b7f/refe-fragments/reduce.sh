set -u -e
for i in `seq 11`; do
  for f in '' '-bestfit'; do
    python $ATTRACTTOOLS/reduce.py frag-$i$f.pdb frag-$i$f-reduced.pdb
    echo $i $f
  done
done