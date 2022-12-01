set -u -e
for d in 0.5 1; do
    $ATTRACTTOOLS/top close-to-bound-${d}A.dat 100 > temp.dat
    $ATTRACTDIR/collect temp.dat /dev/null ../close-to-bound.pdb > collect-top100-attract-${d}.pdb
    rm -f temp.dat
done