for i in `seq 11`; do
  for f in '' '-bestfit'; do
    for p in 3sxl_dom1 3sxl_dom2 1b7f_dom1 1b7f_dom2; do
      ene=$($ATTRACTDIR/attract $ATTRACTDIR/../structure-single.dat $ATTRACTDIR/../attract.par \
        ${p}r.pdb frag-$i$f-reduced.pdb \
        --score --fix-receptor --rcut 9999999 | grep Energy | awk '{print $2+0}')
      echo $ene > frag-$i$f.$p.ene
      echo frag-$i$f $p
    done
  done
done