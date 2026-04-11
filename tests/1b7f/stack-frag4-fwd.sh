# frag4-fwd
rm -rf frag4-fwd
python3 code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --first \
  --angle 30 --dihedral 45 -45 \
  --test-conformers 100 --test-rotamers 1000 \
  --output frag4-fwd/
