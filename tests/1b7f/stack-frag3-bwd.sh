# frag3-bwd
rm -rf frag3-bwd
python3 code/stack.py --sequence UG --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 256 --first \
  --angle 24 --dihedral -45 45 \
  --test-conformers 100 --test-rotamers 1000 \
  --output frag3-bwd/
