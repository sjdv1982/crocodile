# frag3-fwd
rm -rf frag3-fwd
python3 code/stack.py --sequence UG --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --second \
  --angle 24 --dihedral -45 45 \
  --test-conformers 100 --test-rotamers 1000 \
  --output frag3-fwd/
