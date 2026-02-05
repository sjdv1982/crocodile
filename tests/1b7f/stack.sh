# frag3
python3 code/stack.py --sequence UG --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --second \
  --angle 24 --dihedral -45 45 \
  --test-conformers 500 --test-rotamers 2000 \
  --output poses.npy offsets.dat
