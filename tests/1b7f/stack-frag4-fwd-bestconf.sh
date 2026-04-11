# frag4-fwd-bestconf
rm -rf frag4-fwd-bestconf
python3 code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --first \
  --angle 25 --dihedral 45 -45 \
  --conformer 1975 \
  --output frag4-fwd-bestconf/
