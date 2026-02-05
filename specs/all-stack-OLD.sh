export SEAMLESS_READ_BUFFER_FOLDERS=$HOME/rotaconformers/CAC
angle=24  # asin(0.4) in degrees
dihedral_min=45 # in degrees
dihedral_max=-45 # in degrees
python3 ~/data/work/crocodile/util/calc-all-rotamers-stacking-angle.py 1b7f_dom2.pdb all 214 ~/data/work/ProtNAff/templates/UGU-template.pdb all 2 ~/data/work/ProtNAff/database/trilib/UGU-lib-conformer.npy  ~/data/work/crocodile/make-rotaconformers/results/CAC-lib-rotaconformer.json stack-G2-214-radians.npy
python3 ~/data/work/crocodile/util/calc-all-rotamers-stacking-angle.py 1b7f_dom2.pdb all 256 ~/data/work/ProtNAff/templates/UGU-template.pdb all 3 ~/data/work/ProtNAff/database/trilib/UGU-lib-conformer.npy  ~/data/work/crocodile/make-rotaconformers/results/CAC-lib-rotaconformer.json stack-U3-256-radians.npy
python3 ~/data/work/crocodile/util/filter-rotamers-stacking-angle.py \
  $angle $dihedral_min $dihedral_max \
  rotaconformer-mask.npy \
  stack-G2-214-radians.npy stack-U3-256-radians.npy
python3 ~/data/work/crocodile/util/collect-mask.py \
  rotaconformer-mask.npy \
  ~/rotaconformers/CAC.count \
  ~/data/work/crocodile/make-rotaconformers/results/CAC-lib-rotaconformer.json \
  filtered-rotaconformers.npy
python3 ~/data/work/crocodile/util/pseudo-crocodile-2stack.py \
  3sxl_dom2.pdb all 214 256 \
  ~/data/work/ProtNAff/templates/UGU-template.pdb all 2 3 \
  ~/data/work/ProtNAff/database/trilib/UGU-lib-conformer.npy filtered-rotaconformers.npy \
  2stack-solutions.npy 2stack-offsets.npy
