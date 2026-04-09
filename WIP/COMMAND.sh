#!/bin/sh
time python -u attract-jax/util/minfor.py \
    --input-npy poses-1.rotvec.npy --input-format rotvec \
    --input-conformers poses-1.conformers.npy \
    --input-world-centered --score --energy-only --oracle jax --attract-par-npz attract-jax/attract-par.npz \
    --nb-kernel compiled --receptor-pdb 1b7f_dom2-aar.pdb \
    --ligand-ensemble fraglib-UG-ex1b7f.npy --ligand-atomtypes UG-atomtypes.npy \
    --output-npy score.npy