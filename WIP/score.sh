# awk '{print $9}' UG.pdb  | sort -nu > UG.alphabet

# python ../code/decode_rotamer_matrices.py --poses poses-1.npy --offsets offsets-1.dat --sequence UG --output rotamer_matrix.npy

# ~/attract/bin/make-grid-omp 1b7f_dom2-aar.pdb  ~/attract/attract.par 5 7 1b7f_dom2-aar.grid UG.alphabet

# python3 get-conformers.py poses-1.npy poses-1-conformers.npy
#TODO:1b7f_dom2-aar-atomtypes.npy
#TODO: 1b7f_dom2-aar-ocordinates.npy

# python3 ../code/get-reduced-library.py UG -x 1b7f fraglib-UG-ex1b7f.npy
# ~/attract/bin/make-grid-omp 1b7f_dom2-aar.pdb  ~/attract/attract.par 5 7 1b7f_dom2-aar.grid --alphabet UG.alphabet
python protocols/score-attract-jax.py --atrec 1b7f_dom2-aar-atomtypes.npy --grid 1b7f_dom2-aar.grid  --atlig UG-atomtypes.npy --conformers poses-1-conformers.npy rotamer_matrix.npy 1b7f_dom2-aar-ocordinates.npy fraglib-UG-ex1b7f.npy --output  poses-1.ene