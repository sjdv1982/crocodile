#!/bin/bash

set -u -e
NEFERTITI=$(python -c 'import os, nefertiti; print(os.path.dirname(nefertiti.__file__))')

#python $NEFERTITI/functions/parse_pdb.py RNA-aa.pdb RNA-aa.npy
#for m1 in A C G U; do
#    for m2 in A C G U; do
#        for m3 in A C G U; do
#            m=$m1$m2$m3
#            python $NEFERTITI/functions/parse_pdb.py ~/data/work/ProtNAff/templates/$m-template.pdb $m-template-ppdb.npy
#        done
#    done
#done

python ../../crocodile/trinuc/from_ppdb.py RNA-aa.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --ignore-reordered \
    --output 1B7F-trinuc-fit.npy

python ../../crocodile/trinuc/best_fit.py 1B7F-trinuc-fit.npy > 1B7F-trinuc-fit-best.rmsd
python ../../crocodile/trinuc/best_fit.py 1B7F-trinuc-fit.npy 2 > 1B7F-trinuc-fit-2ndbest.rmsd

python ../../crocodile/trinuc/to_ppdb.py 1B7F-trinuc-fit.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --output 1B7F-trinuc-fit-pdb.npy

python $NEFERTITI/functions/write_pdb.py 1B7F-trinuc-fit-pdb.npy 1B7F-trinuc-fit.pdb

mkdir -p rotaconformer-buffers
python identify-needed-rotaconformer-buffers.py 1B7F-trinuc-fit.npy --rna \
    --rotaconformers  ~/data/work/crocodile/make-rotaconformers/results/XXX-lib-rotaconformer.json \
    --output rotaconformer-buffers

source $CONDA_PREFIX_1/etc/profile.d/conda.sh
conda activate seamless-development
seamless-download -y rotaconformer-buffers/*.CHECKSUM
conda deactivate

python ../../crocodile/trinuc/fit_roco.py 1B7F-trinuc-fit.npy RNA-aa.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --rotaconformers  ~/data/work/crocodile/make-rotaconformers/results/XXX-lib-rotaconformer.json \
    --rotaconformer-directory rotaconformer-buffers \
    --prefilter-min 0.1 \
    --prefilter-max 1.0 \
    --margin 0.5 \
    --output 1B7F-trinuc-roco-fit.npy

python ../../crocodile/trinuc/to_ppdb.py 1B7F-trinuc-roco-fit.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --output 1B7F-trinuc-roco-fit-pdb.npy

python $NEFERTITI/functions/write_pdb.py 1B7F-trinuc-roco-fit-pdb.npy 1B7F-trinuc-roco-fit.pdb

python ../../crocodile/trinuc/best_fit.py 1B7F-trinuc-roco-fit.npy > 1B7F-trinuc-roco-fit-best.rmsd
python ../../crocodile/trinuc/best_fit.py 1B7F-trinuc-roco-fit.npy 2 > 1B7F-trinuc-roco-fit-2ndbest.rmsd

python ../../crocodile/trinuc/discretize_grid.py 1B7F-trinuc-roco-fit.npy --output 1B7F-trinuc-grid-fit.npy

python ../../crocodile/trinuc/to_ppdb.py 1B7F-trinuc-grid-fit.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --output 1B7F-trinuc-grid-fit-pdb.npy

python $NEFERTITI/functions/write_pdb.py 1B7F-trinuc-grid-fit-pdb.npy 1B7F-trinuc-grid-fit.pdb

python ../../crocodile/trinuc/best_fit.py 1B7F-trinuc-grid-fit.npy > 1B7F-trinuc-roco-grid-fit.rmsd

python ../../crocodile/trinuc/get_overlap_rmsd.py 1B7F-trinuc-grid-fit.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    > 1B7F-overlap.rmsd


