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
    --output 1B7F-trinuc-fit.npy

python ../../crocodile/trinuc/to_ppdb.py 1B7F-trinuc-fit.npy --rna \
    --templates XXX-template-ppdb.npy \
    --conformers  ~/data/work/ProtNAff/database/trilib/XXX-lib-conformer.npy \
    --output 1B7F-trinuc-fit-pdb.npy


python $NEFERTITI/functions/write_pdb.py 1B7F-trinuc-fit-pdb.npy 1B7F-trinuc-fit.pdb