set -u -e
conda create -n crocodile numpy scipy julia pip
pip install juliacall tqdm biopython opt_einsum
set -u +e
for i in $(seq 10); do
    python -c 'from juliacall import Main'
done
set -u -e
python -c 'from juliacall import Main'
python -c 'from juliacall import Main'

julia -e 'import Pkg; Pkg.add(["NPZ", "ProgressMeter"])'