set -u -e

inp1=$1
inp2=$2
outp=$3
jobdir=$4
seamless-run --dry --write-remote-job $jobdir \
    --conda filter-identical-poses \
    --prolog 'export CROCODILE_ZSTD_THREADS=1; export JULIA_NUM_THREADS=24' \
    --metavar inp1=$inp1 --metavar inp2=$inp2 --metavar outp=$outp \
    'julia code/filter_identical_poses.jl $inp1 $inp2 $outp'
