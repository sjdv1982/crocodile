set -u -e
bfwd=$1
first=$2
last=$3

remote_jobdir=/data3/sdevries/1b7f-test/1b7f-frag4-$bfwd-$first-$last

./score-SEAMLESS.sh \
    /ramscratch/1b7f-frag4-$bfwd/files \
    $first \
    $last \
    GU \
    1b7f_dom2-aar.pdb \
    fraglib-GU-ex1b7f.npy \
    GU-atomtypes.npy \
    compiled \
    "$remote_jobdir"
