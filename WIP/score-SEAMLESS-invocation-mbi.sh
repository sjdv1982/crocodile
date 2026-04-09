set -u -e
bfwd=$1
first=$2
last=$3

remote_jobdir=/data3/sdevries/1b7f-test/1b7f-frag3-$bfwd-$first-$last

./score-SEAMLESS.sh \
    /ramscratch/1b7f-frag3-$bfwd/files \
    $first \
    $last \
    UG \
    1b7f_dom2-aar.pdb \
    fraglib-UG-ex1b7f.npy \
    UG-atomtypes.npy \
    compiled \
    "$remote_jobdir"
