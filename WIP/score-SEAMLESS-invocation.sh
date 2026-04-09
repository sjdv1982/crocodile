remote_jobdir=$(pwd)/RUN-score-frag3-bwd-compiled

./score-SEAMLESS.sh \
    tests/1b7f/frag3-bwd \
    1 \
    1 \
    UG \
    1b7f_dom2-aar.pdb \
    fraglib-UG-ex1b7f.npy \
    UG-atomtypes.npy \
    compiled \
    "$remote_jobdir"
