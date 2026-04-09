remote_jobdir=$(pwd)/RUN-score-frag3-fwd-1-4-compiled

./score-SEAMLESS.sh \
    ramscratch/1b7f-frag3-fwd/files \
    1 \
    4 \
    UG \
    1b7f_dom2-aar.pdb \
    fraglib-UG-ex1b7f.npy \
    UG-atomtypes.npy \
    compiled \
    "$remote_jobdir"
