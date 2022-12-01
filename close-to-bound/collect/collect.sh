for d in  0.5 1; do
    python3 nefertiti/functions/rotmat2euler.py ../close-to-bound-${d}A-mat.npy close-to-bound-${d}A-euler.npy
    python3 nefertiti/functions/euler2dat.py close-to-bound-${d}A-euler.npy > close-to-bound-${d}A.dat
done
python3 collect-top100-nefertiti.py
./collect-top100-attract.sh
