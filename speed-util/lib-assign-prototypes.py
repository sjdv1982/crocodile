import os
import numpy as np
from crocodile.main import import_config
from crocodile.nuc.library import LibraryFactory

import_config(os.getcwd())
from crocodile_library_config import dinucleotide_libraries

for motif in ("AA", "AC", "CA", "CC"):

    libf:LibraryFactory = dinucleotide_libraries[motif]
    lib1 = libf.create(None, only_base=False, nucleotide_mask=[True, False], with_rotaconformers=False)
    lib2 = libf.create(None, only_base=False, nucleotide_mask=[False, True], with_rotaconformers=False)
    coor1 = lib1.coordinates - lib1.coordinates.mean(axis=1)[:, None]
    coor2 = lib2.coordinates - lib2.coordinates.mean(axis=1)[:, None]

    def get_scalevec(coor):
        v, s, wt = np.linalg.svd(coor)
        return s / np.sqrt(len(coor))

    scalevecs1 = np.array([get_scalevec(coor) for coor in coor1])
    scalevecs2 = np.array([get_scalevec(coor) for coor in coor2])

    prototype_scalevecs1 = np.loadtxt(f"monobase-prototypes-{motif[0]}-scalevec.txt")
    prototype_scalevecs2 = np.loadtxt(f"monobase-prototypes-{motif[1]}-scalevec.txt")

    def get_closest(scalevecs, prototype_scalevecs):
        d = scalevecs[:, None] - prototype_scalevecs[None, :]
        dif = (d*d).sum(axis=2)
        return dif.argmin(axis=1)

    closest1 = get_closest(scalevecs1, prototype_scalevecs1)
    closest2 = get_closest(scalevecs2, prototype_scalevecs2)
    print(motif, motif[0], np.unique(closest1, return_counts=True))
    print(motif, motif[1], np.unique(closest2, return_counts=True))
    print()
    np.save(f"rotaclustering/dinuc-{motif}-nuc1-assign-prototypes.npy", closest1)
    np.save(f"rotaclustering/dinuc-{motif}-nuc2-assign-prototypes.npy", closest2)
    