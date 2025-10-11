"""
new clustering experiment with adenosine base

Select 20 000 random poses, the real clusters will be chosen among them.
cluster them such that:
- There will be up to 255 clusters
- Within (radius + 1) angstroms, a structure belongs to around 5-10 clusters
=> radius 1.5 seems to be a sweet spot
"""

import sys
import os
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation
from crocodile.main import import_config
from crocodile.main.superimpose import superimpose_array
from crocodile.nuc.library import LibraryFactory

RADIUS = 1.5
SAMPLE_SIZE = 20000
CHUNKSIZE = 1000000

import_config(os.getcwd())
from crocodile_library_config import dinucleotide_libraries

motif = sys.argv[1]
nucpos = sys.argv[2]
assert nucpos in ("1", "2"), sys.argv[2]
prototype_index = int(sys.argv[3]) - 1

nuc_motif = motif[0] if nucpos == "1" else motif[1]
prototypes_scalevec = np.loadtxt(f"monobase-prototypes-{nuc_motif}-scalevec.txt")
scalevec = prototypes_scalevec[prototype_index]
prototypes = np.load(f"monobase-prototypes-{nuc_motif}.npy")
prototype = prototypes[prototype_index]
conf_prototypes = np.load(
    f"rotaclustering/dinuc-{motif}-nuc{nucpos}-assign-prototypes.npy"
)

outfile = f"rotaclustering/lib-nuc-{motif}-nuc{nucpos}-prototype-{prototype_index+1}-clusters.npy"

libf: LibraryFactory = dinucleotide_libraries[motif]
print("LOAD")
libf.load_rotaconformers()
print("/LOAD")
nucleotide_mask = [True, False] if nucpos == "1" else [False, True]
lib = libf.create(
    None, only_base=False, nucleotide_mask=nucleotide_mask, with_rotaconformers=True
)

transform_mat, _ = superimpose_array(lib.coordinates, prototype)
rotaconformers = np.empty((len(libf.rotaconformers), 3), np.float32)

pos = 0
for n in trange(
    len(lib.coordinates),
    desc=f"{motif} {nucpos} {prototype_index+1}, alignment to prototype",
):
    if conf_prototypes[n] != prototype_index:
        continue
    tf = transform_mat[n]
    rotamers0 = lib.get_rotamers(n)
    rotamers = Rotation.from_rotvec(rotamers0).as_matrix()
    nextpos = pos + len(rotamers)
    tf_rotamers0 = tf.T.dot(rotamers).swapaxes(0, 1)
    tf_rotamers = Rotation.from_matrix(tf_rotamers0).as_rotvec()
    rotaconformers[pos:nextpos] = tf_rotamers
    pos = nextpos

rotaconformers = rotaconformers[:pos]


rotvec = rotaconformers[np.random.choice(len(rotaconformers), size=SAMPLE_SIZE)]
rotations = Rotation.from_rotvec(rotvec)

rmsd_matrix = np.zeros((len(rotations), len(rotations)))
for refenr, refe in enumerate(
    tqdm(
        rotations[:-1],
        desc=f"{motif} {nucpos} {prototype_index+1}, pairwise RMSD calculation",
    )
):
    other = rotations[refenr + 1 :]
    rr = other * refe.inv()
    ax = rr.as_rotvec()
    ang = np.linalg.norm(ax, axis=1)
    ang = np.maximum(ang, 0.0001)
    ax /= ang[:, None]
    fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
    cross = (scalevec * scalevec) * (1 - ax * ax)
    rmsd = np.sqrt(fac * cross.sum(axis=-1))
    rmsd_matrix[refenr, refenr + 1 :] = rmsd
    rmsd_matrix[refenr + 1 :, refenr] = rmsd

nbmat = rmsd_matrix < RADIUS
clusters = []
nclust = 0
min_to_cluster = int(len(nbmat) * 0.98)
with tqdm(
    list(range(min_to_cluster)), f"{motif} {nucpos} {prototype_index+1}, clustering"
) as progress:
    while nclust < min_to_cluster:
        neigh = nbmat.sum(axis=0)
        heart = neigh.argmax()
        leaf = np.where(nbmat[heart])[0]
        for cs in leaf:
            nbmat[cs, :] = False
            nbmat[:, cs] = False
        clusters.append(heart)
        nclust += len(leaf)
        progress.update(len(leaf))

cluster_rotations = rotations[clusters].as_matrix()
np.save(outfile, np.array(cluster_rotations))
