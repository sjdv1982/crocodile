import sys, os
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation

from crocodile.main import import_config
from crocodile.nuc.library import LibraryFactory
from nefertiti.functions.superimpose import superimpose

import_config(os.getcwd())
from crocodile_library_config import dinucleotide_libraries

motif = sys.argv[1]
nucpos = sys.argv[2]
conflist = sys.argv[3]
assert nucpos in ("1", "2"), sys.argv[2]
nuc_motif = motif[0] if nucpos == "1" else motif[1]

conformers = []
for l in open(conflist):
    for ll in l.split():
        conformers.append(int(ll))


conf_prototypes = np.load(
    f"rotaclustering/dinuc-{motif}-nuc{nucpos}-assign-prototypes.npy"
)
prototypes_scalevec = np.loadtxt(f"monobase-prototypes-{nuc_motif}-scalevec.txt")
nconf = len(conf_prototypes)

all_clusters = {}
for prototype_index in range(len(prototypes_scalevec)):
    cluster_file = f"rotaclustering/lib-nuc-{motif}-nuc{nucpos}-prototype-{prototype_index+1}-clusters.npy"
    clusters = np.load(cluster_file)
    all_clusters[prototype_index] = clusters

common_base = motif[0] if nucpos == "1" else motif[1]
prototypes = np.load(f"monobase-prototypes-{common_base}.npy")

if nucpos == "1":
    nucleotide_mask = [True, False]
else:
    nucleotide_mask = [False, True]

libf: LibraryFactory = dinucleotide_libraries[motif]
print("LOAD")
libf.load_rotaconformers()
print("/LOAD")
lib = libf.create(None, nucleotide_mask=nucleotide_mask)


for conformer in tqdm(conformers):
    conf = conformer - 1
    prototype_index = conf_prototypes[conf]

    scalevec = prototypes_scalevec[prototype_index]
    clusters = all_clusters[prototype_index]

    clusters = Rotation.from_matrix(clusters)

    digits = f"{conformer%100:02d}"
    outputdir = f"rotamember/{digits}"
    os.makedirs(outputdir, exist_ok=True)
    outputfile = (
        f"{outputdir}/lib-nuc-{motif}-nuc{nucpos}-{conformer}-cluster-membership.npy"
    )

    if os.path.exists(outputfile):
        continue

    rotvec = libf.create(None, with_rotaconformers=True).get_rotamers(conf)
    prototype = prototypes[prototype_index]
    conf_coor = lib.coordinates[conf].copy()
    conf_coor -= conf_coor.mean(axis=0)
    proto_align, proto_rmsd = superimpose(conf_coor, prototype)
    rotations0 = Rotation.from_rotvec(rotvec).as_matrix()

    rotations = proto_align.T.dot(rotations0).swapaxes(0, 1)
    rotations = Rotation.from_matrix(rotations)

    n_membership = int(len(clusters) / 2 + 0.5)
    membership = np.zeros((n_membership, len(rotations)), np.uint8)
    membership_bins = [
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4,
        4.5,
        5,
    ]
    for n in range(len(clusters)):
        rr = rotations * clusters[n].inv()
        ax = rr.as_rotvec()
        ang = np.linalg.norm(ax, axis=1)
        ang = np.maximum(ang, 0.0001)
        ax /= ang[:, None]
        fac = (np.cos(ang) - 1) ** 2 + np.sin(ang) ** 2
        cross = (scalevec * scalevec) * (1 - ax * ax)
        rmsd = np.sqrt(fac * cross.sum(axis=-1))
        member = np.digitize(rmsd, membership_bins).astype(np.uint8)

        pos = n // 2
        bitshift = (n % 2) * 4
        membership[pos] += member << bitshift

    np.save(outputfile, membership)
