import sys
import numpy as np
from tqdm import tqdm, trange
from crocodile.main.tensorlib import get_structure_tensor

CHUNKSIZE = 5000
SCALEVEC_THRESHOLD = 0.5

base = sys.argv[1]
assert base in ("A", "C")

baselen = 22 if base == "A" else 20

coors = []
for prepost, motif in (
    (1, base + "A"),
    (1, base + "C"),
    (0, "A" + base),
    (0, "C" + base),
):
    lib = np.concatenate(
        (
            np.load(f"library/dinuc-{motif}-0.5.npy"),
            np.load(f"library/dinuc-{motif}-0.5-extension.npy"),
        )
    )
    lib2 = lib[:, :baselen] if prepost else lib[:, -baselen:]
    coors.append(lib2)

coors = np.concatenate(coors)
coors -= coors.mean(axis=1)[:, None]
scalevecs = []
for cnr, coor in enumerate(tqdm(coors)):
    tensor, scalevec = get_structure_tensor(coor)
    coors[cnr] = coor.dot(tensor)
    scalevecs.append(scalevec)
scalevecs = np.array(scalevecs)

np.random.seed(0)
sample = scalevecs[np.random.choice(len(scalevecs), 10000)]

remain = scalevecs
indices = np.arange(len(scalevecs)).astype(int)

clusters = []
with tqdm(len(scalevec)) as progress:
    while len(remain) > 10000:
        nnb = []
        for n in trange(0, len(remain), CHUNKSIZE):
            chunk = remain[n : n + CHUNKSIZE]
            dif = chunk[:, None] - sample[None, :]
            d = np.sqrt((dif * dif).sum(axis=2))
            chunk_nnb = (d < SCALEVEC_THRESHOLD).sum(axis=1)
            nnb.append(chunk_nnb)
        nnb = np.concatenate(nnb)
        assert len(nnb) == len(remain)
        clus = nnb.argmax()
        clusters.append(indices[clus])
        dif = remain - remain[clus]
        d = np.sqrt((dif * dif).sum(axis=1))
        keep = ~(d < SCALEVEC_THRESHOLD)
        remain = remain[keep]
        indices = indices[keep]
        progress.update(len(scalevecs) - len(remain) - progress.n)

    dif = remain[:, None] - remain[None, :]
    d = np.sqrt((dif * dif).sum(axis=2))
    nbmat = d < SCALEVEC_THRESHOLD
    nclust = 0
    while nclust < len(nbmat):
        nnb = nbmat.sum(axis=0)
        heart = nnb.argmax()
        leaf = np.where(nbmat[heart])[0]
        for cs in leaf:
            nbmat[cs, :] = False
            nbmat[:, cs] = False
        clusters.append(indices[heart])
        nclust += len(leaf)
        progress.update(len(leaf))

np.savetxt(f"monobase-prototypes-{base}-scalevec.txt", scalevecs[clusters], fmt="%.12f")

prototypes = coors[clusters]
np.save(f"monobase-prototypes-{base}.npy", prototypes)
