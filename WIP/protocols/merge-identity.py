import sys
import os
import numpy as np

directory1 = sys.argv[1]
directory2 = sys.argv[2]

poses1 = np.load(os.path.join(directory1, "poses-filtered.npy"))
index_file1 = os.path.join(directory1, "prop-indices.txt")
indices1 = None
if os.path.exists(index_file1):
    indices1 = np.loadtxt(index_file1).astype(np.uint)

poses2 = np.load(os.path.join(directory2, "poses-filtered.npy"))
index_file2 = os.path.join(directory2, "prop-indices.txt")
indices2 = None
if os.path.exists(index_file2):
    indices2 = np.loadtxt(index_file2).astype(np.uint)

if indices1 is None:
    ind1 = np.ones(len(poses1), bool)
else:
    ind1 = np.zeros(len(poses1), bool)
    ind1[indices1] = 1

if indices2 is None:
    ind2 = np.ones(len(poses2), bool)
else:
    ind2 = np.zeros(len(poses2), bool)
    ind2[indices2] = 1

map1 = {}
map2 = {}
gridspacing = np.sqrt(3) / 3
for poses, map_, ind in ((poses1, map1, ind1), (poses2, map2, ind2)):
    for pnr, p in enumerate(poses):
        if not ind[pnr]:
            continue
        rotamer, conformer, offset = p["rotamer"], p["conformer"], p["offset"]
        offset = tuple(np.round(offset / gridspacing).astype(int).tolist())
        key = conformer, rotamer, offset
        if key not in map_:
            map_[key] = []
        map_[key].append(pnr)
merged = set(map1.keys()).intersection(set(map2.keys()))
mergedmap1 = {k: map1[k] for k in merged}
mergedmap2 = {k: map2[k] for k in merged}
keep1 = sum(mergedmap1.values(), [])
keepmask1 = np.zeros(len(poses1), bool)
keepmask1[keep1] = 1
keep2 = sum(mergedmap2.values(), [])
keepmask2 = np.zeros(len(poses2), bool)
keepmask2[keep2] = 1

print(
    len(poses1),
    len(poses2),
    ind1.sum(),
    ind2.sum(),
    "merged:",
    len(keep1),
    len(keep2),
)

ind1 &= keepmask1
ind2 &= keepmask2
print(ind1.sum(), ind2.sum())
np.savetxt(index_file1, np.where(ind1)[0] + 1, fmt="%d")
np.savetxt(index_file2, np.where(ind2)[0] + 1, fmt="%d")

mergedmap1keys = sorted(list(mergedmap1.keys()))
mergedmap2keys = sorted(list(mergedmap2.keys()))

rev_mergedmap1 = {}
for knr, k in enumerate(mergedmap1keys):
    vs = mergedmap1[k]
    for v in vs:
        rev_mergedmap1[v] = knr + 1

rev_mergedmap2 = {}
for knr, k in enumerate(mergedmap2keys):
    vs = mergedmap2[k]
    for v in vs:
        rev_mergedmap2[v] = knr + 1


hash_file1 = os.path.join(directory1, "prop-indices-hash.txt")
with open(hash_file1, "w") as f:
    for k in np.where(ind1)[0]:
        print(rev_mergedmap1[k], file=f)

hash_file2 = os.path.join(directory2, "prop-indices-hash.txt")
with open(hash_file2, "w") as f:
    for k in np.where(ind2)[0]:
        print(rev_mergedmap2[k], file=f)
