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

s1 = set(poses1["conformer"])
s2 = set(poses2["conformer"])
common_conformers = s1.intersection(s2)
mask1 = np.array([pose["conformer"] in common_conformers for pose in poses1], bool)
mask2 = np.array([pose["conformer"] in common_conformers for pose in poses2], bool)

keepmask1 = ind1 & mask1
keepmask2 = ind2 & mask2

print(
    len(poses1),
    len(poses2),
    ind1.sum(),
    ind2.sum(),
    "common conformers:",
    len(common_conformers),
    "merged:",
    keepmask1.sum(),
    keepmask2.sum(),
)

np.savetxt(index_file1, np.where(keepmask1)[0] + 1, fmt="%d")
np.savetxt(index_file2, np.where(keepmask2)[0] + 1, fmt="%d")
