import sys, os
import numpy as np
from scipy.spatial import KDTree
fragdir1 = sys.argv[1]
fragdir2 = sys.argv[2]
f1 = f"{fragdir1}/poses-filtered.npy"
f2 = f"{fragdir2}/poses-filtered.npy"
offset1 = np.load(f1)["offset"]
offset2 = np.load(f2)["offset"]
tree1 = KDTree(offset1)
tree2 = KDTree(offset2)
for r in (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9):
    print("R =", r)
    contacts = tree1.query_ball_tree(tree2, r)
    ncontacts = sum([len(l) for l in contacts])
    if ncontacts:
        break
else:
    print("Nothing under 9A")
    exit(1)
col1 = np.repeat(np.arange(len(offset1)), [len(s) for s in contacts]).astype(int)
col2 = np.concatenate(contacts).astype(int)
d = ((offset1[col1] - offset2[col2])**2).sum(axis=1)
print("Closest COM:", np.sqrt(d.min()))
b = d.argmin(); print(col1[b], offset1[col1][b], col2[b], offset2[col2][b], d[b]**0.5)