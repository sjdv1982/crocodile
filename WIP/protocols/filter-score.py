import os
import numpy as np
import argparse
import glob
import sys

parser = argparse.ArgumentParser()
parser.add_argument("target_directory")

args = parser.parse_args()


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


rmsd = np.loadtxt(os.path.join(args.target_directory, "poses.rmsd"))
poses = np.load(os.path.join(args.target_directory, "poses.npy"))
enefiles = glob.glob(os.path.join(args.target_directory, "poses*.ene"))
enefiles = [f for f in enefiles if f.find("poses-filtered") == -1]
if len(enefiles) == 0:
    exit(f"No .ene files in '{args.target_directory}'")
if len(enefiles) > 1:
    exit(f"Multiple .ene files in '{args.target_directory}'")
enefile = enefiles[0]
ene = np.loadtxt(enefile)
assert len(rmsd) == len(ene) == len(poses)
ind = ene.argsort()
rmsd = rmsd[ind]
ene = ene[ind]
poses = poses[ind]

best = rmsd.argmin()
best_ene = ene[best]
mask = ene <= best_ene
print(f"{mask.sum()} / {len(poses)} poses kept")
np.savetxt(
    os.path.join(args.target_directory, "poses-filtered.rmsd"), rmsd[mask], fmt="%.3f"
)
np.savetxt(
    os.path.join(args.target_directory, "poses-filtered.ene"), ene[mask], fmt="%.6f"
)
np.save(os.path.join(args.target_directory, "poses-filtered.npy"), poses[mask])

origin_file = os.path.join(args.target_directory, "poses-origins.txt")
if os.path.exists(origin_file):
    origins = np.loadtxt(origin_file).astype(int)
    assert len(origins) == len(poses)
    origins_filtered = origins[ind][mask]
    np.savetxt(
        os.path.join(args.target_directory, "poses-filtered-origins.txt"),
        origins_filtered,
        fmt="%d",
    )
