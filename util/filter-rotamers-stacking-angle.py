import numpy as np
import sys
angle = float(sys.argv[1]) # in degrees
rad = angle/180 * np.pi
ligand_rotaconformers_file = sys.argv[2]
ligand_rotaconformers = np.load(ligand_rotaconformers_file)
result_file = sys.argv[3]
angle_files = sys.argv[4:] # in radians
assert len(angle_files)
angles = []
for f in angle_files:
    a = np.load(f)
    assert a.dtype in np.sctypes["float"]
    assert a.shape == (len(ligand_rotaconformers),), a.shape
    angles.append(a)
mask = np.ones(len(ligand_rotaconformers),bool)
for a,f in zip(angles, angle_files):
    m = (a <= rad)
    print(f"{f}: {m.sum()}/{len(ligand_rotaconformers)} ({m.sum()/len(ligand_rotaconformers)*100:.2f} %) positive", file=sys.stderr)
    mask = mask & m
print(f"Final: {mask.sum()}/{len(ligand_rotaconformers)} ({mask.sum()/len(ligand_rotaconformers)*100:.2f} %) positive", file=sys.stderr)

results = ligand_rotaconformers[mask]
np.save(result_file, results)