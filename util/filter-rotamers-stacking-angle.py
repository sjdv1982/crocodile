import numpy as np
import sys
angle = float(sys.argv[1]) # in degrees
dihedral_min=float(sys.argv[2]) # in degrees; set this and dihedral_max to 0 to disable
dihedral_max=float(sys.argv[3]) # in degrees; set this and dihedral_min to 0 to disable
result_file = sys.argv[4]
angle_files = sys.argv[5:] # in radians
assert len(angle_files)
angles = []
for f in angle_files:
    a = np.load(f)
    assert a.dtype in np.sctypes["float"]
    assert a.ndim == 2
    assert a.shape[-1] == 2, a.shape
    angles.append(a)
    assert a.shape == angles[0].shape, ((f, a.shape), (angle_files[0], angles[0].shape))
n = len(angles[0])
mask = np.ones(n,bool)
for a,f in zip(angles, angle_files):
    m = (a[:, 0] <= angle/180*np.pi)
    if abs(dihedral_min) > 0.0001  and abs(dihedral_max) > 0.0001:
        if dihedral_max < dihedral_min:
            m2 = (a[:, 1] <= dihedral_max/180*np.pi) | (a[:, 1] >= dihedral_min/180*np.pi)
        else:
            m2 = (a[:, 1] >= dihedral_min/180*np.pi) & (a[:, 1] <= dihedral_max/180*np.pi)
        m = m & m2
    print(f"{f}: {m.sum()}/{n} ({m.sum()/n*100:.3f} %) positive", file=sys.stderr)
    mask = mask & m
print(f"Final: {mask.sum()}/{n} ({mask.sum()/n*100:.4f} %) positive", file=sys.stderr)

np.save(result_file, mask)