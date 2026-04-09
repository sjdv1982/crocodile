import sys, os
import numpy as np

threshold = float(sys.argv[1])
energy_files = sys.argv[2:-1]
output_file = sys.argv[-1]
assert not os.path.exists(output_file)

offset = 0
inds = []
for f in energy_files:
    print(f)
    ene = np.load(f)
    assert ene.ndim == 1
    curr_inds = np.where(ene <= threshold)[0]
    if len(curr_inds):
        inds.append(curr_inds + offset)
    offset += len(ene)
if len(inds):
    inds = np.concatenate(inds)
    np.save(output_file, inds)
    print(f"{len(inds)}/{offset} indices kept")
