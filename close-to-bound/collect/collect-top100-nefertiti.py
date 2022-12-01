import numpy as np
from nefertiti.functions.parse_pdb import parse_pdb
from nefertiti.functions.write_pdb import write_multi_pdb

pdb = parse_pdb(open("../close-to-bound.pdb").read())
pdb_coors = np.stack((pdb["x"], pdb["y"], pdb["z"]), axis=1)
pdb_coors -= pdb_coors.mean(axis=0)
for d in ("0.5", "1"):
    rotmats = np.load("../close-to-bound-{}A-mat.npy".format(d))
    result = np.zeros((100,) + pdb.shape, dtype=pdb.dtype)
    # do it a bit stupidly
    for n in range(100):
        pdb_rot = pdb_coors.dot(rotmats[n])
        result[n] = pdb
        result[n]["x"] = pdb_rot[:, 0]
        result[n]["y"] = pdb_rot[:, 1]
        result[n]["z"] = pdb_rot[:, 2]
    pdbdata = write_multi_pdb(result)
    open("collect-top100-nefertiti-{}.pdb".format(d), "w").write(pdbdata)