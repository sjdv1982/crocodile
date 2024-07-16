import numpy as np
import sys

from tqdm import tqdm

roco_dtype = np.dtype([
    ("conformer", np.uint16),
    ("rotamer", np.uint32),
    ("mat", float, (4,4)),
], align=True)

solutions = np.load(sys.argv[1])
if len(solutions) > 50 * 10**6:
    print("More than 50 million solutions. Use gather-roco-chunk instead.", file=sys.stderr)
    exit(1)

offsets = np.load(sys.argv[2])

rotaconformers = np.load(sys.argv[3])

roco_outfile = sys.argv[4]

roco = np.zeros(len(solutions),dtype=roco_dtype)
chunklen = 100000
for n in tqdm(range(0, len(solutions), chunklen)):
    solutions_chunk = solutions[n:n+chunklen]
    offsets_chunk = offsets[solutions_chunk[:, 0]]
    rotaconformers_chunk = rotaconformers[solutions_chunk[:, 1]]
    roco_chunk = roco[n:n+chunklen]
    roco_chunk["conformer"] = rotaconformers_chunk["conformer"]
    roco_chunk["rotamer"] = rotaconformers_chunk["rotamer"]
    roco_chunk["mat"][:, :3, :3] = rotaconformers_chunk["rotmat"]
    roco_chunk["mat"][:, 3, :3] = offsets_chunk
    roco_chunk["mat"][:, 3, 3] = 1

np.save(roco_outfile, roco)