import numpy as np
import sys

from tqdm import tqdm

roco_dtype = np.dtype([
    ("conformer", np.uint16),
    ("rotamer", np.uint32),
    ("mat", float, (4,4)),
], align=True)

solutions = np.load(sys.argv[1])

offsets = np.load(sys.argv[2])

rotaconformers = np.load(sys.argv[3])

chunklen = int(sys.argv[4])

roco_outfile_pattern = sys.argv[5]

roco_buf = np.zeros(chunklen,dtype=roco_dtype)
count = 0
for n in tqdm(range(0, len(solutions), chunklen)):
    count += 1
    solutions_chunk = solutions[n:n+chunklen]
    offsets_chunk = offsets[solutions_chunk[:, 0]]
    rotaconformers_chunk = rotaconformers[solutions_chunk[:, 1]]
    roco = roco_buf[:len(solutions_chunk)]
    roco["conformer"] = rotaconformers_chunk["conformer"]
    roco["rotamer"] = rotaconformers_chunk["rotamer"]
    roco["mat"][:, :3, :3] = rotaconformers_chunk["rotmat"]
    roco["mat"][:, 3, :3] = offsets_chunk
    roco["mat"][:, 3, 3] = 1

    np.save(f"{roco_outfile_pattern}-{count}", roco)

with open(roco_outfile_pattern, "w") as f:
    print(count, file=f)    