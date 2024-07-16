import json
import sys
import numpy as np
from tqdm import tqdm

import seamless
seamless.delegate(level=1)

from seamless import transformer, Checksum

@transformer(return_transformation=True)
def collect_mask(conformer_index, mask_chunk, rotamer_matrices):
    import numpy as np
    rotaconformer_dtype = np.dtype([
        ("conformer", np.uint16),
        ("rotamer", np.uint32),
        ("rotmat", float, (3,3)),
    ], align=True)
    mask_matrices = rotamer_matrices[mask_chunk]
    if not len(mask_matrices):
        return []
    result = np.empty(len(mask_matrices), dtype=rotaconformer_dtype)
    result["conformer"] = conformer_index
    result["rotamer"] = np.nonzero(mask_chunk)[0]
    result["rotmat"] = mask_matrices
    return result

rotaconformer_mask_file = sys.argv[1]
rotaconformer_mask = np.load(rotaconformer_mask_file) 
conformers_rota_count_file = sys.argv[2]
conformer_rota_count = np.loadtxt(conformers_rota_count_file,dtype=int)
assert conformer_rota_count.sum() == len(rotaconformer_mask)

rotamer_matrices_index_file = sys.argv[3]
with open(rotamer_matrices_index_file) as f:
    rotamer_matrices_index = json.load(f)

output_file = sys.argv[4]

cumsum = np.cumsum(conformer_rota_count)

def func(n):
    rotamer_matrices = Checksum(rotamer_matrices_index[n])
    if n == 0:
        mask_chunk = rotaconformer_mask[:conformer_rota_count[0]]
    else:
        mask_chunk = rotaconformer_mask[cumsum[n-1]:cumsum[n]]
    return collect_mask(n, mask_chunk, rotamer_matrices)

results = [None] * len(rotamer_matrices_index)

with tqdm(total=len(rotamer_matrices_index)) as progress_bar:
    with seamless.multi.TransformationPool(10) as pool:
        def callback(n, tfm):
            progress_bar.update(1)
            value = tfm.value
            if value is None:
                print(
                    f"""Failure for conformer {n}:
        status: {tfm.status}
        exception: {tfm.exception}
        logs: {tfm.logs}"""
                )
            results[n] = value

        transformations = pool.apply(func, len(rotamer_matrices_index), callback=callback)

if not any([r is None for r in results]):
    results = np.concatenate([r for r in results if len(r)])
    np.save(output_file, results)
