import sys
import numpy as np
solution_file = sys.argv[1]
poses_outfile =  sys.argv[2]
conformer_outfile =  sys.argv[3]
solutions = np.load(solution_file)
poses = solutions["mat"]
assert poses.ndim == 3 and poses.shape[-2:] == (4,4), poses.shape
conformers = solutions["conformer"]
np.save(poses_outfile, poses)
np.save(conformer_outfile, conformers)
