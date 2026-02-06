import numpy as np
import sys
from poses import open_pose_array

poses_npy = sys.argv[1]  # BytesIO
outfile = sys.argv[2]
# poses = np.load(poses_npy)
poses = open_pose_array(poses_npy)  # zstd support
np.save(outfile, poses[:, 0])
