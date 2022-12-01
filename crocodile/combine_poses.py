import numpy as np

def combine_poses(poses, format, *, posefilenames = None, indices=None):
    if posefilenames is not None:
        assert len(posefilenames) == len(poses)
    else:
        posefilenames = range(len(poses))
    
    for arr, posefile in zip(poses, posefilenames):
        if arr.dtype not in (np.dtype(float), np.float32):
            raise TypeError("{}: dtype must be float, not {}".format(posefile, arr.dtype))
        if format == "euler":
            dtype = np.dtype([("index", np.uint32),("euler", arr.dtype, (3,))])
            if arr.ndim != 2 or arr.shape[-1] != 3:
                raise TypeError("{}: shape for euler cannot be {}".format(posefile, arr.shape))
        elif format == "quaternion":
            dtype = np.dtype([("index", np.uint32),("quaternion", arr.dtype, (4,))])
            if arr.ndim != 2 or arr.shape[-1] != 4:
                raise TypeError("{}: shape for quaternion cannot be {}".format(posefile, arr.shape))
        elif format == "matrix":
            if arr.ndim != 3 or arr.shape[-2:] not in ((3,3),(4,4)):
                raise TypeError("{}: shape for matrix cannot be {}".format(posefile, arr.shape))
            dtype = np.dtype([("index", np.uint32),("matrix", arr.dtype, arr.shape[-2:])])
    
    totlen = sum([len(arr) for arr in poses])
    result = np.empty(totlen, dtype=dtype)
    pos = 0
    data_prop = format
    if indices is None:
        indices = list(range(len(poses)))
    for narr, arr in enumerate(poses):
        nextpos = pos + len(arr)
        chunk = result[pos:nextpos]
        chunk["index"] = indices[narr]
        chunk[data_prop] = arr
        pos = nextpos
    return result

if __name__ == "__main__":
    import argparse
    import re
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("format", help="""Format of the input numpy arrays""", 
    choices=["euler", "quaternion", "matrix"])
    p.add_argument("poses", nargs="+", help="""Pose files (numpy format).

The shape must be as follows:
euler: (N, 3)
quaternion: (N, 4)
matrix: (N, 3, 3) or (N, 4, 4)

The dtype must be float or float32"""
    )
    p.add_argument("--indices", required=False, help="""The file index where the pose came from.

If specified, must be a string of numbers separated by spaces or newlines.
By default, indices starting with 0 until (number of files) - 1.""")
    p.add_argument("--output", "-o", required=True, help="""Output file (numpy format).
Output will be a structured numpy array 'arr' where:

    - 'arr["index"]' will contain the file indices  
      (arr[0]["index"] for the first file index)

    - 'arr[data]' will contain the poses 
      (arr[0][data] for the first file pose).
      The 'data' variable is "euler", "quaternion" or "matrix", e.g. arr["euler"]
""") 
    args = p.parse_args()

    poses = []
    for posefile in args.poses:
        arr = np.load(posefile)
        poses.append(arr)
    if args.indices:
        indices = re.split(',|\n| ', args.indices)
        indices = [int(index) for index in indices if len(index)]
    else:
        indices = None
    result = combine_poses(poses, args.format, posefilenames=args.poses, indices=indices)
    np.save(args.output, result)