
import numpy as np


def _marsaglia_pairs(n):
    """Generate n pairs x,y uniformly in (-1..1) where x²+y² < 1"""    
    nn = (1 / (np.pi / 4)) * n  # best estimate of how many points we need
    nn += 0.1 * n + 1000  # just to be sure
    while 1:  # just to be really really sure
        pairs = np.random.random_sample(size=(int(nn), 2)) * 2 - 1
        z = pairs[:, 0] ** 2 + pairs[:, 1] ** 2
        pairs_ok = pairs[z < 1]
        if len(pairs_ok) >= n:
            return pairs_ok[:n], z[z<1][:n]

def random_quaternions(n, seed=0):
    #  Generate random quaternions using Marsaglia's method
    np.random.seed(seed)
    xy, z = _marsaglia_pairs(n)
    uv, w = _marsaglia_pairs(n)
    s = np.sqrt((1-z)/w)
    q = np.empty((n, 4))
    q[:, :2] = xy
    q[:, 2:] = s[:, None] * uv
    return q

def quaternions_to_3x3(q):
    """From:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    m11 = 1 - 2*qy2 - 2*qz2
    m12 = 2*qx*qy - 2*qz*qw
    m13 = 2*qx*qz + 2*qy*qw
    m21 = 2*qx*qy + 2*qz*qw
    m22 = 1 - 2*qx2 - 2*qz2
    m23 = 2*qy*qz - 2*qx*qw
    m31 = 2*qx*qz - 2*qy*qw
    m32 = 2*qy*qz + 2*qx*qw
    m33 = 1 - 2*qx2 - 2*qy2
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = m11
    m[:, 0, 1] = m12
    m[:, 0, 2] = m13
    m[:, 1, 0] = m21
    m[:, 1, 1] = m22
    m[:, 1, 2] = m23
    m[:, 2, 0] = m31
    m[:, 2, 1] = m32
    m[:, 2, 2] = m33
    return m


