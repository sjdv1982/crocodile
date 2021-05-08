import numpy as np, sys
from fastcluster import cluster

np.random.seed(0)

def multiply(q1, q2):
    if q1.ndim == 1:
        q1 = q1.reshape(1, 4)
    if q2.ndim == 1:
        q2 = q2.reshape(1, 4)
    m = q2[:,_mul_fields] * _mul_signs
    w = (q1[:] * m[:,0]).sum(axis=-1)
    return (q1[:,None,:] * m).sum(axis=2)

def marsaglia_pairs(n):
    """Generate n pairs x,y uniformly in (-1..1) where x²+y² < 1"""
    nn = (1 / (np.pi / 4)) * n  # best estimate of how many points we need
    nn += 0.1 * n + 1000  # just to be sure
    while 1:  # just to be really really sure
        pairs = np.random.random_sample(size=(int(nn), 2)) * 2 - 1
        z = pairs[:, 0] ** 2 + pairs[:, 1] ** 2
        pairs_ok = pairs[z < 1]
        if len(pairs_ok) >= n:
            return pairs_ok[:n], z[z<1][:n]

def random_quaternions(n):
    #  Generate random quaternions using Marsaglia's method
    xy, z = marsaglia_pairs(n)
    uv, w = marsaglia_pairs(n)
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

import json

if __name__ == "__main__":
    import argparse
    parser =argparse.ArgumentParser()
    parser.add_argument("infile", help="Prototype conformers as NxMx3 coordinates")
    parser.add_argument(
        "outfile",
        help="Rotations as an array of N arrays of Cx3x3 rotation matrices, where C is the number of clusters for that prototype"
    )
    parser.add_argument("--nstruc",type=int,help="Number of random orientations to sample and cluster",required=True)
    parser.add_argument("--rmsd",type=float,help="RMSD threshold for clustering",required=True)
    parser.add_argument("--maxangle",type=float,help="Restrict rotations to a maximum angle (in degrees) from the origin")
    parser.add_argument("--chunksize",type=int,help="Chunk size for RMSD calculation (hyperparameter)",default=50)
    args = parser.parse_args()

    protos = np.load(args.infile)
    clusters = np.empty(len(protos), dtype=object)
    assert len(protos) == 1 # TODO: rocs!
    for n in range(len(protos)):
        print(n+1,  len(protos), file=sys.stderr)
        proto = protos[n]
        q0 = []
        nstruc = 0
        while nstruc < args.nstruc:
            if nstruc > 0:
                print("random quaternions %d/%d" % (nstruc, args.nstruc), file=sys.stderr)
            gen = 5000000
            if args.maxangle is None:
                gen = args.nstruc
            q = random_quaternions(gen)
            angle = 2 * np.arccos(q[:, 0]) / np.pi * 180
            if args.maxangle is not None:
                q = q[angle < args.maxangle]
            nstruc = sum([len(q) for q in q0])
            q0.append(q)
        q = np.concatenate(q0)[:args.nstruc]
        m = quaternions_to_3x3(q)
        struc = np.einsum("jk,ikl->ijl", proto, m) #broadcasted form of proto.dot(m)
        clus, _ = cluster(struc, args.rmsd, 0, args.chunksize, False)
        clus = np.array(list(clus.keys()))
        clusters = m[clus]
        print(clusters.shape)
    np.save(args.outfile, clusters)