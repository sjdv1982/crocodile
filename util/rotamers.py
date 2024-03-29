import numpy as np, sys

from scipy.spatial.transform import Rotation

def get_structure_tensor(struc):
    struc0 = struc - struc.mean(axis=0)
    v, s, wt = np.linalg.svd(struc0) 
    scalevec = s/np.sqrt(len(struc))
    tensor = wt.T
    if np.linalg.det(tensor) < 0:
        tensor[2] *= -1
    assert np.linalg.det(tensor) > 0.999
    return tensor, scalevec


def get_msd(mats, refe, scalevec):
    if refe is None:
        rrmats = mats
    else:
        rrmats = np.einsum("ikl,ml->ikm",mats, refe)
        # broadcasted form of mats[i].dot(refe.T)

    rr = Rotation.from_matrix(rrmats)
    ax =  rr.as_rotvec()
    ang = np.linalg.norm(ax,axis=1)
    ax /= ang[:, None]
    fac = (np.cos(ang)-1)**2+np.sin(ang)**2
    cross = (scalevec * scalevec) * (1 - ax*ax)
    msd = fac * cross.sum(axis=-1)
    return msd

def clusterize(matrices, scalevec, rmsd):
    cluster_storage = np.zeros((1000,3,3))
    clusters = cluster_storage[:1]
    clusters[0] = matrices[0]
    curr = matrices[1:]
    cluspos = 0
    while len(curr):
        msd = get_msd(curr, clusters[cluspos], scalevec)   
        curr = curr[msd>(rmsd*rmsd)]
        cluspos += 1
        if cluspos == len(clusters):
            if not len(curr):
                break
            if len(clusters) == len(cluster_storage):
                new_cluster_storage = np.zeros((int(1.1*len(cluster_storage)), 3, 3))
                new_cluster_storage[:len(cluster_storage)] = cluster_storage
                cluster_storage = new_cluster_storage
            cluster_storage[len(clusters)] = curr[0]
            clusters = cluster_storage[:len(clusters)+1]
            curr = curr[1:] 
    return clusters

if __name__ == "__main__":
    import argparse
    np.random.seed(0)
    parser =argparse.ArgumentParser()
    parser.add_argument("infile", help="Molecule as Mx3 coordinates")
    parser.add_argument(
        "outfile",
        help="Rotations as array of Cx3x3 rotation matrices, where C is the number of rotamers"
    )
    parser.add_argument("--nstruc",type=int,help="Number of random orientations to sample and cluster",required=True)
    parser.add_argument("--rmsd",type=float,help="RMSD threshold for clustering",required=True)
    args = parser.parse_args()

    struc = np.load(args.infile)
    struc -= struc.mean(axis=0)

    v, s, wt = np.linalg.svd(struc)
    scale = s/np.sqrt(len(struc))
    strucp = struc.dot(wt.T)  
    r = Rotation.random(args.nstruc)
    rmat = r.as_matrix()
    chunksize = 2000

    cluster_storage = np.zeros((1000,3,3))
    clusters = cluster_storage[:1]
    clusters[0] = rmat[0]
    thresholdsq = args.rmsd**2
    for n in range(1, args.nstruc, chunksize):
        chunk = rmat[n:n+chunksize]        
        cluspos = 0
        print(n,len(clusters))
        while 1:
            #print(n,len(chunk), cluspos, len(clusters))
            rrmat = np.einsum("ikl,ml->ikm",chunk, clusters[cluspos])
            # broadcasted form of chunk[i].dot(clusters[cluspos].T)

            rr = Rotation.from_matrix(rrmat)
            ax =  rr.as_rotvec()
            ang = np.linalg.norm(ax,axis=1)
            ax /= ang[:, None]
            fac = (np.cos(ang)-1)**2+np.sin(ang)**2
            cross = (scale * scale) * (1 - ax*ax)
            msd = fac * cross.sum(axis=-1)
        
            chunk = chunk[msd>thresholdsq]
            cluspos += 1
            if cluspos == len(clusters):
                if not len(chunk):
                    break
                if len(clusters) == len(cluster_storage):
                    new_cluster_storage = np.zeros((int(1.1*len(cluster_storage)), 3, 3))
                    new_cluster_storage[:len(cluster_storage)] = cluster_storage
                    cluster_storage = new_cluster_storage
                cluster_storage[len(clusters)] = chunk[0]
                clusters = cluster_storage[:len(clusters)+1]
                chunk = chunk[1:] 
    
 
    np.save(args.outfile, clusters)