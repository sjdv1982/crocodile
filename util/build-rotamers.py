import numpy as np
from rotamers import get_msd, get_structure_tensor, clusterize
from scipy.spatial.transform import Rotation
import itertools

def estimate_nclust_curve(rmsd, scalevec,  r1size, r2size):
    r1 = Rotation.random(r1size).as_matrix()
    r2 = Rotation.random(r2size).as_matrix()
    result = {}
    for r_rmsd in np.arange(rmsd, max(scalevec), 0.5):
        clusters = clusterize(r1, scalevec, r_rmsd)
        cov = 0
        for rr2 in r2:
            rr2_msd = get_msd(clusters, rr2, scalevec)
            if rr2_msd.min() < (r_rmsd * r_rmsd):
                cov += 1
        coverage = cov / len(r2)
        est_size = len(clusters)/coverage
        print(r_rmsd, len(clusters), coverage, est_size)
        result[r_rmsd] = est_size
    return result

def estimate_rmsd_dist(rmsds, scalevec, rsize):
    r = Rotation.random(rsize).as_matrix()
    dist = {}
    comb = list(itertools.product(rmsds, repeat=2)) 
    comb = np.array(comb)
    comb_rmsds = set(list(comb.sum(axis=1))+ list(rmsds))
    msds = get_msd(r, None, scalevec)
    for rmsd in comb_rmsds:
        count = (msds<(rmsd*rmsd)).sum()
        dist[rmsd] = count/len(msds)
    return dist

def get_best_clustering_hierarchy(est_size, rmsd_dist, rmsd, max_cost_frac):
    """
    At each RMSD r, we have an estimate of:
    - The absolute size S(r) of the total number of clusters 
    - The distribution D(r), describing 
        which percentage of RMSDs between random rotations is within r

    Now we compare:
    - Direct clustering at r1
    - Hierarchical clustering at r2, then evaluating the members of
       relevant clusters at r1
    
    For direct clustering, we evaluate against existing clusters at r1.
    Let's call this cost C.

    For the first step of hierarchical clustering, we evaluate against
    existing clusters at r2 instead of r1. There are fewer of those. 
    The cost of the first step is then S(r2)/S(r1) * C

    For the second step, we need to select relevant clusters. These are
    the ones where the RMSD < r1 + r2, i.e. D(r1+r2) .
    Only this fraction of the existing r1 clusters we evaluate,
      the others (1 - D(r1+r2)) we skip
    The cost of the second step is then D(r1+r2) * C

    The total cost fraction is thus S(r2)/S(r1) + D(r1+r2)

    For the next level of hierarchical clustering,
    we choose r3 so that we can identify the relevant members of r2
    relevant members are in fact those within r1+r2 RMSD.
    So here, the cost fraction becomes S(r3)/S(r2) + D(r1+r2+r3)

    If the cost fraction is less than 1, we save time
    In practice, use the max_cost_frac. Hierarchy costs overhead too!
    In addition, since we cluster top down, 
      some r2 clusters would be split up by r3 clustering. 
    """

    r1 = rmsd
    hierarchy = [r1]
    sumr = r1
    while 1:
        for r2 in sorted(est_size)[1:]:
            if r2 <= r1:
                continue
            if sumr+r2 not in rmsd_dist:
                continue
            cost = est_size[r2]/est_size[r1] + rmsd_dist[sumr+r2]
            #print(r1, r2, cost)
            if cost < max_cost_frac:                
                r1 = r2
                sumr += r1
                hierarchy.append(r1)
                break
        else:
            break
    return hierarchy


def build_rotamers(struc, rmsd):
    import build_rotamer_lib

    struc -= struc.mean(axis=0)
    
    # Get structure tensor
    tensor, scalevec = get_structure_tensor(struc)
    
    r1size, r2size = 1000, 5000
    ### est_size = estimate_nclust_curve(rmsd, scalevec, r1size, r2size) ###
    ###

    r3size = 5000000
    ### rmsd_dist = estimate_rmsd_dist(list(est_size.keys()), scalevec, r3size)

    #for r in sorted(rmsd_dist):
    #    print(r, est_size.get(r), rmsd_dist[r])
    
    ### hierarchy = get_best_clustering_hierarchy(est_size, rmsd_dist, rmsd, 0.3)
    hierarchy = [1.0, 2.0, 3.5, 6.5]
    ###

    print(hierarchy)
 
    r4 = 300000 # TODO: compute
    r4 = 500 ###
    np.random.seed(0)
    r = Rotation.random(r4)
    #print(r[0].as_matrix())
    return build_rotamer_lib.build_rotamers(r.as_matrix(), scalevec, hierarchy)

if __name__ == "__main__": 
    import argparse
    np.random.seed(0)
    parser =argparse.ArgumentParser()
    parser.add_argument("infile", help="Molecule as Mx3 coordinates")
    parser.add_argument(
        "outfile",
        help="Rotations as array of Cx3x3 rotation matrices, where C is the number of rotamers"
    )
    parser.add_argument("--rmsd",type=float,help="RMSD threshold for clustering",required=True)
    args = parser.parse_args()

    struc = np.load(args.infile)
    rotamers = build_rotamers(struc, args.rmsd)
    np.save(args.outfile, rotamers)
