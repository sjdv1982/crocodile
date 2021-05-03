
"""
Adapted from fastcluster-npy.py from ATTRACT
copyright 2016-2017 Sjoerd de Vries, Isaure Chauvot de Beauchene
GPL license
"""

import sys
import numpy as np

def cluster(structures, threshold, already_clustered, chunksize, assign_structures):
    """Clusters structures using an RMSD threshold
    First structure becomes a cluster,
     second structure only if it doesn't cluster with the first, etc.

    structures: 2D numpy array, second dimension = 3 * natoms
      structures must already have been fitted!
    threshold: RMSD threshold (A)
    already_clustered: if nonzero, the first already_clustered structures are
     considered clusters
    chunksize: number of structures to put in a chunk
      This is an implementation detail that only affects the speed, not the result

    """
    if len(structures.shape) == 3:
        assert structures.shape[2] == 3
        structures = structures.reshape(structures.shape[0], structures.shape[1]*3)
    if len(structures.shape) == 2:
        assert structures.shape[1] % 3 == 0
    natoms = structures.shape[1]/3
    # threshold2 = sum-of-sd threshold = (RMSD threshold **2) * natoms
    threshold2 = threshold**2 * natoms

    nclus = 1
    assert already_clustered >= 0
    if already_clustered == 0:
        already_clustered = 1 ## the first structure is always a cluster
    clus_space = 99 + already_clustered
    clus = np.zeros((clus_space, structures.shape[1]))
    clus[:already_clustered] = structures[:already_clustered]
    clustids = list(range(already_clustered))
    for n in range(already_clustered, len(structures), chunksize):
        if (n-1) % 100 == 0:
            print("{0}/{1} {2}".format(n, len(structures), nclus), file=sys.stderr)

        if chunksize == 1:
            d = structures[n][None, :] - clus
            inter_sd = np.einsum("ij,ij->i", d, d)
            closest_inter = inter_sd.min()
            if closest_inter > threshold2:
                if nclus == clus_space:
                    clus_space = int(clus_space*1.5)
                    clus_old = clus
                    clus = np.zeros((clus_space, structures.shape[1]))
                    clus[:nclus] = clus_old
                clus[nclus] = structures[n]
                clustids.append(n)
                nclus += 1
            continue

        chunk = structures[n:n+chunksize]
        d = chunk[:, np.newaxis, :] - clus[np.newaxis, :nclus, :]
        inter_sd = np.einsum("...ij,...ij->...i", d, d)
        #close_inter is a 2D Boolean matrix:
        #  True  (1): chunk[i] is close to (within RMSD threshold of) clus[j]
        #  False (0): chunk[i] is not close to clus[j]
        close_inter = (inter_sd < threshold2)

        # newclustered contains all structures in the chunk that *don't* cluster with an existing cluster
        newclustered = []
        for chunk_index, closest_inter in enumerate(np.argmax(close_inter,axis=1)):
            # closest_inter contains the *first* index of close_inter
            #   with the highest value of close_inter
            # We are interested in the case where close_inter is all False (=> new cluster)
            # In that case, the highest value of close_inter is False, and closest_inter is 0
            # If close_inter is *not* all False (=> existing cluster), one of these conditions is False
            if closest_inter == 0 and close_inter[chunk_index, 0] == False:
                newclustered.append(chunk_index)

        if len(newclustered):
            # Now we have newclustered: the *chunk* index of all structures in the chunk that will be in new clusters
            # Now we want to cluster them among themselves, and add the *structure* id of each new cluster
            chunk_newclustered = chunk[newclustered]
            d = chunk_newclustered[:, np.newaxis, :] - chunk_newclustered[np.newaxis, :, :]
            intra_sd = np.einsum("...ij,...ij->...i", d, d)
            close_intra = (intra_sd < threshold2)

            # set all upper-triangular indices to False
            close_intra[np.triu_indices(len(chunk_newclustered))] = 0
            for nn in range(len(chunk_newclustered)):
                # same logic as for closest_inter;
                #  except that we don't have the chunk index, but the chunk_newclustered index (nn)
                #  and, since we modify close_intra in the "else" clause, argmax is computed later
                closest_intra = np.argmax(close_intra[nn])
                if closest_intra == 0 and close_intra[nn, 0] == False:
                    chunk_index = newclustered[nn]
                    # if clus is full, re-allocate it as a 50 % larger array
                    if nclus == clus_space:
                        clus_space = int(clus_space*1.5)
                        clus_old = clus
                        clus = np.zeros((clus_space, structures.shape[1]))
                        clus[:nclus] = clus_old
                    clus[nclus] = chunk[chunk_index]
                    clustids.append(n+chunk_index)
                    nclus += 1
                else:  # in addition, if we aren't a new cluster, being close to us doesn't matter
                    close_intra[:, nn] = False

    # After assigning the cluster centers,
    #  assign all structures to the closest cluster
    clusters = {a:[a] for a in clustids}
    if assign_structures:
        for n in range(0, len(structures), chunksize):
            chunk = structures[n:n+chunksize]
            d = chunk[:, np.newaxis, :] - clus[np.newaxis, :, :]
            inter_sd = np.einsum("...ij,...ij->...i", d,d)
            best = np.argmin(inter_sd, axis=1)
            for nn in range(len(chunk)):
                bestclust = clustids[best[nn]]
                if bestclust == (n+nn):
                    continue
                clusters[bestclust].append(n+nn)

    return clusters, clustids
