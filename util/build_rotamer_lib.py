# to be ported to C

import numpy as np
from scipy.spatial.transform import Rotation


def get_axis_angle(mat):
    """From https://github.com/rock-learning/pytransform3d/blob/master/pytransform3d/rotations.py""" 
    trace = mat[0][0] + mat[1][1] + mat[2][2]
    angle = np.arccos((trace - 1.0) / 2.0)
    axis = np.empty(3)

    if abs(angle) < 1e-12:  # mat == np.eye(3)
        axis[:] = 1,0,0
        angle = 0
    else:

        # We can usually determine the rotation axis by inverting Rodrigues'
        # formula. Subtracting opposing off-diagonal elements gives us
        # 2 * sin(angle) * e,
        # where e is the normalized rotation axis.
        axis_unnormalized = np.array([
            mat[2, 1] - mat[1, 2], 
            mat[0, 2] - mat[2, 0], 
            mat[1, 0] - mat[0, 1]
        ])

        if abs(angle - np.pi) < 1e-4:  # np.trace(mat) close to -1
            # The threshold is a result from this discussion:
            # https://github.com/rock-learning/pytransform3d/issues/43
            # The standard formula becomes numerically unstable, however,
            # Rodrigues' formula reduces to mat = I + 2 (ee^T - I), with the
            # rotation axis e, that is, ee^T = 0.5 * (mat + I) and we can find the
            # squared values of the rotation axis on the diagonal of this matrix.
            # We can still use the original formula to reconstruct the signs of
            # the rotation axis correctly.
            axis[0] = np.sqrt(0.5 * (mat[0][0] + 1.0)) * np.sign(axis_unnormalized[0])
            axis[1] = np.sqrt(0.5 * (mat[1][1] + 1.0)) * np.sign(axis_unnormalized[1])
            axis[2] = np.sqrt(0.5 * (mat[2][2] + 1.0)) * np.sign(axis_unnormalized[2])
            #axis[:] = np.sqrt(0.5 * (np.diag(mat) + 1.0)) * np.sign(axis_unnormalized)
        else:
            axis[:] = axis_unnormalized
            # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
            # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
            # but the following is much more precise for angles close to 0 or pi:
        axis /= np.linalg.norm(axis)

    # Check
    r = Rotation.from_matrix(mat)
    axis2 =  r.as_rotvec()
    angle2 = np.linalg.norm(axis2)
    axis2 /= angle2    
    assert np.isclose(angle, angle2)
    if np.abs(angle) > 0.1:
        assert np.isclose(np.abs(np.dot(axis, axis2)), 1), (axis, axis2)
    # /Check

    return axis, angle

def dif_matrix(mat1, mat2):
    
    result = np.zeros((3,3))
    result[0][0] = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[0][1] + mat1[0][2] * mat2[0][2]
    result[0][1] = mat1[0][0] * mat2[1][0] + mat1[0][1] * mat2[1][1] + mat1[0][2] * mat2[1][2]
    result[0][2] = mat1[0][0] * mat2[2][0] + mat1[0][1] * mat2[2][1] + mat1[0][2] * mat2[2][2]
    
    result[1][0] = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[0][1] + mat1[1][2] * mat2[0][2]
    result[1][1] = mat1[1][0] * mat2[1][0] + mat1[1][1] * mat2[1][1] + mat1[1][2] * mat2[1][2]
    result[1][2] = mat1[1][0] * mat2[2][0] + mat1[1][1] * mat2[2][1] + mat1[1][2] * mat2[2][2]    
    
    result[2][0] = mat1[2][0] * mat2[0][0] + mat1[2][1] * mat2[0][1] + mat1[2][2] * mat2[0][2]
    result[2][1] = mat1[2][0] * mat2[1][0] + mat1[2][1] * mat2[1][1] + mat1[2][2] * mat2[1][2]
    result[2][2] = mat1[2][0] * mat2[2][0] + mat1[2][1] * mat2[2][1] + mat1[2][2] * mat2[2][2]     
    
    # Check
    result2 = mat1.dot(mat2.T)
    assert np.allclose(result, result2), (result, result2) 
    # /Check
    
    return result

def get_msd(dmat, scalevec):
    ax, ang = get_axis_angle(dmat)
    fac = (np.cos(ang)-1)**2+np.sin(ang)**2
    cross = (scalevec * scalevec) * (1 - ax*ax)
    msd = fac * cross.sum()
    return msd

def is_close_to_cluster(
    mat, matnr, random_mats, 
    clusters, indices, 
    curr_cluster, 
    scalevec, cum_thresholds,
    hierarchy_pos
):
    c_matindex, c_offset, c_size, _ = clusters[curr_cluster]
    #print("ISCL ", hierarchy_pos, curr_cluster, c_size)
    if curr_cluster == 0:
        assert hierarchy_pos == -1
    else:
        assert hierarchy_pos >= 0
        c_mat = random_mats[c_matindex]
        dmat = dif_matrix(mat, c_mat)
        msd = get_msd(dmat, scalevec)
        #print("MSD1", hierarchy_pos, msd, cum_thresholds[hierarchy_pos])
        if msd > cum_thresholds[hierarchy_pos]:
            return False
    if (hierarchy_pos == len(cum_thresholds) - 2):
        for childnr in range(c_size):
            child = indices[c_offset + childnr]
            c_mat = random_mats[child]
            dmat = dif_matrix(mat, c_mat)
            msd = get_msd(dmat, scalevec)
            #print("MSD2", len(cum_thresholds)-1, msd, cum_thresholds[-1], childnr, child)
            if (msd < cum_thresholds[-1]): 
                return True
        return False
    
    for childnr in range(c_size):
        ind = indices[c_offset + childnr]
        result = is_close_to_cluster(
            mat, matnr, random_mats, 
            clusters, indices,
            ind, 
            scalevec, cum_thresholds, 
            hierarchy_pos+1
        )
        if result:
            return True
        c_matindex, c_offset, c_size, _ = clusters[curr_cluster]
    return False

def add_to_cluster(clus, index, indices, nindices):
    c_matindex, c_offset, c_size, c_maxsize = clus
    if c_size == c_maxsize:
        new_maxsize = 2 * c_maxsize
        indices[nindices:nindices+c_size] = indices[c_offset:c_offset+c_size]
        c_offset = clus[1] = nindices  # c_offset
        clus[3] = new_maxsize  # c_maxsize
        nindices += new_maxsize
    indices[c_offset + c_size] = index
    clus[2] += 1  # c_size
    return nindices

def new_cluster(matindex, clusters, nclusters, indices, nindices):
    ini_size = 10
    clusters[nclusters] = matindex, nindices, 0, ini_size
    nclusters += 1
    nindices += ini_size 
    return nclusters, nindices

def new_clusters(matindex, clusters, nclusters, indices, nindices, nest):
    nclusters, nindices = new_cluster(matindex, clusters, nclusters, indices, nindices)
    clus = clusters[nclusters-1]
    if nest == 1:
        print("ADD  ", 1, nclusters-1, matindex, clusters[nclusters-1,2]+1)
        nindices = add_to_cluster(clus, matindex, indices, nindices)
        return nclusters, nindices
    print("PRE  ", nest, nclusters-1, nclusters)
    nindices = add_to_cluster(clus, nclusters, indices, nindices)
    return new_clusters(matindex, clusters, nclusters, indices, nindices, nest-1)

def print_clust(clusnr, clusters, indices, hierarchy_pos, hierarchy):
    c_matindex, c_offset, c_size, _ = clusters[clusnr]
    i = indices[c_offset: c_offset+c_size]
    if hierarchy_pos == len(hierarchy)-1:
        ind = " ".join(["M" + str(ii) for ii in i])
    else:
        ind = " ".join(["C" + str(ii) for ii in i])        
    print(
        hierarchy_pos+1, hierarchy[-hierarchy_pos-1], 
        "C"+str(clusnr), c_matindex, ind
    )
    if hierarchy_pos < len(hierarchy)-1:
        for child in i:
            print_clust(child, clusters, indices, hierarchy_pos+1, hierarchy)

def insert_in_clustering(
    matnr, mat, random_mats, 
    clusters, nclusters, 
    indices, nindices, 
    curr_cluster, 
    scalevec, thresholds,
    hierarchy_pos
):
    c_matindex, c_offset, c_size, _ = clusters[curr_cluster]
    if curr_cluster == 0:
        assert hierarchy_pos == -1
    else:
        assert hierarchy_pos >= 0
        c_mat = random_mats[c_matindex]
        dmat = dif_matrix(mat, c_mat)
        msd = get_msd(dmat, scalevec)
        #print("MSD3", hierarchy_pos, msd, thresholds[hierarchy_pos], curr_cluster, c_matindex)
        if msd > thresholds[hierarchy_pos]:
            return nclusters, nindices, False

    found = False
    if (hierarchy_pos == len(thresholds) - 2):
        print("ADD2 ", curr_cluster, matnr, clusters[curr_cluster,2]+1)
        nindices = add_to_cluster(clusters[curr_cluster], matnr, indices, nindices)
    else:
        for childnr in range(c_size):
            ind = indices[c_offset + childnr]
            nclusters, nindices, result = insert_in_clustering(
                matnr, mat, 
                random_mats, 
                clusters, nclusters, 
                indices, nindices,
                ind, 
                scalevec, thresholds,
                hierarchy_pos+1
            )
            if result:
                return nclusters, nindices, True
        print("PRE2 ", curr_cluster, nclusters)
        nindices = add_to_cluster(clusters[curr_cluster], nclusters, indices, nindices)
        nclusters, nindices = new_clusters(
            matnr,
            clusters, nclusters,
            indices, nindices,
            len(thresholds) - 2 - hierarchy_pos
        )
    return nclusters, nindices, True

def build_rotamers(random_mats, scalevec, hierarchy):
    max_clusters = int(1e07)
    max_indices = int(1e08)
    clusters = np.zeros((max_clusters, 4), np.uint32) # matindex, offset, size, maxsize
    nclusters = 0
    indices = np.zeros(max_indices, np.uint32)
    nindices = 0
    nclusters, nindices = new_cluster(0, clusters, nclusters, indices, nindices)
    thresholds = np.array(hierarchy)**2
    cum_thresholds = np.cumsum(hierarchy)**2
    thresholds = thresholds[::-1]
    cum_thresholds = cum_thresholds[::-1]
    print(thresholds, cum_thresholds)
    for matnr in range(len(random_mats)):
        print("CONF ", matnr)
        mat = random_mats[matnr]
        close = is_close_to_cluster(
            mat, matnr, random_mats,
            clusters, indices,
            0, scalevec,
            cum_thresholds, -1
        )
        print("CLOSE", matnr, int(close))
        print("STAT", nclusters, nindices)
        if not close:
            nclusters, nindices, _ = insert_in_clustering (
                matnr, mat, random_mats,
                clusters, nclusters,
                indices, nindices,
                0, scalevec,
                thresholds, -1
            )
        #print_clust(0, clusters, indices, 0, hierarchy)
        #print()
    #print_clust(0, clusters, indices, 0, hierarchy)
