import numpy as np, sys

from scipy.spatial.transform import Rotation

'''
def get_structure_tensor(struc):
    struc0 = struc - struc.mean(axis=0)
    v, s, wt = np.linalg.svd(struc0) 
    scalevec = s/np.sqrt(len(struc))
    tensor = wt.T
    if np.linalg.det(tensor) < 0:
        tensor[2] *= -1
    assert np.linalg.det(tensor) > 0.999
    return tensor, scalevec
'''

def get_structure_tensor(conf):
    curr_tensor = np.eye(3)
    niter = 0
    while 1:
        conft = conf.dot(curr_tensor)
        
        conf0 = conft - conft.mean(axis=0)
        v, s, wt = np.linalg.svd(conf0) 
        scalevec = s/np.sqrt(len(conf))
        tensor = wt.T
        if np.linalg.det(tensor) < 0:
            tensor[2] *= -1
        assert np.linalg.det(tensor) > 0.999

        curr_tensor = curr_tensor.dot(tensor)
        assert np.linalg.det(curr_tensor) > 0.999

        if np.abs(tensor - np.eye(3)).sum() < 0.01:
            break
        niter += 1
        if niter > 1000:
            if (np.abs(tensor) - np.eye(3)).sum() < 0.01:
                break

        if niter > 10000:
            print(niter, np.abs(tensor - np.eye(3)).sum(), tensor, curr_tensor)
        if niter > 10010:
            exit(1)

    return curr_tensor, scalevec


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
