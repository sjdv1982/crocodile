import numpy as np

def euler2rotmat(phi, ssi, rot):
    phi, ssi, rot = np.array(phi), np.array(ssi), np.array(rot)
    assert phi.shape == ssi.shape == rot.shape
    cs = np.cos(ssi)
    cp = np.cos(phi)
    ss = np.sin(ssi)
    sp = np.sin(phi)
    cscp = cs*cp
    cssp = cs*sp
    sscp = ss*cp
    sssp = ss*sp
    crot = np.cos(rot)
    srot = np.sin(rot)

    r1 = crot * cscp + srot * sp
    r2 = srot * cscp - crot * sp
    r3 = sscp

    r4 = crot * cssp - srot * cp
    r5 = srot * cssp + crot * cp
    r6 = sssp

    r7 = -crot * ss
    r8 = -srot * ss
    r9 = cs
    result = np.zeros(phi.shape + (3,3))
    result_view = result.reshape(-1, 3, 3)
    result_view[:, 0, 0] = r1
    result_view[:, 0, 1] = r2
    result_view[:, 0, 2] = r3
    result_view[:, 1, 0] = r4
    result_view[:, 1, 1] = r5
    result_view[:, 1, 2] = r6
    result_view[:, 2, 0] = r7
    result_view[:, 2, 1] = r8
    result_view[:, 2, 2] = r9
    return result
