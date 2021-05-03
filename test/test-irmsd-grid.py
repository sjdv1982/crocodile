import itertools
import numpy as np
from numpy import sqrt
from numpy.linalg import svd, det
from crocodile.parse_pdb import parse_pdb
receptor = parse_pdb(open("rotate-A.pdb").read())
ligand = parse_pdb(open("rotate-B.pdb").read())
refe_receptor = parse_pdb(open("1AVXA-bound-heavy.pdb").read())
refe_ligand = parse_pdb(open("1AVXB-bound-heavy.pdb").read())

from crocodile.imask import imask
imasks = imask([refe_receptor, refe_ligand], cutoff=10)

def get_coor(pdb):
    coor = np.zeros((len(pdb), 3))
    coor[:, 0] = pdb["x"]
    coor[:, 1] = pdb["y"]
    coor[:, 2] = pdb["z"]
    return coor

def calc_covar(atoms1, atoms2):
    assert atoms1.shape == atoms2.shape
    return atoms1.T.dot(atoms2)

if_r, if_l = get_coor(receptor[imasks[0]]), get_coor(ligand[imasks[1]])
refe_if_r, refe_if_l = get_coor(refe_receptor[imasks[0]]), get_coor(refe_ligand[imasks[1]])

###
#if_r, if_l = refe_if_r, refe_if_l # bound docking
###

lenr, lenl = len(refe_if_r), len(refe_if_l)
natoms = lenr + lenl    

refe_com = (refe_if_r.sum(axis=0) + refe_if_l.sum(axis=0)) / natoms
refe_if_r -= refe_com
refe_if_l -= refe_com
refe_residual = (refe_if_r*refe_if_r).sum() + (refe_if_l*refe_if_l).sum()

ini_com = (if_r.sum(axis=0) + if_l.sum(axis=0)) / natoms
if_r -= ini_com
if_l -= ini_com

r_sum = if_r.sum(axis=0)
l_sum = if_l.sum(axis=0)
refe_r_sum = refe_if_r.sum(axis=0)
refe_l_sum = refe_if_l.sum(axis=0)

covar_r = calc_covar(if_r, refe_if_r)
covar_l = calc_covar(if_l, refe_if_l)

residual_r = (if_r*if_r).sum()
residual_l = (if_l*if_l).sum()

f = sqrt(lenl/natoms * lenr/natoms)

def calc_irmsds(offsets):
    vec1 = -offsets * lenl/natoms
    vec2 = offsets + vec1

    residual1 = residual_r + (2 * vec1 * r_sum).sum(axis=1) + (lenr * vec1 * vec1).sum(axis=1)
    residual2 = residual_l + (2 * vec2 * l_sum).sum(axis=1) + (lenl * vec2 * vec2).sum(axis=1)
    residual = residual1 + residual2

    covar1 = covar_r + vec1[:, :, None] * refe_r_sum[None, None, :]
    covar2 = covar_l + vec2[:, :, None] * refe_l_sum[None, None, :]
    covar = covar1 + covar2

    v, s, wt = np.linalg.svd(covar)
    reflect = np.linalg.det(v) * np.linalg.det(wt)
    s[:,-1] *= reflect
    ss = np.maximum( (residual + refe_residual) - 2 * s.sum(axis=1), 0)
    irmsd = np.sqrt(ss / natoms)
    return irmsd

subd = list(itertools.product([-0.5, 0.5], repeat=3))
subd = np.array(subd)

lowest_voxelsize = 0.5

accept = []
accept_lowest = []

def irmsd_voxel(offset, gridlevel, threshold):
    voxelsize = lowest_voxelsize * 2**(gridlevel-1)
    corner = voxelsize * sqrt(3) 
    irmsd_margin = corner * f
    offsets = voxelsize * subd + offset
    irmsds = calc_irmsds(offsets)
    if gridlevel == 1:
        for n, irmsd, curr_offset in zip(range(8), irmsds, offsets):
            if irmsd < threshold:
                #print("ACCEPT LOWEST", offset, n)
                accept_lowest.append(curr_offset)
    else:
        for n, irmsd, curr_offset in zip(range(8), irmsds, offsets):
            if irmsd < threshold - irmsd_margin:
                #print("ACCEPT", gridlevel, offset, n)
                accept.append((gridlevel, curr_offset))
            elif irmsd > threshold + irmsd_margin:
                #print("REJECT", gridlevel, offset, n)
                pass # reject
            else:
                irmsd_voxel(curr_offset, gridlevel-1, threshold)

max_gridlevel = 10
threshold=4
print("Voxel size:", lowest_voxelsize, "Å")
print("Box size", lowest_voxelsize * 2**max_gridlevel, "Å")
print("iRMSD threshold", threshold, "Å")
irmsd_voxel(np.zeros(3), max_gridlevel, threshold)
"""
for a in accept:
    if a[0] > 2:
        print(a)
"""
print("Nodes:", len(accept), "leaves:", len(accept_lowest), "total:", sum([8**(a[0]-1) for a in accept])+len(accept_lowest))
print("Voxels: {:e}".format(8**max_gridlevel))