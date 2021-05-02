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


def calc_irmsd(offset):
    vec1 = -offset * lenl/natoms
    vec2 = offset + vec1

    residual1 = residual_r + (2 * vec1 * r_sum).sum() + (lenr * vec1 * vec1).sum()
    residual2 = residual_l + (2 * vec2 * l_sum).sum() + (lenl * vec2 * vec2).sum()
    residual = residual1 + residual2    

    covar1 = covar_r + vec1[:, None] * refe_r_sum[None, :]
    covar2 = covar_l + vec2[:, None] * refe_l_sum[None, :]
    covar = covar1 + covar2

    v, s, wt = svd(covar)
    if det(v) * det(wt) < 0:
        s[-1] *= -1
        #v[:, -1] *= -1
    ss = (residual + refe_residual) - 2 * s.sum()
    if ss < 0:
        ss = 0
    irmsd = sqrt(ss / natoms)
    return irmsd

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

import itertools
offsets = []
for offset0 in itertools.product(range(10), repeat=3):
    offset = np.array(offset0) * 1.2
    offsets.append(offset)
offsets = np.array(offsets)
offsets = np.concatenate([offsets] * 500) # 500 000 calculations

import time
t = time.time()

irmsds = calc_irmsds(offsets)
print("Time elapsed: {:.3f} seconds".format(time.time() - t))
print(irmsds.shape)
for irmsd in irmsds[:10]:
    print(irmsd)