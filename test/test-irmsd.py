import numpy as np
from numpy import sqrt
import itertools
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

gold_refe = np.concatenate([refe_if_r, refe_if_l])

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

offset = np.zeros(3)

for offset0 in itertools.product(range(10), repeat=3):
    offset[:] = offset0 
    offset *= 1.2

    """
    gold = np.concatenate([if_r, if_l + offset])
    gold -= gold.sum(axis=0)/len(gold)
    gold_covar = calc_covar(gold, gold_refe)
    print("Gold covar")
    print(gold_covar)
    gold_residual = (gold*gold).sum()
    print("Gold residual")
    print(gold_residual)
    """


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
    print(irmsd)

#print(covar1 + covar2)
# sumN((x + a) * y) = sumN(x*y) + sumN(a * y) = sumN(x*y) + a * sumN(y)
# sumN((x + a)**2) = sumN(x**2) + sumN(2xa) + sumN(a**2) =  sumN(x**2) + 2a * sumN(x) + N * a**2