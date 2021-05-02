"""
Test of the effects of delta-iRMSD:
If you have a receptor interface of size A,
and a ligand interface of size B,
with initial iRMSD R.
what is the effect on the iRMSD of a displacement of the ligand with a vector V with size sV?

Limiting case 1: all atom deviation is concentrated in the ligand
Rlig = sqrt(R**2 * (A + B)/B) = R / sqrt(B/(A+B))
R = Rlig * sqrt(B/(A+B))
displacement of sV will displace the ligand with dlig = (1-B/(A+B)) * sV = A/(A+B) * sV
Rlig will change at most by dlig
=> R will change at most by dlig * sqrt(B/(A+B)) = A/(A+B) * sqrt(B/(A+B)) * sV

Limiting case 2: all atom deviation is concentrated in the receptor
Rrec = sqrt(R**2 * (A + B)/A) = R / sqrt(A/(A+B))
R = Rlig * sqrt(A/(A+B))
displacement of sV will displace the receptor with drec = B/(A+B) * sV
Rrec will change at most by drec
=> R will change at most by drec * sqrt(A/(A+B)) = B/(A+B) * sqrt(A/(A+B)) * sV

Limiting case 3: all atom deviation is equally distributed
displacement of sV will displace the receptor with drec = B/(A+B) * sV
Rrec will change at most by drec
displacement of sV will displace the receptor with dlig = A/(A+B) * sV
Rlig will change at most by dlig

APPARENTLY (TODO: more math):
- This limiting case 3 is always dominant
- The answer is the geometric mean of drec and dlig

"""

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

if_r0, if_l0 = get_coor(receptor[imasks[0]]), get_coor(ligand[imasks[1]])
refe_if_r0, refe_if_l0 = get_coor(refe_receptor[imasks[0]]), get_coor(refe_ligand[imasks[1]])
for recsize, ligsize in ((1000, 10), (1000, 50), (1000, 100), (1000, 1000), (len(if_l0), 1000), (100, 1000), (50, 1000), (10, 1000)):
    refe_if_r = refe_if_r0[:recsize]
    if_r = if_r0[:recsize]
    refe_if_l = refe_if_l0[:ligsize]
    if_l = if_l0[:ligsize]

    lenr, lenl = len(refe_if_r), len(refe_if_l)
    natoms = lenr + lenl    
    f = max( 
        lenr/natoms,
        lenl/natoms
    )
    f = sqrt(lenl/natoms * lenr/natoms) ###

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

    offset_vectors = 2 * np.random.random((10000,3)) - 1
    offset_vectors /= np.linalg.norm(offset_vectors,axis=1)[:, None]
    x=(offset_vectors*offset_vectors).sum(axis=1)

    
    for disp1, disp2 in ((0, 5), (5, 10), (0, 10), (5, 15), (15, 20), (10, 10.01)):
        delta_disp = abs(disp2 - disp1)
        all_irmsds = []
        for disp in disp1, disp2:
            irmsds = []
            for offset0 in offset_vectors:
                offset = offset0 * disp

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
                irmsds.append(irmsd)
            all_irmsds.append(irmsds)
        all_irmsds = np.array(all_irmsds)
        d = np.abs(all_irmsds[1] - all_irmsds[0])
        print(d.max(), delta_disp * f, lenr, lenl, disp1, disp2, f)
