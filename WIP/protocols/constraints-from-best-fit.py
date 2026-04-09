import json
import numpy as np
from crocodile.nuc.best_fit import best_fit
from crocodile.nuc.reference import Reference
from nefertiti.functions.superimpose import superimpose
from nefertiti.functions.write_pdb import write_pdb
from library_config import (
    mononucleotide_templates,
    dinucleotide_templates,
    dinucleotide_libraries,
)


def fl(v):
    return float(np.round(v + 0.0005, 3))


refe_ppdb = np.load("1b7f-rna-aa.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

anchors = [
    refe._resids.index(int(l.split()[1])) + 1 for l in open("anchors.txt").readlines()
]
anchors = {anchor: refe.get_coordinates(anchor, 1) for anchor in anchors}

fit = best_fit(
    2,
    [refe],
    ["1b7f"],
    fragment_libraries=dinucleotide_libraries,
    discrete_representation=True,
    keep_best_conformer=True,
    rotamer_precision=0.5,
    grid_spacing=np.sqrt(3) / 3,
)[0]
np.save("best-fit.npy", fit)

for fragpos, fragfit in zip(refe.get_fragment_positions(2), fit):
    seq = refe.get_sequence(fragpos, 2)
    libf = dinucleotide_libraries[seq]
    lib = libf.create(pdb_code="1b7f")
    fragfit_coors = lib.coordinates[fragfit["conformer"]]
    fitted_coors = fragfit_coors.dot(fragfit["rotation"]) + fragfit["offset"]

    tmpl = dinucleotide_templates[seq].copy()
    tmpl["x"] = fitted_coors[:, 0]
    tmpl["y"] = fitted_coors[:, 1]
    tmpl["z"] = fitted_coors[:, 2]
    pdbtxt = write_pdb(tmpl)
    with open(f"fragments/frag-{fragpos}-bestfit.pdb", "w") as f:
        f.write(pdbtxt)

constraints = {"pdb_code": "1b7f"}

for fragpos, fragfit in zip(refe.get_fragment_positions(2), fit):
    refe_coor = refe.get_coordinates(fragpos, 2)

    seq = refe.get_sequence(fragpos, 2)
    libf = dinucleotide_libraries[seq]
    curr = {
        "sequence": seq,
        "fit": fl(fragfit["rmsd"]),
        "primary_conformer": bool(fragfit["conformer"] < libf.nprimary),
    }
    constraints[f"frag{fragpos:d}"] = curr

    if fragpos in anchors:
        anchor = anchors[fragpos]
        lib = libf.create(pdb_code="1b7f", nucleotide_mask=[True, False])
        fragfit_coors = lib.coordinates[fragfit["conformer"]]
        fitted_coors = fragfit_coors.dot(fragfit["rotation"]) + fragfit["offset"]
        dif = fitted_coors - anchor
        rmsd = np.sqrt((dif * dif).sum() / len(fitted_coors))
        curr_anchor_full = {}
        curr_anchor = {"full": curr_anchor_full}
        curr_anchor_full["ovRMSD"] = fl(rmsd)
        _, crmsd = superimpose(fragfit_coors, anchor)
        curr_anchor_full["cRMSD"] = fl(crmsd)

        lib = libf.create(
            pdb_code="1b7f", nucleotide_mask=[True, False], only_base=True
        )
        base_anchor = refe.get_coordinates(fragpos, 2)[lib.atom_mask]

        fragfit_coors = lib.coordinates[fragfit["conformer"]]
        fitted_coors = fragfit_coors.dot(fragfit["rotation"]) + fragfit["offset"]
        dif = fitted_coors - base_anchor
        rmsd = np.sqrt((dif * dif).sum() / len(fitted_coors))
        curr_anchor["base"] = fl(rmsd)

        curr["anchor_up"] = curr_anchor

    if fragpos + 1 in anchors:
        anchor = anchors[fragpos + 1]
        lib = libf.create(pdb_code="1b7f", nucleotide_mask=[False, True])
        fragfit_coors = lib.coordinates[fragfit["conformer"]]
        fitted_coors = fragfit_coors.dot(fragfit["rotation"]) + fragfit["offset"]
        dif = fitted_coors - anchor
        rmsd = np.sqrt((dif * dif).sum() / len(fitted_coors))
        curr_anchor_full = {}
        curr_anchor = {"full": curr_anchor_full}
        curr_anchor_full["ovRMSD"] = fl(rmsd)
        _, crmsd = superimpose(fragfit_coors, anchor)
        curr_anchor_full["cRMSD"] = fl(crmsd)

        lib = libf.create(
            pdb_code="1b7f", nucleotide_mask=[False, True], only_base=True
        )
        base_anchor = refe.get_coordinates(fragpos, 2)[lib.atom_mask]

        fragfit_coors = lib.coordinates[fragfit["conformer"]]
        fitted_coors = fragfit_coors.dot(fragfit["rotation"]) + fragfit["offset"]
        dif = fitted_coors - base_anchor
        rmsd = np.sqrt((dif * dif).sum() / len(fitted_coors))
        curr_anchor["base"] = fl(rmsd)

        curr["anchor_down"] = curr_anchor

pairs = []
constraints["pairs"] = pairs
fragment_positions = refe.get_fragment_positions(2)
last_triseq = None
last_pairlist = None
for frag_ind, (fragpos, fragfit1) in enumerate(zip(fragment_positions, fit)):
    if frag_ind == len(fragment_positions) - 1:
        continue
    if fragment_positions[frag_ind + 1] != fragpos + 1:
        continue
    fragfit2 = fit[frag_ind + 1]

    seq1 = refe.get_sequence(fragpos, 2)
    libf1 = dinucleotide_libraries[seq1]
    seq2 = refe.get_sequence(fragpos + 1, 2)
    libf2 = dinucleotide_libraries[seq2]

    assert seq1[-1] == seq2[0]
    triseq = (seq1 + seq2[1:]).replace("G", "A").replace("U", "C")

    lib1 = libf1.create(pdb_code="1b7f", nucleotide_mask=[False, True])
    lib2 = libf2.create(pdb_code="1b7f", nucleotide_mask=[True, False])
    coor_conf1 = lib1.coordinates[fragfit1["conformer"]]
    coor_conf2 = lib2.coordinates[fragfit2["conformer"]]
    _, cRMSD = superimpose(coor_conf1, coor_conf2)
    if cRMSD < 0.1:
        cRMSD = 0.25
    coor1 = np.dot(coor_conf1, fragfit1["rotation"]) + fragfit1["offset"]
    coor2 = np.dot(coor_conf2, fragfit2["rotation"]) + fragfit2["offset"]
    dif = coor1 - coor2
    ovRMSD = np.sqrt((dif * dif).sum() / len(dif))
    pair = {
        "down": f"frag{fragpos:d}",
        "up": f"frag{fragpos+1:d}",
        "ovRMSD": fl(ovRMSD),
        "cRMSD": fl(cRMSD),
    }
    c1, c2 = constraints[pair["down"]], constraints[pair["up"]]
    if cRMSD < 0.5 and c1["primary_conformer"] and c2["primary_conformer"]:
        conf_down = fragfit2["conformer"]
        conf_up = fragfit1["conformer"]
        if triseq == last_triseq:
            pairlist = last_pairlist
        else:
            pairlist = {
                (v[0], v[1])
                for v in np.loadtxt(
                    open(f"dinuc-trinuc-pairs/{triseq}-pairlist.txt")
                ).astype(int)
            }
            last_pairlist = pairlist
            last_triseq = triseq
        if (conf_up + 1, conf_down + 1) in pairlist:
            pair["primary"] = True
    pairs.append(pair)

with open("constraints.json", "w") as f:
    json.dump(constraints, f, indent=2)
