import os
import argparse
import json
import sys
import numpy as np
from tqdm import tqdm
from crocodile.nuc.all_fit import (
    all_fit,
    conformer_mask_from_crmsd,
    conformer_mask_from_general_pairing,
    conformer_masks_from_specific_pairing,
)
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries

parser = argparse.ArgumentParser()
parser.add_argument("fragment", type=int)
parser.add_argument("target_directory")
parser.add_argument("origin_type", choices=["anchor", "fragment"])
parser.add_argument("origin_directory", nargs="?")
parser.add_argument("--up", action="store_true")
parser.add_argument("--down", action="store_true")

args = parser.parse_args()
assert not (args.up and args.down)


def err(*args):
    print(*args, file=sys.stderr)
    exit(1)


if not os.path.exists("constraints.json"):
    err("constraints.json must exist")

poses_file = os.path.join(args.target_directory, "poses.npy")
if os.path.exists(args.target_directory):
    if os.path.exists(poses_file):
        err(f"poses.npy in '{args.target_directory}' already exists")
else:
    os.makedirs(args.target_directory)
    os.link("score-data", os.path.join(args.target_directory, "score-data"))
    os.link("templates", os.path.join(args.target_directory, "templates"))

with open("constraints.json") as f:
    constraints = json.load(f)
pdb_code = constraints["pdb_code"]

fragment = args.fragment
fragmentstr = f"frag{fragment}"
frag_constr = constraints[fragmentstr]

anchor = None
anchor_updown = None
only_base = False
prev_frag = None
prev_fragstr = None

conformer_pair_masks = None
if args.origin_type == "fragment":
    if args.origin_directory is None:
        err("'fragment' requires an origin directory")
    fragfile = os.path.join(args.origin_directory, "FRAG")
    prev_frag = int(json.load(open(fragfile)))
    prev_fragstr = f"frag{prev_frag}"
    if prev_frag != fragment + 1 and prev_frag != fragment - 1:
        exit("Origin directory must be from an adjacent fragment")
    if prev_frag < fragment:
        frag0, frag1 = prev_fragstr, fragmentstr
    else:
        frag0, frag1 = fragmentstr, prev_fragstr
    for p in constraints["pairs"]:
        if p["down"] == frag0 and p["up"] == frag1:
            ovRMSD = p["ovRMSD"]
            cRMSD = p["cRMSD"]
            constraint_pair = p
            break
    else:
        exit("Cannot find constraint fragment pair")

else:
    if "anchor_up" in frag_constr and "anchor_down" in frag_constr:
        if not args.up and not args.down is None:
            err("Fragment has two anchors. Specify --up or --down")
        elif args.up:
            anchor = frag_constr["anchor_up"]
            anchor_updown = True
        else:
            anchor = frag_constr["anchor_down"]
            anchor_updown = False
    elif "anchor_up" in frag_constr:
        anchor = frag_constr["anchor_up"]
        anchor_updown = True
    elif "anchor_down" in frag_constr:
        anchor = frag_constr["anchor_down"]
        anchor_updown = False
    else:
        err("Fragment has no anchors specified")

    if "full" in anchor and "base" in anchor:
        err(
            "Anchor has both 'full' and 'base' specified. Remove one from the constraints file"
        )
    if "base" in anchor:
        only_base = True
        cRMSD = None
        ovRMSD = anchor["base"]
    else:
        only_base = False
        cRMSD = anchor["full"]["cRMSD"]
        ovRMSD = anchor["full"]["ovRMSD"]

refe_ppdb = np.load("reference.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

if anchor:
    if anchor_updown:
        nucleotide_mask = [True, False]
    else:
        nucleotide_mask = [False, True]
else:
    if prev_frag == fragment - 1:
        nucleotide_mask = [True, False]
        last_nucleotide_mask = [False, True]
    else:
        nucleotide_mask = [False, True]
        last_nucleotide_mask = [True, False]

seq = refe.get_sequence(fragment, fraglen=2)
assert seq == frag_constr["sequence"]
libf = dinucleotide_libraries[seq]
libf.load_rotaconformers()
print("START")
lib = libf.create(
    pdb_code=pdb_code,
    nucleotide_mask=nucleotide_mask,
    only_base=only_base,
    with_rotaconformers=True,
)
prop_indices = None
if anchor is not None:
    prev_poses = [None]
    prev_coors = refe.get_coordinates(fragment, fraglen=2)[lib.atom_mask]
else:
    index_file = os.path.join(args.origin_directory, "prop-indices.txt")
    if os.path.exists(index_file):
        prop_indices = set(np.loadtxt(index_file, ndmin=1).astype(np.uint).tolist())
    prev_poses = np.load(os.path.join(args.origin_directory, "poses-filtered.npy"))
    prev_coors = None
    prev_seq = refe.get_sequence(prev_frag, fraglen=2)
    prev_libf = dinucleotide_libraries[prev_seq]
    prev_lib = prev_libf.create(pdb_code="1b7f", nucleotide_mask=last_nucleotide_mask)
    if constraint_pair is not None and constraint_pair.get("primary"):
        print("Build conformer-specific pairing mask")
        otherseq = constraints[prev_fragstr]["sequence"]
        otherconf = len(
            dinucleotide_libraries[otherseq].create(pdb_code=pdb_code).coordinates
        )
        if prev_frag < fragment:
            trinuc_seq = otherseq[:-1] + seq
            trinuc_seq = trinuc_seq.replace("G", "A").replace("U", "C")
            trinuc_pairing = json.load(
                open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt")
            )
            conformer_pair_masks = conformer_masks_from_specific_pairing(
                trinuc_pairing,
                cRMSD,
                grow_up=True,
                nconformers=len(prev_lib.coordinates),
                mask_length=len(lib.coordinates),
            )
        else:
            trinuc_seq = seq + otherseq[1:]
            trinuc_seq = trinuc_seq.replace("G", "A").replace("U", "C")
            trinuc_pairing = json.load(
                open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt")
            )

            conformer_pair_masks = conformer_masks_from_specific_pairing(
                trinuc_pairing,
                cRMSD,
                grow_up=False,
                nconformers=len(prev_lib.coordinates),
                mask_length=len(lib.coordinates),
            )

conformer_mask = None

for p in constraints["pairs"]:
    if not p.get("primary"):
        continue
    primary_up = p["up"] == fragmentstr and p["down"] != prev_fragstr
    primary_down = p["down"] == fragmentstr and p["up"] != prev_fragstr

    if primary_up or primary_down:
        conformer_mask = np.ones(len(lib.coordinates), bool)

        def update_conformer_mask(trinuc_seq, mask_ind):
            trinuc_seq = trinuc_seq.replace("G", "A").replace("U", "C")
            trinuc_pairing = json.load(
                open(f"dinuc-trinuc-pairs/{trinuc_seq}-pairing.txt")
            )
            pair_mask = conformer_mask_from_general_pairing(
                trinuc_pairing, p["cRMSD"], min_mask_length=len(conformer_mask)
            )[mask_ind]
            conformer_mask[:] = conformer_mask & pair_mask

        if primary_down:
            trinuc_seq = seq + constraints[p["up"]]["sequence"][1:]
            update_conformer_mask(trinuc_seq, mask_ind=0)
        if primary_up:
            trinuc_seq = constraints[p["down"]]["sequence"][:-1] + seq
            update_conformer_mask(trinuc_seq, mask_ind=1)


new_poses = []
new_pose_origins = []
print(ovRMSD, cRMSD)
for prev_pose_nr, prev_pose in enumerate(tqdm(prev_poses)):  # type: ignore
    if prop_indices is not None:
        if prev_pose_nr + 1 not in prop_indices:
            continue

    curr_conformer_mask = None

    if prev_coors is not None:  # anchor
        prev_pose_coors = prev_coors
    else:
        prev_pose_conf_coors = prev_lib.coordinates[prev_pose["conformer"]]
        prev_pose_coors = (
            prev_pose_conf_coors.dot(prev_pose["rotation"]) + prev_pose["offset"]
        )
        if conformer_pair_masks is not None:
            conformer_pair_mask = conformer_pair_masks[prev_pose["conformer"]]
            if curr_conformer_mask is None:
                curr_conformer_mask = conformer_pair_mask.copy()
            else:
                curr_conformer_mask &= conformer_pair_mask

    if cRMSD is not None:
        if curr_conformer_mask is None:
            curr_conformer_mask = np.ones(len(lib.coordinates), bool)
        if conformer_mask is not None:
            curr_conformer_mask &= conformer_mask
        curr_conformer_mask &= conformer_mask_from_crmsd(
            reference=prev_pose_coors,
            fragment_library=lib,
            conformer_rmsd_threshold=cRMSD,
        )

    # if curr_conformer_mask is not None:
    #    print("mask", curr_conformer_mask.sum(), len(curr_conformer_mask))
    curr_new_poses = all_fit(
        prev_pose_coors,
        fragment_library=lib,
        rmsd_threshold=ovRMSD,
        conformer_mask=curr_conformer_mask,
        rotamer_precision=0.5,
        grid_spacing=np.sqrt(3) / 3,
        with_tqdm=(anchor is not None),
        return_rotamer_indices=True,
    )
    if len(curr_new_poses):
        print(prev_pose_nr + 1, len(curr_new_poses))
        new_poses.append(curr_new_poses)
        new_pose_origins.append(prev_pose_nr)

if anchor is None:
    with open(os.path.join(args.target_directory, "poses-origins.txt"), "w") as f:
        for ori, new_p in zip(new_pose_origins, new_poses):
            if not len(new_p):
                continue
            for n in range(len(new_p)):
                print(ori + 1, file=f)

poses = []
if len(new_poses):
    poses = np.concatenate(new_poses)

json.dump(fragment, open(os.path.join(args.target_directory, "FRAG"), "w"))
if prev_frag is not None:
    json.dump(prev_frag, open(os.path.join(args.target_directory, "ORIGIN"), "w"))
np.save(poses_file, poses)
print(f"{len(poses)} poses grown")
