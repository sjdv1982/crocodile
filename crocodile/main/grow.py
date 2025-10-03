import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation
'''
from crocodile.nuc.all_fit import (
    all_fit,
    conformer_mask_from_crmsd,
    conformer_mask_from_general_pairing,
    conformer_masks_from_specific_pairing,
)
'''
from crocodile.nuc.library import LibraryFactory
from crocodile.main.superimpose import superimpose_array
from crocodile.main import tensorlib

def _grow_from_anchor(command, constraints, state):
    from crocodile_library_config import dinucleotide_libraries

    origin = command["origin"]
    assert origin in ("anchor-up", "anchor-down")
    
    fragment = command["fragment"]
    key = "frag" + str(fragment)
    assert key in constraints

    if origin == "anchor-up":
        last_frag = fragment - 1
        nucleotide_mask = [True, False]
    else:
        last_frag = fragment + 1
        nucleotide_mask = [False, True]

    anchor_key = origin.replace("-", "_")
    assert anchor_key in constraints[key], constraints[key]
    anchor = constraints[key][anchor_key]
    if "base" in anchor:
        only_base = True
        cRMSD = None
        ovRMSD = anchor["base"]
    else:
        only_base = False
        cRMSD = anchor["full"]["cRMSD"]
        ovRMSD = anchor["full"]["ovRMSD"]

    refe = state["reference"]
    seq = refe.get_sequence(fragment, fraglen=2)

    pdb_code = constraints["pdb_code"]
    libf:LibraryFactory = dinucleotide_libraries[seq]
    libf.load_rotaconformers()
    print("START")

    lib = libf.create(
        pdb_code=pdb_code,
        nucleotide_mask=nucleotide_mask,
        only_base=only_base,
        with_rotaconformers=True,
    )

    anchor_coors = refe.get_coordinates(fragment, fraglen=2)[lib.atom_mask]
    anchor_offset = anchor_coors.mean(axis=0)
    anchor_coors = anchor_coors - anchor_offset

    lib_offset = lib.coordinates.mean(axis=1)
    lib_coors = lib.coordinates - lib_offset[:, None]

    lib_cmat, lib_crmsd = superimpose_array(lib_coors, anchor_coors)
    anchor_tensor, anchor_scalevec = tensorlib.get_structure_tensor(anchor_coors)
    lib_indices = np.arange(len(lib_coors)).astype(int)
    if not only_base:
        # 1. calculate merged scalevecs => lib_scalevec
        # 2. apply cRMSD filter to lib_cmat, lib_indices
        raise NotImplementedError
    else:
        lib_scalevec = np.repeat(anchor_scalevec, len(lib_coors))
    lib_cmat_t = lib_cmat.dot(anchor_tensor)
    
    for n in trange(len(lib_indices)):
        conf = lib_indices[n]
        scalevec = lib_scalevec[n]
        curr_lib_cmat_t = lib_cmat_t[n]
        rotamers0 = lib.get_rotamers(conf)
        rotamers = Rotation.from_rotvec(rotamers0)
        rotamers_t = rotamers.as_matrix().dot(lib_cmat_t[n])
        continue
        msd_conformational = lib_crmsd[n]**2
        msd_rotational = tensorlib.get_msd(rotamers_t, None, scalevec)
        msd_maxtrans = ovRMSD**2 - msd_conformational - msd_rotational
        #print(msd_maxtrans.shape, (msd_maxtrans>0).sum(), msd_maxtrans.max())
    

def grow(command, constraints, state):
    assert command["type"] == "grow", command
    print(command)
    print("GROW")
        
    if command["origin"] in ("anchor-up", "anchor-down"):
        return _grow_from_anchor(command, constraints, state)
    else:
        raise NotImplementedError