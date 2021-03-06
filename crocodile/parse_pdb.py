import warnings
import Bio.PDB
from Bio.PDB.StructureBuilder import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
from io import StringIO
import numpy as np

atomic_dtype = [
    ("model", 'uint16'),            
    ("hetero", "S1"),
    ("name", "S4"),
    ("altloc","S1"),
    ("resname", "S3"),            
    ("chain","S1"),
    ("index", 'uint32'),
    ("icode", "S1"), 
    ("resid", 'uint16'),            
    ("x", 'float32'),
    ("y", 'float32'),
    ("z", 'float32'),
    ("occupancy", 'float32'),
    ("bfactor", 'float32'),
    ("segid", "S4"),
    ("element", "S2")                  
]

atomic_dtype = np.dtype(atomic_dtype, align=True)

def parse_pdb(pdbdata):
    
    pdb_obj = StringIO(pdbdata)
    
    p = Bio.PDB.PDBParser()
    struc = p.get_structure("PDB", pdb_obj)
    natoms = len(list(struc.get_atoms()))        
    atomstate = np.zeros(natoms,dtype=atomic_dtype)
    
    a = atomstate
    count = 0
    for modelnr, model in enumerate(struc.get_models()):
        atomlist = list(model.get_atoms())
        atomlist.sort(key=lambda atom: atom.serial_number)
        for atom in atomlist:
            residue = atom.get_parent()
            hetero, resid, icode = residue.get_id()
            segid = residue.segid
            resname = residue.resname
            chainid = residue.get_parent().id
            aa = a[count]
            aa["model"] = modelnr + 1
            aa["hetero"] = hetero
            aa["name"] = atom.name
            aa["altloc"] = atom.altloc
            aa["resname"] = resname
            aa["chain"] = chainid
            aa["index"] = atom.serial_number
            aa["icode"] = icode
            aa["resid"] = resid
            aa["x"] = atom.coord[0]
            aa["y"] = atom.coord[1]
            aa["z"] = atom.coord[2]
            occ = atom.occupancy
            if occ is None or occ < 0:
                occ = 0
            aa["occupancy"] = occ
            aa["segid"] = segid
            aa["element"] = atom.element
            count += 1
    return atomstate

if __name__ == "__main__":
    import sys
    pdbfile = sys.argv[1]
    outfile = sys.argv[2]
    data = parse_pdb(open(pdbfile).read())
    np.save(outfile, data, allow_pickle=False)