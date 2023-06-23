import numpy as np

from seamless.highlevel import Context, Cell, Transformer, Module
ctx = Context()

ctx.modules = Context()
ctx.modules.build_rotamers = Module()
ctx.modules.build_rotamers.dependencies = ["rotamers"]
ctx.modules.build_rotamers.mount("util/build-rotamers.py", "r")
ctx.modules.rotamers = Module()
ctx.modules.rotamers.mount("util/rotamers.py", "r")
ctx.modules.parse_pdb = Module()
ctx.modules.parse_pdb.mount("crocodile/parse_pdb.py", "r")

ctx.rmsd = Cell("float").set(4)

def gen_rotations(n):
    import numpy as np
    from scipy.spatial.transform import Rotation
    np.random.seed(0)
    return Rotation.random(n).as_matrix()
ctx.gen_rotations = gen_rotations
ctx.translate()
ctx.gen_rotations.inp.example = {}
ctx.gen_rotations.inp.example.n = 0
ctx.gen_rotations.inp.schema.properties.n.minimum = 0
ctx.translate()
ctx.n_random_rotations = Cell("int").set(2)  #.set(300000)
ctx.gen_rotations.n = ctx.n_random_rotations
ctx.gen_rotations.result.example.set(np.zeros((5,3,3)))
ctx.gen_rotations.result.schema["form"].shape = -1, 3, 3

ctx.random_rotations = ctx.gen_rotations
ctx.translate()

ctx.pdb = Cell("text")
def parse_pdb(pdb):
    return parse_pdb_module.parse_pdb(pdb)

ctx.parse_pdb = parse_pdb
ctx.parse_pdb.pdb = ctx.pdb
ctx.parse_pdb.parse_pdb_module = ctx.modules.parse_pdb
ctx.struc = ctx.parse_pdb
ctx.struc.celltype = "binary"
ctx.translate()

###
ctx.pdb.set(open("1AVXB-unbound-bb.pdb").read())
###
ctx.compute()

def get_structure_tensor(struc):
    import numpy as np
    coor = np.zeros((len(struc), 3))
    coor[:, 0] = struc["x"]
    coor[:, 1] = struc["y"]
    coor[:, 2] = struc["z"]
    return build_rotamers.get_structure_tensor(coor)

ctx.get_structure_tensor = get_structure_tensor
ctx.get_structure_tensor.struc = ctx.struc
ctx.get_structure_tensor.build_rotamers = ctx.modules.build_rotamers
ctx.get_structure_tensor.rotamers = ctx.modules.rotamers
ctx.structure_tensor = ctx.get_structure_tensor

ctx.scalevec = Cell()
ctx.translate()
ctx.scalevec.example.set(np.empty(3))
ctx.scalevec.schema["form"].shape = (3,)
ctx.scalevec = ctx.structure_tensor[1]
ctx.compute()

def do_pre_analysis(rsize, rmsd, scalevec, maxcostfrac):
    import numpy as np
    from .build_rotamers import estimate_nclust_curve, estimate_rmsd_dist, get_best_clustering_hierarchy
    r1size, r2size, r3size = rsize
  
    result = {}
    est_size = estimate_nclust_curve(rmsd, scalevec, r1size, r2size)
  
    result["nclust_curve"] = est_size
    rmsd_dist = estimate_rmsd_dist(list(est_size.keys()), scalevec, r3size)
    result["rmsd_dist"] = rmsd_dist
  
    hierarchy = get_best_clustering_hierarchy(est_size, rmsd_dist, rmsd, maxcostfrac)
    result["hierarchy"] = hierarchy 

    return result

ctx.do_pre_analysis = do_pre_analysis
tf = ctx.do_pre_analysis
tf.rsize = 1000, 5000, 5000000
tf.rmsd = ctx.rmsd
tf.scalevec = ctx.scalevec
tf.maxcostfrac = 0.3#
tf.rotamers = ctx.modules.rotamers
tf.build_rotamers = ctx.modules.build_rotamers
ctx.pre_analysis = tf.result
ctx.translate()

ctx.hierarchy = Cell("binary")
ctx.hierarchy = ctx.pre_analysis.hierarchy

MAX_ROTAMERS = 1e8
tf = ctx.build_rotamers = Transformer()
tf.random_rotations = ctx.random_rotations
tf.scalevec = ctx.scalevec
tf.hierarchy = ctx.hierarchy
tf.language = "cpp"
ctx.compute()
tf.inp.example.random_rotations = np.zeros((5,3,3))
tf.inp.example.scalevec = np.zeros(3)
tf.inp.example.hierarchy = np.zeros(10)
tf.schema.required = ["random_rotations", "scalevec", "hierarchy"]
form = tf.schema.properties.random_rotations["form"]
form.shape = -1, 3, 3
form.contiguous = True
form = tf.schema.properties.scalevec["form"]
form.shape = (3,)
form.contiguous = True
form = tf.schema.properties.hierarchy["form"]
form.contiguous = True
tf.result.example.set(np.zeros((5,3,3)))
form = tf.result.schema["form"]
form.shape = (0, int(MAX_ROTAMERS)), 3, 3
form.contiguous = True
ctx.compute()

c = ctx.build_rotamers_code = Cell("code")
c.language = "cpp"
c.mount("build-rotamers.cpp")
tf.code = c
ctx.translate()
ctx.compute(0.5)
if c.value is None:
    header = tf.header.value
'''    
    if header is not None:
        code = header.replace("int transform", 'extern "C" int transform') + "\n"
        code += """
#include <cstdio>

extern "C" int transform(
    const HierarchyStruct* hierarchy, const RandomRotationsStruct* random_rotations, 
    const ScalevecStruct* scalevec, ResultStruct *result
) {
    for (int i = 0; i < 3; i++) {
        printf("%.3f ", scalevec->data[i]);
    }
    printf("\\n\\n");
    return 0;
}        
"""        
        c.set(code)
'''

ctx.save_graph("graph/build-rotamer.seamless")
ctx.save_zip("graph/build-rotamer.zip")
ctx.save_vault("vault")