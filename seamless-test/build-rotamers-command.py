print("Run this inside a Seamless container with biopython installed")
from seamless.highlevel import Context, SubContext
import sys, os
import numpy as np

pdbfile = sys.argv[1]
pdbdata = open(pdbfile).read()
rmsd = float(sys.argv[2])
max_rotations = int(sys.argv[3])
outfile = sys.argv[4]

currdir = os.path.dirname(__file__)
graph = os.path.join(currdir, "graph/build-rotamer.seamless")
zipf = os.path.join(currdir, "graph/build-rotamer.zip")
ctx = Context()
ctx.add_zip(zipf)
ctx.set_graph(graph, mounts=False, shares=False)
ctx.translate()

def status(ctx):
    for childname in dir(ctx.children):
        if childname.startswith("_"):
            continue # bug
        try:
            child = getattr(ctx, childname)
        except AttributeError: #bug
            continue
        if isinstance(child, type):
            continue # bug
        elif isinstance(child, SubContext):
            status(child)
        elif child.status != "Status: OK":
            print(child, child.status, child.exception)

ctx.pdb = pdbdata
ctx.n_random_rotations = max_rotations
ctx.rmsd = rmsd
ctx.compute()
if ctx.build_rotamers.result is not None:
    arr = ctx.build_rotamers.result.value.unsilk
    np.save(outfile, arr)
else:
    status(ctx)

