import sys, os
print("Run this inside a Seamless container with biopython installed", file=sys.stderr)
import numpy as np

from seamless.highlevel import Context, SubContext
pdbfile = sys.argv[1]
pdbdata = open(pdbfile).read()
rmsd = float(sys.argv[2])
max_rotations = int(sys.argv[3])
outfile = None
if len(sys.argv) > 4:
    outfile = sys.argv[4]
else:
    print("No output file", file=sys.stderr)

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
            print(child, child.status, child.exception, file=sys.stderr)

ctx.pdb = pdbdata
ctx.n_random_rotations = max_rotations
ctx.rmsd = rmsd
ctx.compute()
if ctx.build_rotamers.result.checksum is None:
    status(ctx)
    sys.exit(1)

clusters = ctx.build_rotamers.result.value.unsilk
print(f"OUTPUT {max_rotations} random rotations => {len(clusters)} clusters")

# check
print("Checking output", file=sys.stderr)
def gen_rotations(n):
    import numpy as np
    from scipy.spatial.transform import Rotation
    np.random.seed(777)
    return Rotation.random(n).as_matrix()

tensor, scalevec = ctx.structure_tensor.value
verif = gen_rotations(100000)
m = ctx.modules.rotamers.module
rmsdref = np.sqrt(m.get_msd(verif, refe=None, scalevec=scalevec))

# paranoid check
struc = ctx.struc.value
coor = np.stack((struc["x"], struc["y"], struc["z"]),axis=1)
coor -= coor.mean(axis=0)
coorx = coor.dot(tensor)
verif0 = verif[:10]
verif0_coors = np.einsum("ij,kjl->kil", coorx, verif0) # broadcasted coorx.dot(verif0[k])
dref0 = verif0_coors - coorx
rmsdref0 = np.sqrt((dref0*dref0).sum(axis=2).mean(axis=1))
try:
    assert np.all(np.abs(rmsdref0 - rmsdref[:10]) < 0.05), (rmsdref0, rmsdref[:10])
except AssertionError:
    struc = ctx.struc.value
    coor = np.stack((struc["x"], struc["y"], struc["z"]),axis=1)
    coor -= coor.mean(axis=0)
    coorx = coor.dot(tensor.T)
    verif0 = verif[:10]
    verif0_coors = np.einsum("ij,kjl->kil", coorx, verif0) # broadcasted coorx.dot(verif0[k])
    dref0 = verif0_coors - coorx
    rmsdref0 = np.sqrt((dref0*dref0).sum(axis=2).mean(axis=1))
    assert np.all(np.abs(rmsdref0 - rmsdref[:10]) < 0.05), (rmsdref0, rmsdref[:10])

verifx = np.einsum("ij,kjl->kil", tensor, verif) #broadcasted tensor.dot(verif[k])
verif0x = verifx[:10]
verif0_xcoors = np.einsum("ij,kjl->kil", coor, verif0x)
dref0 = verif0_xcoors - coor.dot(tensor)
rmsdref0 = np.sqrt((dref0*dref0).sum(axis=2).mean(axis=1))
try:
    assert np.all(np.abs(rmsdref0 - rmsdref[:10]) < 0.05), (rmsdref0, rmsdref[:10])
except AssertionError:
    verifx = np.einsum("ij,kjl->kil", tensor.T, verif) #broadcasted tensor.dot(verif[k])
    verif0x = verifx[:10]
    verif0_xcoors = np.einsum("ij,kjl->kil", coor, verif0x)
    dref0 = verif0_xcoors - coor.dot(tensor.T)
    rmsdref0 = np.sqrt((dref0*dref0).sum(axis=2).mean(axis=1))
    assert np.all(np.abs(rmsdref0 - rmsdref[:10]) < 0.05), (rmsdref0, rmsdref[:10])

#/paranoid check
#import sys; sys.exit()

not_in_cluster = 0
for vnr, v in enumerate(verif):
    if not (vnr % 1000):
        print("{}/{}, not in cluster: {}".format(vnr+1, len(verif), not_in_cluster), file=sys.stderr)
    vrmsd = np.sqrt(m.get_msd(clusters, refe=v, scalevec=scalevec).min())
    if vrmsd > rmsd:
        not_in_cluster += 1
print("OUTPUT Tested X random rotations, of which Y do not cluster within the {} threshold".format(rmsd))
print(f"OUTPUT Y / X: {not_in_cluster} / {len(verif)}")

clusters_backrotated = np.einsum("ij,kjl->kil", tensor, clusters) #broadcasted tensor.dot(verif[k])
if outfile is not None:
    np.save(outfile, clusters_backrotated)
