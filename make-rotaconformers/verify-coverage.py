
 # Verify that the rotamer clusters cover all of rotational space
# TODO: 
# - input a conformer index, 
# - load the input conformer library, get the conformer
# - load the result file JSON, get the checksum, load the rotamers

raise NotImplementedError # to adapt

# tensor should be identity for ProtNAff-generated libs...
tensor, scalevec = ctx.structure_tensor.value
verif = gen_rotations(100000)
m = ctx.modules.rotamers.module
rmsdref = np.sqrt(m.get_msd(verif, refe=None, scalevec=scalevec))

# paranoid check
struc = ctx.struc.value
coor = np.stack((struc["x"], struc["y"], struc["z"]),axis=1)
coor -= coor.mean(axis=0)
coor = np.dot(coor, tensor)
verif0 = verif[:10]
verif0_coors = np.einsum("ij,kjl->kil", coor, verif0) # broadcasted coor.dot(verif0[k])
dref0 = verif0_coors - coor
rmsdref0 = np.sqrt((dref0*dref0).sum(axis=2).mean(axis=1))
assert np.all(np.abs(rmsdref0 - rmsdref[:10]) < 0.1), (rmsdref0, rmsdref[:10])


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

# below should not be needed (tensor should be identity)
clusters_backrotated = np.einsum("ij,kjl->kil", tensor, clusters) #broadcasted tensor.dot(verif[k])
if outfile is not None:
    np.save(outfile, clusters_backrotated)
