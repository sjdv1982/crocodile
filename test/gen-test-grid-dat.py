print("""#pivot auto
#centered receptor: false
#centered ligands: false""")
rotation = "0.12 0.21 0.08"
i = 0

voxelsize = 0.5
nvox=64
base = -0.5 * voxelsize*nvox
for px in range(nvox):
    x = base + (px + 0.5) * voxelsize
    for py in range(nvox):
        y = base + (py + 0.5) * voxelsize
        z = base - 0.5 * voxelsize
        for pz in range(nvox):
            z += voxelsize
            i += 1
            print("#{}".format(i))
            print("0 0 0 0 0 0")
            print("{} {} {} {}".format(rotation, x, y, z))