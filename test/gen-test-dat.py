print("""#pivot auto
#centered receptor: false
#centered ligands: false""")
rotation = "0.12 0.21 0.08"
i = 0
offset = 1.2
for x in range(10):
    for y in range(10):
        for z in range(10):
            i += 1
            print("#{}".format(i))
            print("0 0 0 0 0 0")
            print("{} {} {} {}".format(rotation, offset * x, offset * y, offset * z))