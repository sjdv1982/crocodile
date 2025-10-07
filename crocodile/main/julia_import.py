import os

print("import julia...")
from juliacall import Main


currdir = os.path.dirname(os.path.realpath(__file__))
Main.include(os.path.join(currdir, "croco_candidates.jl"))
print("...done")
