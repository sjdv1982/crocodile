import sys
import os
import numpy as np
import pathlib
from tqdm import tqdm

import seamless
from seamless import transformer
from seamless.highlevel import Context, Module, Transformer, Cell

currdir = os.path.dirname(os.path.abspath(__file__))
CROCODILE_DIR = pathlib.Path(currdir).parent

MAX_ROTATIONS = 10000000

conformers_file = sys.argv[1]
conformers = np.load(conformers_file)

seamless.delegate()


def gen_random_rotations(n):
    import numpy as np
    from scipy.spatial.transform import Rotation

    np.random.seed(0)
    result = Rotation.random(n).as_matrix()

    # This will make the code reproducible.
    # At double precision, differences in the CPU AVX instruction set
    #  will cause tiny rounding errors
    result = result.astype(np.float32).astype(float)

    return result


random_rotations = gen_random_rotations(MAX_ROTATIONS)


def numpy2checksum(arr):
    from seamless.core.protocol.serialize import serialize_sync as serialize
    from seamless import calculate_checksum

    return calculate_checksum(serialize(arr, "binary"), hex=True)


print(random_rotations.shape, numpy2checksum(random_rotations))

gen_random_rotations = transformer(
    gen_random_rotations, scratch=True, return_transformation=True
)
tf = gen_random_rotations(MAX_ROTATIONS)
tf.compute()
random_rotations_checksum = tf.checksum
assert random_rotations_checksum is not None
print("random rotations checksum:", random_rotations_checksum)

ctx = Context()

# TODO: build Module without context
ctx.modules = Context()
ctx.modules.build_rotamers = Module()
ctx.modules.build_rotamers.dependencies = ["rotamers"]
ctx.modules.build_rotamers.mount(
    str(CROCODILE_DIR.joinpath("util", "build-rotamers.py")), "r"
)
ctx.modules.rotamers = Module()
ctx.modules.rotamers.mount(str(CROCODILE_DIR.joinpath("util", "rotamers.py")), "r")
ctx.compute()


@transformer
def get_structure_tensors(conformers):
    result = []
    for coor in conformers:
        assert coor.ndim == 2 and coor.shape[-1] == 3
        tensor = build_rotamers.get_structure_tensor(coor)
        result.append(tensor)
    return result


get_structure_tensors.modules.build_rotamers = ctx.modules.build_rotamers
get_structure_tensors.modules.rotamers = ctx.modules.rotamers

tensors = get_structure_tensors(conformers)
# tensors = tensors[:10]
# conformers = conformers[:10]
print(len(tensors))


@transformer(return_transformation=True)
def do_pre_analysis(rsize, rmsd, scalevec, maxcostfrac):
    from .build_rotamers import (
        estimate_nclust_curve,
        estimate_rmsd_dist,
        get_best_clustering_hierarchy,
    )

    r1size, r2size, r3size = rsize

    result = {}
    est_size = estimate_nclust_curve(rmsd, scalevec, r1size, r2size)

    result["nclust_curve"] = est_size
    rmsd_dist = estimate_rmsd_dist(list(est_size.keys()), scalevec, r3size)
    result["rmsd_dist"] = rmsd_dist

    hierarchy = get_best_clustering_hierarchy(est_size, rmsd_dist, rmsd, maxcostfrac)
    result["hierarchy"] = hierarchy

    return result


do_pre_analysis.modules.build_rotamers = ctx.modules.build_rotamers
do_pre_analysis.modules.rotamers = ctx.modules.rotamers

with tqdm(total=len(conformers), desc="Pre-analysis") as progress_bar:

    def pre_analyze(conformer):
        return do_pre_analysis(
            rsize=(1000, 5000, 5000000),
            rmsd=0.5,
            scalevec=tensors[conformer][1],
            maxcostfrac=0.3,
        )

    def callback(n, pre_analysis):
        progress_bar.update(1)
        if pre_analysis.checksum is None:
            print(
                f"""Failure for conformer {n}:
    status: {pre_analysis.status}
    exception: {pre_analysis.exception}
    logs: {pre_analysis.logs}"""
            )

    with seamless.multi.TransformationPool(1000) as pool:
        pre_analyses = pool.apply(pre_analyze, len(conformers), callback=callback)

if any([pre_analysis.checksum is None for pre_analysis in pre_analyses]):
    exit(1)

print("Collect pre-analysis results...")
pre_analyses = [pre_analysis.value for pre_analysis in pre_analyses]
print("...done")

del ctx

print("Set up main context")
MAX_ROTAMERS = 1e6
ctx = Context()
ctx.random_rotations = Cell("binary")
ctx.scalevec = Cell("binary")
ctx.hierarchy = Cell("binary")
tf = ctx.build_rotamers = Transformer()
tf.random_rotations = ctx.random_rotations
tf.scalevec = ctx.scalevec
tf.hierarchy = ctx.hierarchy
tf.language = "cpp"
ctx.translate()
tf.inp.example.random_rotations = np.zeros((5, 3, 3))
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
tf.result.example.set(np.zeros((5, 3, 3)))
form = tf.result.schema["form"]
form.shape = (0, int(MAX_ROTAMERS)), 3, 3
form.contiguous = True
# tf.code.mount("build-rotamers.cpp")
tf.code = open(os.path.join(currdir, "build-rotamers.cpp")).read()
tf.meta = {"ncores": 2, "memory": "3 GB"}
ctx.random_rotations.scratch = True
ctx.translate()
ctx.random_rotations.set_checksum(random_rotations_checksum)
ctx.compute(10)

contexts = []

NCONTEXTS = 100
print("Setting up context pool")
with seamless.multi.ContextPool(ctx, NCONTEXTS) as pool:
    print("...done")
    with tqdm(total=len(conformers), desc="Build rotamers") as progress_bar:

        def setup_func(ctx, conformer):
            ctx.scalevec = tensors[conformer][1]
            ctx.hierarchy = pre_analyses[conformer]["hierarchy"]

        def result_func(ctx, conformer):
            progress_bar.update(1)
            tf = ctx.build_rotamers
            print("Result", conformer, tf.result.checksum, tf.status, tf.exception)

        pool.apply(setup_func, len(conformers), result_func)
