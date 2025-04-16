"""
This version of the workflow script runs the job submissions themselves inside a transformation
The result of these transformations (stage1 and stage2) can be easily verified, allowing
an already completed workflow to return its result instantly.
The main make-rotaconformers version may take more than half an hour to verify
  that the workflow has run to completion,

In addition, this version takes an output file argument where the result checksums will be written
(JSON format)
"""

import sys
import os
import numpy as np
import pathlib

import seamless
from seamless import transformer, Buffer

currdir = os.path.dirname(os.path.abspath(__file__))
CROCODILE_DIR = pathlib.Path(currdir).parent

MAX_ROTATIONS = 10000000

conformers_file = sys.argv[1]
result_file = sys.argv[2]
conformers = np.load(conformers_file)

# If the purpose of this script is to verify that the computation is done,
# then there is no need to contact an assistant.
###seamless.delegate(level=3)

# The script can also be used to do the entire calculation.
# In that case, comment the previous line and uncomment the next ones:

seamless.delegate()
seamless.config.unblock_local()

# (In addition, decrease NCONTEXTS in the code below )

# Note that local computation (level 3 delegation) works up to stage 1
#  ***but not for stage 2***
# All seamless tools to execute computations do a fingertip on the inputs.
#   These tools include: all assistants, seamless-run-transformation, seamless-fingertip
# However, for performance reasons, local execution does not do this,
#  which means that the scratch on the random rotations will lead to a cachemiss error.
# If you really need to do local computation/testing stage 2 of the workflow, do:
# seamless.config.fingertip_all = True

from seamless.workflow import Context, Module

############################################
# Preparation 1: generate random rotations
############################################


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


"""
# Execute in plain Python, and calculate checksum
random_rotations = gen_random_rotations(MAX_ROTATIONS)

print(random_rotations.shape, Buffer(random_rotations, "binary").get_checksum())
"""

# Execute in Seamless (either remotely via the assistant, or retrieve the result from the database)
gen_random_rotations = transformer(
    gen_random_rotations, scratch=True, return_transformation=True
)
tf = gen_random_rotations(MAX_ROTATIONS)
tf.compute()
random_rotations_checksum = tf.checksum
assert random_rotations_checksum
print("random rotations checksum:", random_rotations_checksum)

############################################
# Preparation 2
# Build Seamless modules for CrRoCODiLe functions we will need
############################################

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

############################################
# Preparation 3: convert conformers to structure tensors
############################################


@transformer(local=True)
def get_structure_tensors(conformers):
    result = []
    for coor in conformers:
        assert coor.ndim == 2 and coor.shape[-1] == 3
        tensor = build_rotamers.get_structure_tensor(coor)
        result.append(tensor)
    return result


get_structure_tensors.modules.build_rotamers = ctx.modules.build_rotamers
get_structure_tensors.modules.rotamers = ctx.modules.rotamers

print("Calculating structure tensors locally...")
tensors = get_structure_tensors(conformers)
print(len(tensors))

############################################
# Stage 1: pre-analysis to get the best clustering parameters
############################################


@transformer(in_process=True)
def stage1_pre_analysis(tensors):
    import seamless
    from tqdm import tqdm

    nconformers = len(tensors)

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

        hierarchy = get_best_clustering_hierarchy(
            est_size, rmsd_dist, rmsd, maxcostfrac
        )
        result["hierarchy"] = hierarchy

        return result

    do_pre_analysis.modules.build_rotamers = build_rotamers
    do_pre_analysis.modules.rotamers = rotamers

    print("START")
    with tqdm(total=nconformers, desc="Pre-analysis") as progress_bar:

        def pre_analyze(conformer):
            return do_pre_analysis(
                rsize=(1000, 5000, 5000000),
                rmsd=0.5,
                scalevec=tensors[conformer][1],
                maxcostfrac=0.3,
            )

        def callback(n, pre_analysis):
            progress_bar.update(1)
            if pre_analysis.checksum is None or pre_analysis.checksum.value is None:
                print(
                    f"""Failure for conformer {n}:
        status: {pre_analysis.status}
        exception: {pre_analysis.exception}
        logs: {pre_analysis.logs}"""
                )

        with seamless.multi.TransformationPool(1000) as pool:
            pre_analyses = pool.apply(pre_analyze, nconformers, callback=callback)

    print("Collect pre-analysis results...")
    ok = True
    pre_analysis_results = []
    for n, pre_analysis in enumerate(pre_analyses):
        cs = pre_analysis.checksum
        if cs is None or cs.value is None:
            print(
                f"Failed pre-analysis {n}, transformation checksum {pre_analysis.as_checksum()}"
            )
            ok = False
        else:
            v = pre_analysis.value
            if v is not None:
                pre_analysis_results.append(v)
            else:
                print(
                    f"Cannot get value for pre-analysis {n}, transformation checksum {pre_analysis.as_checksum()}, result checksum {pre_analysis.checksum}"
                )
                ok = False
    if not ok:
        exit(1)
    print("... done")
    return pre_analysis_results


stage1_pre_analysis.modules.build_rotamers = ctx.modules.build_rotamers
stage1_pre_analysis.modules.rotamers = ctx.modules.rotamers
pre_analyses = stage1_pre_analysis(tensors)

del ctx

############################################
# Stage 2: The actual clustering of the random rotations in C++
# Seamless C++ transformers must be built as a workflow graph
# using a Seamless Context.
#
# This stage must explicitly be done with delegation.
# The inputs (in particular, the random numbers),
# must be recomputed from their arguments.
# For security/performance reasons,
#  local execution does not do this automatically, but assistants do.
############################################


@transformer(in_process=True)
def stage2_main(tensors, pre_analyses, build_rotamers_code, random_rotations_checksum):
    from tqdm import tqdm
    import seamless
    import numpy as np

    from seamless.workflow import Context, Cell, Transformer

    MAX_ROTAMERS = 1e6
    NCONTEXTS = 50
    nconformers = len(tensors)

    print("Set up main context")
    ctx = Context()
    ctx.random_rotations = Cell("binary")
    ctx.scalevec = Cell("binary")
    ctx.hierarchy = Cell("binary")
    tf = ctx.build_rotamers = Transformer()
    tf.random_rotations = ctx.random_rotations
    tf.scalevec = ctx.scalevec
    tf.hierarchy = ctx.hierarchy
    tf.language = "cpp"

    # For some reason, this is necessary for some deployments but not all
    tf.link_options = ["-lm"]

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
    tf.code = build_rotamers_code
    tf.meta = {"ncores": 2, "memory": "3 GB"}
    ctx.random_rotations.scratch = True
    ctx.translate()
    ctx.random_rotations.set_checksum(random_rotations_checksum)
    ctx.compute(10)

    result_checksums = [None] * nconformers
    print("Setting up context pool")
    with seamless.multi.ContextPool(ctx, NCONTEXTS, show_progress=True) as pool:
        print("...done")
        with tqdm(total=nconformers, desc="Build rotamers") as progress_bar:

            def setup_func(ctx, conformer):
                ctx.scalevec = tensors[conformer][1]
                ctx.hierarchy = pre_analyses[conformer]["hierarchy"]

            def result_func(ctx, conformer):
                progress_bar.update(1)
                tf = ctx.build_rotamers
                result_checksum = tf.result.checksum
                if result_checksum is None or result_checksum.value is None:
                    print("No result", conformer, tf.status, tf.exception)
                    result_checksums[conformer] = None
                else:
                    result_checksums[conformer] = result_checksum.value

            pool.apply(setup_func, nconformers, result_func)
    if any([result_checksum is None for result_checksum in result_checksums]):
        raise RuntimeError

    return result_checksums


result_checksums = stage2_main(
    tensors,
    pre_analyses,
    open(os.path.join(currdir, "build-rotamers.cpp")).read(),
    random_rotations_checksum.value,
)

############################################
# Write result file
############################################

from seamless import Buffer

buf = Buffer(result_checksums, "plain")
buf.save(result_file)

print("Result file written")
