import numpy as np

dinuc_dtype = np.dtype(
    [
        ("first_resid", np.uint16),
        ("sequence", "S2"),
        ("conformer", np.uint16),
        ("matrix", np.float64, (4, 4)),
        ("rmsd", np.float32),
    ],
    align=True,
)


dinuc_roco_dtype = np.dtype(
    [
        ("first_resid", np.uint16),
        ("sequence", "S2"),
        ("conformer", np.uint16),
        ("rotation_matrix", np.float64, (3, 3)),
        ("offset", np.float64, 3),
        ("rmsd", np.float32),
    ],
    align=True,
)

from .from_ppdb import from_ppdb, ppdb2nucseq
