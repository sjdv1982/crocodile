import numpy as np

offset = np.linspace(-4, 4, 9).astype(int)
grid = np.stack(np.meshgrid(offset, offset, offset, indexing="ij"), axis=-1).reshape(
    -1, 3
)
ind_grid = 10000 * grid[:, 0] + 100 * grid[:, 1] + grid[:, 2]
grid = grid[np.argsort(ind_grid)]

np.random.seed(0)
scalevecs = np.random.random(size=(1000000, 3))
x, y, z = scalevecs.T
scalevecs = scalevecs[(x >= y) & (y >= z)]
scalevecs /= np.linalg.norm(scalevecs, axis=1)[:, None]

print("START")
for rmsd in np.linspace(0.2, 1.8, 17):
    print("RMSD", rmsd)
    rmsd2 = 2 * rmsd  # 0.5 grid units

    pts0 = grid[(grid**2).sum(axis=1) < rmsd2**2]
    delta = (0, 1)
    delta_grid = np.stack(
        np.meshgrid(delta, delta, delta, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    pts = (pts0[:, None] + delta_grid[None, :]).reshape(-1, 3)
    ind_pts = 10000 * pts[:, 0] + 100 * pts[:, 1] + pts[:, 2]
    pts = pts[np.unique(ind_pts, return_index=True)[1]]
    ind_pts = 10000 * pts[:, 0] + 100 * pts[:, 1] + pts[:, 2]
    pts = pts[np.argsort(ind_pts)]
    print(f"Reduced grid: {len(pts)}/{len(grid)}")
    mask1 = ((grid[None, :] - scalevecs[:, None]) ** 2).sum(axis=2) < rmsd2**2
    mask2 = ((pts[None, :] - scalevecs[:, None]) ** 2).sum(axis=2) < rmsd2**2
    print(
        "Same number of points selected:",
        np.all(mask1.sum(axis=1) == mask2.sum(axis=1)),
    )
    ok = 0
    for n in range(len(scalevecs)):
        sel1 = grid[mask1[n]]
        sel2 = pts[mask2[n]]
        if len(sel1) != len(sel2):
            continue
        if np.all(sel1 == sel2):
            ok += 1
    print(f"Same points selected: {ok}/{len(scalevecs)}")
    print()

"""
grid = grid[(grid * grid).sum(axis=1) < 4 * 4]

orders = set()

D = 0.01
for x in np.arange(2 * D, 0.5 + D, D):
    dxsq = (grid[:, 0] - x) ** 2
    for y in np.arange(D, 0.5 + D, D):
        if y >= x:
            break
        dxysq = dxsq + (grid[:, 1] - y) ** 2
        for z in np.arange(0, 0.5 + D, D):
            if z >= y:
                break
            dsq = dxysq + (grid[:, 2] - z) ** 2
            order = np.argsort(dsq)
            orders.add(tuple(order))
"""
