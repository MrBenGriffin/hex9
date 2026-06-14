import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


if __name__ == '__main__':
    data = np.load("phi_fit_l5_m0_vtw010820_n16.npz", allow_pickle=True)
    uv_cent = data["uv_cent"]
    row_w   = data["row_weight"]
    tri = mtri.Triangulation(uv_cent[:, 0], uv_cent[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    tpc = ax.tripcolor(tri, row_w, shading="flat")
    fig.colorbar(tpc, ax=ax, label="row weight")
    ax.set_aspect("equal", "box")
    ax.set_title("vertex taper weights (L2)")
    plt.tight_layout()
    plt.show()




