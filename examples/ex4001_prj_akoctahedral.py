# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Proof of concepts taken from AKOctahedral, demonstrating jacobian

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed - with rewrite)
"""

import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.algorithms.pickers import gcd_rnd


if __name__ == '__main__':
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    c_oct = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    s_oct = reg.domain('s_oct')
    ake = reg.projection('oct_ell')

    london = np.array([[51.50744520, -0.1278120321]])  # London Latitude/Longitude
    ldn = g_gcd.adopt(london)
    t_oct = reg.project(ldn, [g_gcd, c_ell, c_oct])
    r_ldn = reg.project(t_oct, [c_oct, c_ell, g_gcd])
    delta = wgs84(ldn.coords[0], r_ldn.coords[0]) * 1e+9
    print("1nm Accuracy: Ellipsoid residual in nanometres:", delta)  # Should be small

    ake.set_accuracy(1000.0)
    t_oct = reg.project(ldn, [g_gcd, c_ell, c_oct])
    r_ldn = reg.project(t_oct, [c_oct, c_ell, g_gcd])
    delta = wgs84(ldn.coords[0], r_ldn.coords[0]) * 0.001
    print("1km Accuracy: Ellipsoid residual in kilometres:", delta)  # Should be small

    # ake.set_accuracy(10000.0)
    ake.set_accuracy(0.000000000001)
    wgs = gcd_rnd(25000)
    w_pts = Points(wgs, domain=g_gcd)
    ocs = reg.project(w_pts, [g_gcd, c_ell, c_oct])

    # dv = bk.coords - ox.coords
    # dd = np.linalg.norm(dv, axis=1)
    px = ocs.coords
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(px[:, 0], px[:, 1], px[:, 2], marker='.', s=5.0)
    ax.set_aspect('equal', adjustable='box')
    f_name = f'output/ex4001_octagon.png'
    fig.savefig(f_name, dpi=100)
    plt.close(fig)
    print(f'fig saved at f_name')

    def err_stats(j_a, j_b):
        diff = np.abs(j_a - j_b)
        rel = diff / np.maximum(1e-9, np.abs(j_b))
        q = lambda x, p: np.percentile(x, p)
        return dict(
            abs_min=diff.min(),
            abs_max=diff.max(),
            abs_mean=diff.mean(),
            abs_p95=q(diff, 95),
            abs_med=q(diff, 50),
            rel_max=rel.max(),
            rel_p95=q(rel, 95),
            rel_med=q(rel, 50),
        )

    eps = 1e-7
    # avoid non-differentiable edges (any coord near 0 on the octahedron)
    mask = (np.abs(ocs.coords) > 1e-6).all(axis=1)
    u_in = ocs.coords[mask]
    j_analytic = ake.jacobian(u_in)
    j_numeric = np.stack([
        (ake.forward(Points(u_in + eps * np.eye(3)[i], ake.c_oct)).coords -
         ake.forward(Points(u_in - eps * np.eye(3)[i], ake.c_oct)).coords) / (2 * eps)
        for i in range(3)], axis=-1)

    s2 = err_stats(j_analytic, j_numeric)
    print("Jacobian: err_stats ", s2)
    # abs_err = np.max(np.abs(j_analytic - j_numeric))
    # rel_err = np.max(np.abs((j_analytic - j_numeric) / (np.maximum(1e-9, np.abs(j_numeric)))))
    # print(f"Jacobian check: max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}")
    #
    bak = reg.project(ocs, [c_oct, c_ell])
    px = bak.coords
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(px[:, 0], px[:, 1], px[:, 2], marker='.', s=5.0)
    ax.set_aspect('equal', adjustable='box')
    f_name = f'output/ex4001_ell.png'
    fig.savefig(f_name, dpi=100)
    plt.close(fig)
    print(f'fig saved at f_name')
