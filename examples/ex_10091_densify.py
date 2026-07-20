# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

import numpy as np

if __name__ == '__main__':
    from hhg9.h9.region import recover_stats_reset, recover_stats_report

    from hhg9 import Registrar
    from hhg9.h9.grid import HexMesh

    recover_stats_reset()

    reg = Registrar()

    # Build mesh with L0 through L3 all sharing the L3 vertex pool
    mesh = HexMesh.create(range(6), reg)
    print(repr(mesh))
    print(f'layers:{mesh.layers}')

    # densify L0 edges into L1 verts (delta=1, factor=3, 4 verts per edge)
    try:
        d0 = mesh.densify(0)
        print(f'densify(0) shape: {d0.shape}  expected ({12*3},{972 * 6 * 3})')
    except Exception as e:
        print(f'densify(0) FAILED: {e}')

        # densify L0 edges into L3 verts (delta=3, factor=27, 28 verts per edge)
    try:
        d0_3 = mesh.densify(0)  # with fine=3
        print(f'densify(0→3) shape: {d0_3.shape}  expected ({12}, {6 * 28})')
    except Exception as e:
        print(f'densify(0→3) FAILED: {e}')

        # densify L1 edges into L3 verts (delta=2, factor=9, 10 verts per edge)
    try:
        d1 = mesh.densify(1)
        print(f'densify(1→3) shape:{d1.shape}  expected ({12 * 9}, {6 * 10})')
    except Exception as e:
        print(f'densify(1→3) FAILED: {e}')

    n_v = len(mesh.pts.coords)
    for L in [0, 1]:
        try:
            d = mesh.densify(L)
            bad = (d < 0).any() or (d >= n_v).any()
            print(f'densify({L}) index range ok: {not bad} (max={d.max()}, n_verts={n_v})')
        except Exception as e:
            print(f'densify({L}) error:{e}')
    recover_stats_report()

