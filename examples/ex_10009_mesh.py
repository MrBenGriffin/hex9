# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9K
from hhg9.h9.addressing import neighbours, H9_RA
from hhg9.h9.polygon import tri_mesh
from hhg9.h9.protocols import BaryLoc
from hhg9.h9.region import xy_regions, regions_xy
from hhg9.h9.classifier import location, in_scope, H9CL

if __name__ == '__main__':
    rg = Registrar()  # Manage Domains & Projections
    b_oct = rg.domain('b_oct')

    hex_layer = 3  # 0,...5 √
    oct_id = 0
    oct_cmp = H9O.oid_cmp[oct_id]
    oct_mode = H9O.oid_mo[oct_id]
    tri_verts, _, _ = tri_mesh(hex_layer, oct_mode)  # verts, edges, triangles.
    tvp = Points(tri_verts, b_oct, oct_cmp)
    t_oc, t_mo = tvp.cm()
    tx, ty = tvp.coords[:, 0], tvp.coords[:, 1]
    tẋ = tx * H9K.R3
    lc = location(tẋ, ty, t_mo, detailed=True)
    ext = np.flatnonzero(lc == BaryLoc.EXT)
    assert len(ext) == 0
    out_sc = np.flatnonzero(~in_scope(tẋ, ty, t_mo))
    lut = [
        "Undefined!", "External", "Internal",
        "APEX Vertex", "LEFT Vertex", "RGT Vertex",
        "bad_val", "bad_val",
        "C2=0 (flat edge)", "C2=1 (+ve edge)", "C2=2 (-ve edge)"
    ]
    for sc in out_sc:
        loca = lc[sc]
        l_str = lut[loca]
        l_mo = t_mo[sc]
        print(f'ẋ:{tẋ[sc]:.10f}, y:{tvp.coords[sc, 1]:.10f}; mode:{l_mo}; {l_str}')
    # assert len(in_sc) == 0, 'in_scope different from location'
    # # convert to classifier cell lists.
    # xyr = xy_regions(tri_verts, t_mo)
    # # dx = xyr[H9CL.in_scope]
    #
    # # round-trip back to coordinates.
    # rt_m = regions_xy(xyr)
    # rx, ry, rmo = rt_m[:, 0], rt_m[:, 1], rt_m[:, 2].astype(np.uint8)
    # rẋ = rx * H9K.R3
    # rt_lc = location(rẋ, ry, rmo)
    # rt_ext = np.flatnonzero(rt_lc == BaryLoc.EXT)
    # assert len(rt_ext) == 0
    # rt_sc = np.flatnonzero(~in_scope(rẋ, ry, rmo))
    # assert len(rt_sc) == 0
    # hex_regns = H9_RA.cell2rid[xyr]  # convert to full region list [0, 4, 4, etc]
    # bad_rx = np.flatnonzero(np.any(hex_regns == -1, axis=1))  # empty
    # assert len(bad_rx) == 0
    #
    #
    # print('all ok')
