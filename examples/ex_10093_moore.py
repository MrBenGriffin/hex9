# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.addressing import x_adr_curve, hex_str_encode, hex_curve
from hhg9.h9.uuid_address import h9_curve_decode, h9_curve_uuid, h9_dec
from uuid import UUID
from hhg9.algorithms.distance import wgs84

# from hhg9.h9.grid import HexMesh

if __name__ == '__main__':
    # LAYER = 4
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    stonehenge = np.array([[51.1787980000210725, -1.8261898473293335]])
    g_pt = Points(stonehenge, 'g_gcd')
    b_pt = reg.project(g_pt, ['g_gcd', 'b_oct'])
    cpt = hex_curve(b_pt)
    cu = h9_curve_uuid(cpt)  # -> [UUID('c1157078-3641-03ff-...')]
    hd = h9_curve_decode(cu, reg)
    du = h9_dec(hd, b_oct)
    rt_pt = reg.project(du, ['b_oct', 'g_gcd'])
    print(rt_pt.coords[0])
    b_deltas = wgs84(g_pt.coords, rt_pt.coords) * 1e+9
    print(b_deltas)

    hxd = hex_str_encode(b_pt)
    cxd = x_adr_curve(hxd)
    clipped = cxd[:, :31]
    hod = h9_curve_decode(clipped, reg)
    u_b = h9_dec(hd, b_oct)
    rt_pt2 = reg.project(u_b, ['b_oct', 'g_gcd'])
    b_deltas = wgs84(g_pt.coords, rt_pt2.coords) * 1e+9
    print(b_deltas)

