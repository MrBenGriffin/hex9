# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Showing 2 layers of hexagons:  land-usage, hill-shading
Last Tested
16 Jun 2026 0.1.3a0 (passed) 1873.1s
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2026 0.1.0a4 (passed)
"""

import numpy as np
from hhg9 import Points, Registrar
from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
from hhg9.h9.addressing import hex_layer, TailStyle
from hhg9.h9.grid import poly_net_field


if __name__ == '__main__':
    # WGS84 ellipsoid surface area (m^2). Using helper keeps this aligned with the rest of the stack.
    earth = ellipsoid_area_wgs84()
    hex_0 = earth / 12
    l_hex = hex_0
    h_areas = np.zeros((64,), dtype=np.float64)
    for i in range(64):
        h_areas[i] = l_hex
        l_hex /= 9

    tests = {
        'Hyde Park': [
            (51.515948, -0.212225),
            (51.518371, -0.151910),
            (51.500830, -0.143740),
            (51.497451, -0.197695)
        ],
        'LAX, CA': [
            (-118.418990325877004, 33.9333814840111),
            (-118.419070754935007, 33.933926959066604),
            (-118.382766218171994, 33.9376406125145),
            (-118.382686017115006, 33.937095114458899),
        ],
        'Utqiagvik, AK':
        [
            [-156.738405694211991, 71.284672766744194],
            [-156.798810301001993, 71.284662977615298],
            [-156.798811581733986, 71.285072736580005],
            [-156.738405700945009, 71.285082525709299],
            [-156.738405694211991, 71.284672766744194]
         ],
    }

    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    n_oct = rg.domain('n_oct:butterfly:1000')

    for name, values in tests.items():
        coords = np.array(values, dtype=np.float64)[:, ::-1]
        gcd_pts = Points(coords, g_gcd)  # expects (lat,lon) in g_gcd
        gcd_area = wgs84_area(rg, gcd_pts, coords.shape[0])
        print(f'\n{name}: WGS84 area: {float(gcd_area[0])}m²')
        h9_pts = rg.project(gcd_pts, [g_gcd, b_oct])  #
        net_pts = rg.project(h9_pts, [b_oct, n_oct])
        for layer in range(9, 16):
            scn = poly_net_field(net_pts, layer)
            b_pts = rg.project(scn, [n_oct, b_oct])
            h_key = hex_layer(b_pts, layer=layer, tail_style=TailStyle.key)
            hex_k = np.unique(h_key, axis=0)
            hexes = hex_k.shape[0]
            one_hex = h_areas[layer]
            area = hexes * one_hex
            print(f'Layer {layer} {one_hex}m² / hex; {hexes} estimated area {area}m²')
