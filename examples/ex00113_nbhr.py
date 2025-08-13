"""
Part of the H9 project - Visualisation of neighbouring Half-hexagons.
Last Tested 05 August 2025 √
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, H9Engine, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util, Display
from matplotlib.patches import Polygon

if __name__ == '__main__':
    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})
    u = Util()
    reg = Registrar()  # Manage Domains & Projections
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    h9 = H9Engine()
    eg = EllipsoidGCD(reg)           # [g_gen, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]
    acc = ak.set_accuracy(0.1)
    locs = u.json_load('../assets/locations.json')
    for name in locs:
        print(name)
        region = locs[name]
        spots = np.array(list(region.values()))
        names = np.array(list(region.keys()))
        ll0 = g_gen.adopt(spots)
        sp0 = reg.project(ll0, [g_gen, c_ell])  # spherical cart
        oc0 = reg.project(sp0, [c_ell, c_oct])
        bc0 = reg.project(oc0, [c_oct, b_oct])
        cmp = tuple(bc0.components[0])
        sdo = bc0.domain.components[cmp]
        co, mo = bc0.cm()
        uri = h9.ugc_regions(bc0.coords, mo, acc)
        nbr = h9.neighbours(uri)
        ub2 = h9.neighbours(nbr)
        xyb = h9.ugc_dec(uri)
        xyn = h9.ugc_dec(nbr)
        xyr = h9.ugc_dec(ub2)
        for n, a, b, c, p0, p1, p2 in zip(names, uri, ub2, nbr, xyb, xyn, xyr):
            print(f'{n:<30}: {np.abs(p2-p0)};\n'
                  f'Original :{p0}; {a}\n'
                  f'Roundtrip:{p2}; {b}\n'
                  f'Neighbour:{p1}; {c}\n')
        done = True
