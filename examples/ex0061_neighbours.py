"""
Part of the H9 project - uses H9 addresses and round-trips them.
This loads up a set of addresses and generates their h9 formats.
Tested and validated against Geodetic Conversions.
Last Tested 13 August 2025 √
"""
import numpy as np
from hhg9 import Registrar, H9Engine
from hhg9.domains import GeneralGCD, OctahedralCartesian, OctahedralBarycentric, EllipsoidCartesian
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
from hhg9.formats import DMS
from support import Util


if __name__ == '__main__':
    """
        Convert a set of locations to UGC lists and show their neighbours, with a roundtrip.
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    h9 = H9Engine()
    g_gcd.register_format(DMS())

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)             # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)   # c_sph <=> (c_oct <=> b_oct)

    # Support Classes
    u = Util()
    layers = ak.set_accuracy(0.0000001)
    locs = u.json_load('../assets/locations.json')
    np.set_printoptions(formatter={'int': lambda x: f'0x{x:02x},'})
    print('Selection of famous points, showing the neighbour & roundtrip.')
    for region, spots in locs.items():
        if region in b_oct.sides:
            dom = b_oct.sides[region]
            print(f'\nOctant {region} – {dom.sign}')
        else:
            print(f'\n{region}')
        pos = g_gcd.adopt(np.array(list(spots.values())))
        for name, ll0 in zip(spots.keys(), pos):
            print(f'\n{name:<24}             #{ll0:dms} (Reference Coordinates)')
            bc0 = reg.project(ll0, [g_gcd, c_ell, c_oct, b_oct])  # octa.
            do = [bc0.domain.components[tuple(c)].mo for c in bc0.components]
            # if region in b_oct.sides and b_oct.sides[region] != do[0]:
            #     print('octant error:', b_oct.sides[region], do)
            ugc = h9.ugc_regions(bc0.coords, do, 28)
            ngc = h9.neighbours(ugc.copy())
            rtp = h9.neighbours(ngc.copy())
            idx = np.argmax((ugc != ngc), axis=1)
            err = np.argmax((ugc != rtp), axis=1)
            print(f'{name:<24} ∂{err}; {idx}\n     Base:{ugc}\nRoundtrip:{rtp}\nNeighbour:{ngc}\n')

