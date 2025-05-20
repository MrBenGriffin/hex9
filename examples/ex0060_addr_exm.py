"""
Part of the H9 project
This loads up a set of addresses and generates their h9 formats.
"""
import numpy as np
from hhg9 import Registrar
from hhg9.domains import SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import CartesianGCD, AKOctahedralSpherical
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util

if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9 formats.
    """
    reg = Registrar()  # Manage Domains & Projections
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    h9 = OctahedralH9()            # formatter.
    g_sph.register_format(DMS())
    g_sph.register_format(DecimalDegrees())
    c_sph.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    CartesianGCD(reg)           # g_sph <=> c_sph
    AKOctahedralSpherical(reg)  # c_sph <=> (c_oct <=> b_oct)

    # Support Classes
    u = Util()

    locs = u.json_load('../assets/locations.json')
    for region, spots in locs.items():
        pos = g_sph.adopt(np.array(list(spots.values())))
        for name, ll0 in zip(spots.keys(), pos):
            print(f'\n{name:<24} {ll0:dms}')
            sp0 = reg.project(ll0, [g_sph, c_sph])  # spherical cart
            ll1 = reg.project(sp0, [c_sph, g_sph])  # sph rt.
            print(f'{name:<24} ∂{np.abs(ll0.coords-ll1.coords)} (roundtrip via c_sph)')
            oc0 = reg.project(ll0, [g_sph, c_sph, c_oct])  # octa.
            ll2 = reg.project(oc0, [c_oct, c_sph, g_sph])  # sph rt..
            print(f'{name:<24} ∂{np.abs(ll0.coords-ll2.coords)} (roundtrip via c_oct)')
            bc0 = reg.project(ll0, [g_sph, c_sph, c_oct, b_oct])  # octa.
            ll3 = reg.project(bc0, [b_oct, c_oct, c_sph, g_sph])  # sph rt..
            print(f'{name:<24} ∂{np.abs(ll0.coords-ll3.coords)} (roundtrip via b_oct)')
            h9_a = f'{bc0:h9}'
            h9_r = h9.revert(h9_a)
            h9h = f'{bc0:h9.h20}'
            ll4 = reg.project(h9_r, [b_oct, c_oct, c_sph, g_sph])  # sph rt..
            print(f'{name:<24} {ll4:dms} via {h9_r:h9}')

