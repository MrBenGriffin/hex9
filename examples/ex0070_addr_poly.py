"""
Part of the H9 project
HHex.tree binning by coordinate. (is the plan).
"""
import numpy as np
from hhg9 import Registrar, H9Engine
from hhg9.domains import SphericalGCD, SphericalCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import CartesianGCD, AKOctahedralSpherical
from hhg9.formats import OctahedralH9
from support import Util

if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9 formats.
    """
    reg = Registrar()                   # Manage Domains & Projections
    g_sph = SphericalGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_sph = SphericalCartesian(reg)     # Cartesian Spherical (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
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
            bc0 = reg.project(ll0, [g_sph, c_sph, c_oct, b_oct])  # octa.
            print(f'{name:<24}, {bc0[0]:h9.x}')

