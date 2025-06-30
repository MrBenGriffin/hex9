"""
Part of the H9 project
This loads up a set of addresses and generates their h9 formats.
Tested and validated against Geodetic Conversions.
Last Tested 21 June 2025 √
"""
import numpy as np
from hhg9 import Registrar, H9Engine
from hhg9.domains import GeneralGCD, OctahedralCartesian, OctahedralBarycentric, EllipsoidCartesian
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util
from geographiclib.geodesic import Geodesic


def distance(p1, p2):
    """Return difference in metres between two points"""
    return Geodesic.WGS84.Inverse(p1[0], p1[1], p2[0], p2[1])['s12']


if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9 formats.
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)           # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    h9 = OctahedralH9()            # formatter.
    h9e = H9Engine()
    g_gcd.register_format(DMS())
    g_gcd.register_format(DecimalDegrees())
    c_ell.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)             # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)   # c_sph <=> (c_oct <=> b_oct)

    # Support Classes
    u = Util()
    # layers = ak.set_accuracy(0.025)
    layers = ak.set_accuracy(0.000000001)
    # nanometre.
    locs = u.json_load('../assets/locations.json')
    print('Selection of famous points, projected forwards and backwards, showing deviation ∂ in nanometres.')
    for region, spots in locs.items():
        dom = b_oct.sides[region]
        print(f'\nOctant {region} – {dom.sign}')
        pos = g_gcd.adopt(np.array(list(spots.values())))
        for name, ll0 in zip(spots.keys(), pos):
            print(f'\n{name:<24}             {ll0:dms} (Reference Coordinates)')
            sp0 = reg.project(ll0, [g_gcd, c_ell])  # spherical cart
            ll1 = reg.project(sp0, [c_ell, g_gcd])  # sph rt.
            d1 = distance(ll0.coords, ll1.coords) * 1000000000.
            print(f'{name:<24} ∂{d1:.6f}nm {ll1:dms} (roundtrip via GCD<->Ellipsoid)')
            oc0 = reg.project(ll0, [g_gcd, c_ell, c_oct])  # octa.
            ll2 = reg.project(oc0, [c_oct, c_ell, g_gcd])  # sph rt..
            d2 = distance(ll0.coords, ll2.coords[0]) * 1000000000.
            print(f'{name:<24} ∂{d2:.6f}nm {ll2:dms} (roundtrip via GCD<->Octahedral)')
            bc0 = reg.project(ll0, [g_gcd, c_ell, c_oct, b_oct])  # octa.
            ll3 = reg.project(bc0, [b_oct, c_oct, c_ell, g_gcd])  # sph rt..
            d3 = distance(ll0.coords, ll3.coords[0]) * 1000000000.
            print(f'{name:<24} ∂{d3:.6f}nm {ll3:dms} (roundtrip via GCD<->Barycentric)')
            address = f'{bc0:h9.{layers}}'
            print(f'{name:<24} {address} (Grid Address)')
            h9_r = h9.revert(address)  # Convert from address.
            ll4 = reg.project(h9_r, [b_oct, c_oct, c_ell, g_gcd])  # sph rt..
            d4 = distance(ll0.coords, ll4.coords[0]) * 100
            print(f'{name:<24} ∂{d4:.6f}cm {ll3:dms} (roundtrip via Grid Address)')

