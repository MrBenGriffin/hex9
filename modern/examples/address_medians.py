import numpy as np
from modern.ak_projection import AKProjection
from modern.octahedron import Octahedron
from modern.util import Util
from modern.octahedron_h9 import H9Octahedron

# 2025-05-12 This currently fails - possibly an issue in side identification?

# Using information derived from the Octahedral Projection, that represent 3 points all of which
# are slightly inside each Octahedral Side
# Generate the addresses of each of these points and print out their values,
# to ensure that the addresses match the half-hexagons of each side correctly.
# For example, the results for NWP will be
# [  79.87691379 -135.        ]: NPV....
# [  7.13938016 -97.19546005]:   NWV...
# [   7.13938016 -172.80453995]: WPV...


def addresses():
    return {
        'SEA': np.array([[-79.87691379, 45.], [-7.13938016, 82.80453995], [-7.13938016, 7.19546005]]),
        'SWA': np.array([[-79.87691379, -45.], [-7.13938016, -82.80453995], [-7.13938016, -7.19546005]]),
        'NWP': np.array([[79.87691379, -135.], [7.13938016, -97.19546005], [7.13938016, -172.80453995]]),
        'SEP': np.array([[-79.87691379, 135.], [-7.13938016, 97.19546005], [-7.13938016, 172.80453995]]),
        'SWP': np.array([[-79.87691379, -135.], [-7.13938016, -97.19546005], [-7.13938016, -172.80453995]]),
        'NEA': np.array([[79.87691379, 45.], [7.13938016, 82.80453995], [7.13938016, 7.19546005]]),
        'NEP': np.array([[79.87691379, 135.], [7.13938016, 97.19546005], [7.13938016, 172.80453995]]),
        'NWA': np.array([[79.87691379, -45.], [7.13938016, -82.80453995], [7.13938016, -7.19546005]]),
    }


if __name__ == '__main__':
    c = Octahedron()
    ak = AKProjection()  # An implementation of octagon/sphere projection.
    u = Util()
    o = H9Octahedron(c)
    ex = addresses()
    for side, values in ex.items():
        props = o.sides[side]
        xyz = u.ll_xyz(values)  # converted from gcd to xyz
        # Following lines are merely for test/verification
        # sides = {tuple(o.h9side(v).tolist()): v for v in xyz}
        # if np.any(sides != side):
        #     print(f'An example in {side} is out of context.')
        uvw = ak.so(xyz)  # these are now octahedral values.
        # Following lines verifies inverse operation of projection.
        oct_bk = ak.os(uvw)
        if not np.allclose(xyz, oct_bk):
            print(f'Projection error found for {side} using AK projection.')
        # places = ex[side].keys()
        for (gcd, s_pt, o_pt) in zip(values, xyz, uvw):
            h9a = o.enc(o_pt, side)
            if h9a is None:
                print(f'GCD {gcd}, S{s_pt} O{o_pt} was not recognised as a legitimate value.')
                continue
            uvw_i = o.dec(h9a)
            h9b = o.enc(uvw_i[0])
            if np.allclose(o_pt, uvw_i) and h9a == h9b:
                print(f'{gcd}: {h9a}')
                continue
            print(f'\nTest {side} {gcd}; Oct:{o_pt}; Raw:{h9a}')
            print(f'OC:{o_pt}=>{uvw_i}')
            print(f'H9:{h9a}=>{h9b}')
