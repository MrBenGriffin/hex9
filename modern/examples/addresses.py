import numpy as np
from modern.ak import AK
from modern.grid_h9 import Style
from modern.util import Util
from modern.octahedron_h9 import H9Octahedron

# Using the json file 'examples'
# generate the H9 addresses for each location, and then use the inverse to reconstitute the address.
# These are then printed out; and a relevant difference/error message is printed if necessary.
# For example, Stonehenge (51.178863, -1.826177) will output
# Stonehenge; [51.178863, -1.826177]:
# NAΛ2035211610266553407865006553346V; (the standard H9 global address)
# 22035211G1026G55340TXG500G553346     (the 678 extended address for NAΛ (aka: 2Λ))
# zzoe52iig1026g5534otxgs00g553346     (the half-hex extended address for NAΛ (aka: 2Λ/z))


if __name__ == '__main__':
    ak = AK()  # An implementation of octagon/sphere projection.
    u = Util()
    o = H9Octahedron()
    ex = u.json_load('examples.json')
    for side, values in ex.items():
        props = o.sides[side]
        places = list(values.keys())
        lls = list(values.values())
        xyz = u.ll_xyz(lls)  # converted from gcd to xyz
        # Following lines are merely for test/verification
        sides = o.pts_faces(xyz)
        if np.any(sides != side):
            print(f'An example in {side} is out of context.')
        uvw = ak.so(xyz)  # these are now octahedral values.
        # Following lines verifies inverse operation of projection.
        oct_bk = ak.os(uvw)
        if not np.allclose(xyz, oct_bk):
            print(f'Projection error found for {side} using AK projection.')
        for (name, gcd, s_pt, o_pt) in zip(places, lls, xyz, uvw):
            h9a = o.enc(o_pt, side)
            uvw_i = o.dec(h9a)
            ab = props.addr_pt(o_pt)
            h9b = o.enc(uvw_i[0])
            ext = o.grid.encode(ab, props.c2s, 32, Style.EXTENDED)
            hh = o.grid.encode(ab, props.c2s, 32, Style.HALFHEX)
            if np.allclose(o_pt, uvw_i) and h9a == h9b:
                print(f'{name}; {gcd}: {h9a}; EXT:{ext}; HH:{hh}')
                continue
            print(f'\nTest {side}: {name}, {gcd}; Oct:{o_pt}; H9:{h9a}')
            print(f'OC:{o_pt}=>{uvw_i}')
            print(f'H9:{h9a}=>{h9b}; Ext:{ext}')
