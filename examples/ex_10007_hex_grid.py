# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 HexGrid generation proofs-of-concept
This does not print much unless there are failures!
Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
"""


import numpy as np
from hhg9.h9 import H9_RA, H9O
from hhg9.h9.addressing import reg_hex_digits, hex_digits_reg, hex_digits, hex_layer
from hhg9.base import Registrar, Points
from hhg9.h9.region import regions_xy, xy_regions


def chain_generator(initial_seed, depth, props=H9_RA.props, modes=H9_RA.modes):
    """Generator for comprehensive region chain generation"""

    def _recurse(current_chain):  # Recursive Closure
        if len(current_chain) - 2 == depth:  # Stop condition
            yield current_chain
            return

        seed = current_chain[-1]  # Get the current seed (last element)

        # INLINED LOOKUP (No function overhead)
        # Note: Since 'seed' is a single integer here,
        # props[modes[seed]] returns a 1D array (the children).
        children = props[modes[seed]].flatten()

        for child in children:  # Iterate and dive deeper
            yield from _recurse(current_chain + [child])  # Create new list

    yield from _recurse([initial_seed])  # yield from closure.


def get_data(reg: Registrar, depth):
    """Load up global sample data"""
    # grab generation for given depth
    b_oct = reg.domain('b_oct')
    all_rgn = [   # these are 0..11
        list(chain_generator(H9_RA.proto[0], depth)),
        list(chain_generator(H9_RA.proto[1], depth))
    ]
    rgn = H9_RA.rid2cell[np.array(all_rgn)]  # cell addresses.
    sides = []
    for oc in range(8):  # all octants
        rgc = rgn[H9O.oid_mo[oc]]
        xym = regions_xy(rgc)
        xy = xym[:, :-1]
        sides.append(Points(xy, b_oct, H9O.oid_cmp[oc]))
    globe = Points.concat(sides)
    return globe  # This will return six points per hexagon.


def layer_hex_roundtrip(depth: int = 0):
    """
    What is 'depth' here?
    At depth = 0, there are 12 hexagons, which take 2 bytes.
    How long should the [(octant)]: region chain be?
    If we have the octant, we don't really need a root indicative as a part of the chain.
    But maybe that's something to consider downstream. (it serves value in the cell-chain)
    Therefore, let's ignore the first (root).
    We then have the region/cell itself, which (being one of c2) is enough to give us a hex address.
    So, eg [0, 6] = net_mode 0, region 6. (c2=0)
    """
    reg = Registrar()
    b_oct = reg.domain('b_oct')

    globe = get_data(reg, depth)
    hxd = hex_layer(globe, layer=depth)
    ah, inv = np.unique(hxd, axis=0, return_inverse=True)
    # counts per unique row, based on the inverse
    bins = ah.shape[0]
    if bins != 12*9**depth:
        print(f'Depth {depth}; {3*6**depth} bins were expected; {bins} were found')
    # Because there are 6 regions per hexagon, we should see 6 in each bincount.
    cnt = np.bincount(inv, minlength=bins)
    mis_keys = np.flatnonzero((cnt != 6))
    for k in mis_keys:
        rows = np.flatnonzero(inv == k)
        hx_key = ah[k]
        print("key:", k.tolist())
        print("hex:", hx_key.tolist())
        print("rows:", rows.tolist())
        print()


if __name__ == "__main__":
    for depth in range(5):
        layer_hex_roundtrip(depth)
