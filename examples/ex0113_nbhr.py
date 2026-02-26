# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - using locations.json,
Test and report the neighbours, showing deviation from original.
There is normally a least-hex_layer accuracy error

Last Tested
26 December 2025 0.1.0a4 (passed - but questionable)
16 December 2025 0.1.0a3 (passed - but of questionable value)
12 October 2025 (passed)

"""
import json
import numpy as np
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84
from hhg9.formats import DecimalDegrees
from hhg9.h9.region import region_neighbours, xy_regions, regions_xy
from hhg9.h9.addressing import neighbours


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


if __name__ == '__main__':
    np.set_printoptions(formatter={'int': lambda x: f'{x:02x}'})
    reg = Registrar()  # Manage Domains & Projections
    locs = json_load('../assets/locations.json')
    decimal = DecimalDegrees(reg)
    b_oct = reg.domain('b_oct')
    b_oct.set_warp('src/l4_polished.npz')
    g_gcd = reg.domain('g_gcd')
    g_gcd.register_format(decimal)

    accuracy = 36
    loc_names = list(locs.keys())
    for name in loc_names:
        print(name)
        region = locs[name]
        spots = np.array(list(region.values()))
        names = np.array(list(region.keys()))
        ll0 = Points(spots, g_gcd)
        bc0 = reg.project(ll0, [g_gcd, b_oct])  # spherical cart
        cmp = tuple(bc0.components[0])
        co, mo = bc0.cm()
        uri = xy_regions(bc0.coords, mo, accuracy)
        nbr, nmo = region_neighbours(uri)
        ub2, rmo = region_neighbours(nbr)
        xyb = regions_xy(uri)
        xyn = regions_xy(nbr)
        xyr = regions_xy(ub2)
        npt = neighbours(bc0, accuracy)
        nll = reg.project(npt, [b_oct, g_gcd])
        dx = wgs84(nll.coords, ll0.coords)
        nll.domain = g_gcd
        for n, d, a, b, c, p0, p1, p2, nl, og in zip(names, dx, uri, ub2, nbr, xyb, xyn, xyr, nll, ll0):
            print(f'{n:<30}: {np.abs(p2-p0)};\n'
                  f'Original :{p0}; {a}\n'
                  f'Roundtrip:{p2}; {b}\n'
                  f'Neighbour:{p1}; {c}\n'
                  f'∂x: {d};\nNB: {nl:deg}\nOG: {og:deg}\n')
        done = True
