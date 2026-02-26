# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
uses hand-selected set of addresses and
round-trips them, reporting on a series of measures and metrics,
reporting to console.
# ⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed; after rewriting formatter)
25 November 2025 (passed)
"""
import json
import numpy as np
from hhg9 import Registrar, Points
from hhg9.formats import OctahedralH9, DMS
from hhg9.algorithms.distance import wgs84
from hhg9.h9 import H9_RA, H9O
from hhg9.h9.addressing import Style
import hhg9.h9.region as rg


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9f formats.
    """
    reg = Registrar()  # Manage Domains & Projections
    h9f = OctahedralH9(reg)  # formatter.
    dms = DMS(reg)
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    b_oct.register_format(h9f)
    g_gcd.register_format(dms)
    b_oct.set_warp('src/l4_polished.npz')

    locs = {
        'centroids': {
            "bad": {"name": "poor", "lat": -71.7856153167678000,	"lon": 0.4539945781898160},
            "EAV": {"name": "centroid", "lat": 1.860751, "lon": 43.328763},
            "NAV": {"name": "centroid", "lat": 35.952813, "lon": 18.328413},
            "NEV": {"name": "centroid", "lat": 36.477125, "lon": 66.347867},
            "NPΛ": {"name": "centroid", "lat": 62.739925, "lon": 140.52874},
            "EPΛ": {"name": "centroid", "lat": 32.801097, "lon": 159.38627},
            "NEΛ": {"name": "centroid", "lat": 33.290631, "lon": 115.36871},
            "NWΛ": {"name": "centroid", "lat": 62.739925, "lon": -39.47126},
            "WAΛ": {"name": "centroid", "lat": 32.801097, "lon": -19.44909},
            "NAΛ": {"name": "centroid", "lat": 33.290631, "lon": -64.94493},
            "WPV": {"name": "centroid", "lat": 1.860751, "lon": -136.67064},
            "NPV": {"name": "centroid", "lat": 35.952813, "lon": -161.22549},
            "NWV": {"name": "centroid", "lat": 36.477125, "lon": -113.92880},
            "SAΛ": {"name": "centroid", "lat": -62.739925, "lon": 39.47126},
            "EAΛ": {"name": "centroid", "lat": -32.801097, "lon": 19.449089},
            "SEΛ": {"name": "centroid", "lat": -33.290631, "lon": 64.944931},
            "EPV": {"name": "centroid", "lat": -1.860751, "lon": 136.67064},
            "SPV": {"name": "centroid", "lat": -35.952813, "lon": 161.225489},
            "SEV": {"name": "centroid", "lat": -36.477125, "lon": 113.92880},
            "WAV": {"name": "centroid", "lat": -1.860751, "lon": -43.328763},
            "SAV": {"name": "centroid", "lat": -35.952813, "lon": -18.328413},
            "SWV": {"name": "centroid", "lat": -36.477125, "lon": -66.347867},
            "SPΛ": {
                "name": "centroid", "lat": -62.739925, "lon": -140.52874,
                "name": "bad girl", "lat": -45.372511693007, "lon": -148.055724862128
            },
            "WPΛ": {"name": "centroid", "lat": -32.801097, "lon": -159.899089},
            "SWΛ": {"name": "centroid", "lat": -33.093812, "lon": -115.138392}
        },
        'places': {
            "EAV": {
                "name": "Mogadishu, Somalia",
                "lat": 2.0469, "lon": 45.3354
            },
            "NAV": {
                "name": "Valletta, Malta",
                "lat": 35.8989, "lon": 14.5146
            },
            "NEV": {
                "name": "Samarkand, Uzbekistan",
                "lat": 39.6542, "lon": 66.9751
            },
            "NPΛ": {
                "name": "Oymyakon, Russia",
                "lat": 63.4649, "lon": 142.7866
            },
            "EPΛ": {
                "name": "Midway Atoll",
                "lat": 28.2074, "lon": -177.3754
            },
            "NEΛ": {
                "name": "Xi'an, China",
                "lat": 34.2667, "lon": 108.9500
            },
            "NWΛ": {
                "name": "Nuuk, Greenland",
                "lat": 64.1835, "lon": -51.7216
            },
            "WAΛ": {
                "name": "Canary Islands, Spain",
                "lat": 28.1189, "lon": -15.4347
            },
            "NAΛ": {
                "name": "Bermuda",
                "lat": 32.3078,
                "lon": -64.7505
            },
            "WPV": {
                "name": "Galapagos Islands, Ecuador",
                "lat": -0.9538,
                "lon": -90.9656
            },
            "NPV": {
                "name": "Great Salt Lake, USA",
                "lat": 41.1664,
                "lon": -112.5833
            },
            "NWV": {
                "name": "Denver, USA",
                "lat": 39.7392,
                "lon": -104.9903
            },
            "SAΛ": {
                "name": "Novolazarevskaya Station, Antarctica",
                "lat": -70.7766,
                "lon": 11.8227
            },
            "EAΛ": {
                "name": "Cape Town, South Africa",
                "lat": -33.9249,
                "lon": 18.4241
            },
            "SEΛ": {
                "name": "Kerguelen Islands",
                "lat": -49.3499,
                "lon": 69.3486
            },
            "EPV": {
                "name": "Jayapura, Indonesia",
                "lat": -2.5333,
                "lon": 140.7167
            },
            "SPV": {
                "name": "Lord Howe Island, Australia",
                "lat": -31.5567,
                "lon": 159.0869
            },
            "SEV": {
                "name": "Lake Eyre, Australia",
                "lat": -28.6808,
                "lon": 137.2650
            },
            "WAV": {
                "name": "Fernando de Noronha, Brazil",
                "lat": -3.8551,
                "lon": -32.4296
            },
            "SAV": {
                "name": "Rio de Janeiro, Brazil",
                "lat": -22.9068,
                "lon": -43.1729
            },
            "SWV": {
                "name": "Buenos Aires, Argentina",
                "lat": -34.6037,
                "lon": -58.3816
            },
            "SPΛ": {
                "name": "Amundsen-Scott South Pole Station",
                "lat": -89.9982,
                "lon": -139.2723
            },
            "WPΛ": {
                "name": "Wellington, New Zealand",
                "lat": -41.2865,
                "lon": 174.7762
            },
            "SWΛ": {
                "name": "Perth, Australia",
                "lat": -31.9523,
                "lon": 115.8613
            }
        }
    }

    print('Selection points, projected forwards and backwards, showing deviation ∂ in nanometres.')
    names = []
    ll = []
    locs2 = json_load('../assets/locations.json')
    for region in locs2:
        for spot in locs2[region]:
            names.append(f'{region}: {spot}')
            ll.append(tuple(locs2[region][spot]))
    for source, spots in locs.items():
        for k, v in spots.items():
            names.append(f'{k}: {v["name"]}')
            ll.append((v['lat'], v['lon']))

    pos = Points(np.array(ll), g_gcd)
    bry = reg.project(pos, [g_gcd, b_oct])  # spherical cart
    bel = reg.project(pos, [g_gcd, 'c_ell'])
    boc = reg.project(pos, ['c_ell', 'c_oct'])

    rtp = reg.project(bry, [b_oct, g_gcd])  # sph rt..
    dif = wgs84(pos.coords, rtp.coords) * 1e+9
    oc, mo = bry.cm()
    cxx = rg.xy_regions(bry.coords, mo)
    xym = rg.regions_xy(cxx)
    uvr = Points(xym[:, :2], components=oc, domain=b_oct)
    rgx = [H9_RA.cell2rid[i] for i in cxx]
    rga = [''.join([f'{a:x}' for a in p]) for p in rgx]
    hxx = h9f.format(bry, None, 'r36')
    ub1 = h9f.revert(hxx, Style.UR64)

    rrp = reg.project(uvr, [b_oct, g_gcd])
    ltp = reg.project(ub1, [b_oct, g_gcd])
    ltp.domain = g_gcd
    rif = wgs84(pos.coords, rrp.coords) * 1e+9
    lif = wgs84(pos.coords, ltp.coords) * 1e+9

    bcc, bmo = bry.cm()
    for idx, name in enumerate(names):
        i = bcc[idx]
        mo = bmo[idx]
        ci = H9O.oid_cmp[i]
        nm = H9O.oid_str[i]
        mode = H9O.oid_mo[i]
        print(f'{name:<24} {ci}, mode:{mode}')
        print(f'Regions {cxx[i]}')
        print(f'{nm} {pos[i]:dms} (Reference Coordinates)')
        print(f'{nm} {ltp[i]:dms} (Label RT Coordinates)')
        print(f'{nm} {ltp[i]:dms} (Label GCD Coordinates)')

        print(f'∂{dif[i]:.6f}nm (roundtrip via GCD<->Barycentric)')
        print(f'∂{rif[i]:.6f}nm (roundtrip via GCD<->Bary Regions)')
        print(f'∂{lif[i]:.6f}nm (roundtrip via GCD<->Hex9 Label)')
        print(f'H9.adr:{bry[i]:h9}')
        print(f'H9.key:{bry[i]:h9.k}')
        print(f'H9.adr.13:{bry[i]:h9.13}')
        print(f'H9.u64.13:{bry[i]:h9.u13}')
        if name == 'NWA: Stonehenge':
            print(f'ECEF: {bel.coords[i][0]:.8f},{bel.coords[i][1]:.8f},{bel.coords[i][2]:.8f}')
            print(f'OCTA: {boc.coords[i][0]:.8f},{boc.coords[i][1]:.8f},{boc.coords[i][2]:.8f}')
            print(f'BARY: {bry.coords[i][0]:.8f},{bry.coords[i][1]:.8f}')
            for layer in range(35):
                print(f'H9.key; layer {layer}:{bry[i]:h9.k{layer}}')
        print(f'Reference BRY: {bry.coords[i][0]:.18f},{bry.coords[i][1]:.18f}')
        print(f'Label RT  BRY: {uvr.coords[i][0]:.18f},{uvr.coords[i][1]:.18f}')
        print()
