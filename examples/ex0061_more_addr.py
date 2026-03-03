# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
uses hand-selected set of addresses and
round-trips them, reporting on a series of measures and metrics,
reporting to console.
# ⚠️ 'nm' == NANOMETRES, not NAUTICAL MILES.

Last Tested
02 March 2026 0.1.1a1 (passed; found and fixed a warp bug)
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


def test(pts, pl):
    out = reg.project(pts, pl)
    ret = reg.project(out, pl[::-1])
    err = np.max(np.abs(ret.coords - pts.coords))
    journey = '->'.join([p.name for p in pl])
    print(f'max error {journey}: {err:.16f}')


if __name__ == '__main__':
    """
        Convert a set of locations into a set of h9f formats.
    """
    reg = Registrar()  # Manage Domains & Projections
    h9f = OctahedralH9(reg)  # formatter.
    dms = DMS(reg)
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    c_oct = reg.domain('c_oct')
    b_oct.register_format(h9f)
    g_gcd.register_format(dms)

    lat_d = 0.000000000001
    # lat_d = 0.00000015
    locs = {
        'errors': {
            "011": {"name": "Equator NE", "lat": lat_d, "lon": 45.00000000},
            "002": {"name": "Equator SE", "lat": -lat_d, "lon": 45.00000000},
            "003": {"name": "Equator NW", "lat": lat_d, "lon": -45.00000000},
            "004": {"name": "Equator SW", "lat": -lat_d, "lon": -45.00000000},
            "005": {"name": "Equator NFE", "lat": lat_d, "lon": 135.00000000},
            "006": {"name": "Equator SFE", "lat": -lat_d, "lon": 135.00000000},
            "007": {"name": "Equator NFW", "lat": lat_d, "lon": -135.00000000},
            "008": {"name": "Equator SFW", "lat": -lat_d, "lon": -135.00000000}
        },
        'centroids': {
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
            "odd": {"name": "Ex ECEF NE", "lat": -32.13561916, "lon": 47.14912943},

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

    p1 = reg.project(pos, [g_gcd, b_oct])
    p2 = reg.project(bry, [b_oct, g_gcd])
    dxr = wgs84(pos.coords, p2.coords) * 1e+9

    rtp = reg.project(bry, [b_oct, g_gcd])  # sph rt..
    dif = wgs84(pos.coords, rtp.coords) * 1e+9
    oc, mo = bry.cm()
    cxx = rg.xy_regions(bry.coords, mo)
    xym = rg.regions_xy(cxx)
    uvr = Points(xym[:, :2], components=oc, domain=b_oct)
    rgx = [H9_RA.cell2rid[i] for i in cxx]
    rga = [''.join([f'{a:x}' for a in p]) for p in rgx]
    hxx = h9f.format(bry, None, 'r36')
    h64k = h9f.format(bry, None, 'uk')
    h64a = h9f.format(bry, None, 'ua')
    ha36 = h9f.format(bry, None, 'ua36')
    h36 = h9f.format(bry, None, '36')
    r36_str = h9f.revert(h36)                 # from full H9 layer-36 string
    ruk = h9f.revert(h64k, Style.UH64K)       # key/id (representative)
    rua = h9f.revert(h64a, Style.UH64A)       # reversible u64 (representative)
    r36_u64 = h9f.revert(ha36, Style.UH64A)   # u64 version of layer-36 (representative)

    g36 = reg.project(r36_str, [b_oct, g_gcd])
    u36 = reg.project(r36_u64, [b_oct, g_gcd])
    atp = reg.project(rua, [b_oct, g_gcd])

    d36 = wgs84(pos.coords, g36.coords) * 1e+9
    dmf = wgs84(pos.coords, u36.coords) * 1e+9
    daf = wgs84(pos.coords, atp.coords)
    l36 = h36.split('\n')
    l64a = h64a.split('\n')
    l64k = h64k.split('\n')

    bcc, bmo = bry.cm()
    for idx, name in enumerate(names):
        oid = bcc[idx]        # octant id (0..5)
        mo = bmo[idx]
        ci = H9O.oid_cmp[oid]
        nm = H9O.oid_str[oid]
        mode = H9O.oid_mo[oid]
        print(f'{name:<24} {ci}, mode:{mode}')
        print(f'Regions {cxx[idx]}')
        print(f'{nm} {pos[idx]:dms} (Reference Coordinates)')
        print(f'{nm} {rtp[idx]:dms} (Reverted Coordinates)')
        print(f'{nm} {l36[idx]} (H9 Layer 36)')
        print(f'{nm} {l64a[idx]} (U64A Address)')
        print(f'{nm} {l64k[idx]} (U64K Identity)')

        print(f'∂{dxr[idx]:.6f}nm (roundtrip via GCD<->Barycentric<->GCD)')
        print(f'∂{dif[idx]:.6f}nm (roundtrip via GCD<->Barycentric)')
        print(f'∂{d36[idx]:.6f}nm (roundtrip via GCD<->Bary Regions)')
        print(f'∂{d36[idx]:.6f}nm (roundtrip via GCD<->H9 Layer 36)')
        print(f'∂{dmf[idx]:.6f}nm (roundtrip via GCD<->H9 U64 36)')
        print(f'∂{daf[idx]:.6f}m (roundtrip via GCD<->H9 U64 Reversible)')

        # print(f'H9.adr:{bry[i]:h9}')
        # print(f'H9.key:{bry[i]:h9.k}')
        # print(f'H9.u64k:{bry[i]:h9.uk}')
        # print(f'H9.u64a:{bry[i]:h9.ua}')
        if name == 'NWA: Stonehenge':
            # print(f'ECEF: {bel.coords[oid][0]:.8f},{bel.coords[oid][1]:.8f},{bel.coords[oid][2]:.8f}')
            # print(f'OCTA: {boc.coords[oid][0]:.8f},{boc.coords[oid][1]:.8f},{boc.coords[oid][2]:.8f}')
            print(f'BARY: {bry.coords[oid][0]:.8f},{bry.coords[oid][1]:.8f}')
            for layer in range(35):
                print(f'H9.key; layer {layer}:{bry[oid]:h9.k{layer}}')
        print(f'Reference BRY: {bry.coords[oid][0]:.18f},{bry.coords[oid][1]:.18f}')
        print(f'Label RT  BRY: {uvr.coords[oid][0]:.18f},{uvr.coords[oid][1]:.18f}')
        print()
