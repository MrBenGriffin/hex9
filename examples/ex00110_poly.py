"""
Part of the H9 project - Visualisation of neighbouring Half-hexagons.
This serves to depict the calculations used to identify neighbours.
The neighbourhood logic seemed to be quite elusive, and this is a demonstrator.
Last Tested 13 August 2025 √
"""
import numpy as np
from matplotlib import pyplot as plt
from hhg9 import Registrar, H9Engine
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS
from support import Util
from matplotlib.patches import Polygon


def grid_lines(h9, ax):
    """
    Plots the underlying triangular 'region' grid of the h9 grid.
    """
    xb = np.array([h9.TL*10, h9.TR*10])
    xt = h9.R3 * xb
    h_conditions = [h9.VC * c for c in range(-8, 9)]
    v_conditions = [h9.Ẇ * c for c in range(-8, 9)]
    for h in h_conditions:
        ax.axhline(h, color='gray', linestyle='-', alpha=0.25)
    for h in v_conditions:
        ax.plot(xb, xt + h, color='red', linestyle='-', alpha=0.25)
        ax.plot(xb, h - xt, color='red', linestyle='-', alpha=0.25)


def get_hh(h9e, address, c1):
    """
    Decodes a URI address to find the barycentric vertices of its
    final half-hexagon.
    """
    p_uri = address[:, -2]
    p_mode = h9.ugc_lut[p_uri, h9.mode]
    return h9e.poly_hh[p_mode, c1]


if __name__ == '__main__':
    u = Util()
    reg = Registrar()  # Manage Domains & Projections
    g_gen = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    p_plt = PlatePixel(reg)             # 2D Pixel Cartesian Domain

    h9f = OctahedralH9()            # formatter.
    g_gen.register_format(DMS())
    b_oct.register_format(h9f)
    h9 = H9Engine()

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)           # [g_gen, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]
    ppg = PlatePixelGCD(reg)
    acc = ak.set_accuracy(0.00000000001)
    locs = u.json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = g_gen.adopt(np.array([spot]))
    sp0 = reg.project(ll0, [g_gen, c_ell])  # spherical cart
    oc0 = reg.project(sp0, [c_ell, c_oct])
    bc0 = reg.project(oc0, [c_oct, b_oct])
    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    co, mo = bc0.cm()
    uri = h9.ugc_regions(bc0.coords, mo, acc)
    fig = plt.figure(figsize=(14, 17), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    nut = h9.neighbour_lut()
    nof = h9.local_offset_lut()
    for depth in range(3, 30):  # 10
        size = depth + 2     # Size of regions in POI
        poi = uri[:, :size]
        par = poi[:, :-1]   # PARent is one less size than POI
        poi_region = poi[:, -2]  # final value holding region is at -2.
        par_region = par[:, -2]  # final value holding region is at -2.
        c1 = h9.c1(poi, depth)  # c1 of POI, via its parent.
        pmo = h9.ugc_lut[par_region, h9.mode]  # parent mode
        imo = h9.ugc_lut[poi_region, h9.mode]  # poi mode
        nbm = nut[poi_region, pmo, c1]  # neighbour_lut returns region and neighbour's parent mode.
        nbr = nbm[:, 0]
        nhb = poi.copy()
        nhb[:, -2] = nbr  # neighbour's region
        npm = nbm[:, 1]   # neighbour's parent mode
        sib = np.array(npm == pmo, dtype=np.uint8)
        # (2, 2, 3, 2) :
        # poi = 0x35 (imo=1) poi=0x39 (imo=1)
        # need to get c1 both local and neighbour: sib 1,0
        # sib
        n_off = nof[imo, npm, c1, sib]  # This returns the offset via modes, c1, bool.
        original_hh = get_hh(h9, poi, c1)
        neighbor_hh = get_hh(h9, nhb, c1) + n_off
        ax = fig.add_subplot(6, 5, depth)  # (*nrows*, *ncols*, *index*)
        ax.set_xlim(h9.TL * 3.2, h9.TR * 3.2)  # Use TL/TR with a 10% margin
        ax.set_ylim(h9.VF * 3.2, h9.ΛC * 3.2)  # Use VF/ΛC with a 10% margin
        ax.set_aspect('equal', adjustable='box')
        grid_lines(h9, ax)
        r_col = 'black' if pmo[0] == 1 else 'blue'
        l_col = 'yellow' if imo[0] == 1 else 'cyan'
        n_col = 'green' if sib[0] else 'red'
        ax.add_patch(Polygon(original_hh[0], linewidth=3, edgecolor=l_col, facecolor=r_col, alpha=0.95))
        ax.add_patch(Polygon(neighbor_hh[0], linewidth=3, edgecolor=n_col, facecolor=n_col, alpha=0.35))
        ax.set_title(f"{depth}, {imo[0]}{npm[0]}{c1[0]}{sib[0]} "
                     f"POI:{poi_region[0]:02x}; NOI:0x{nbr[0]:02x}")
    plt.show()
