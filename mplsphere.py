# This draws onto a spherical projection of the world
# using the mpl basemap
from mpl_toolkits.basemap import Basemap
from utility.spheremap import SphereMap, SphereMapHalfHex
from utility.spherical import Spherical
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def ak(oct_pt: tuple):
    # Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
    # oct_pt is a Euclidean point on the surface of a unit octagon.
    a = 3.227806237143884260376580641604959964752197265625  # 𝛂 - vis. Kaseorg.
    e = 0.25  # Kaseorg's exponent is 0.25
    uvw = np.array(oct_pt) + 0.000000001
    pi_uvw = (np.pi * uvw) / 2.0
    t_uvw = np.tan(pi_uvw)
    xu, xv, xw = t_uvw
    u2, v2, w2 = t_uvw ** 2
    y0p = xu * (v2 + w2 + a*w2*v2) ** e
    y1p = xv * (u2 + w2 + a*u2*w2) ** e
    y2p = xw * (u2 + v2 + a*u2*v2) ** e
    pv = np.array([y0p, y1p, y2p])
    return pv / np.linalg.norm(pv, keepdims=True)


def convert(px: list):
    return [ak(pt) for pt in px]


def depict_gon(base, h_hex, recurse):
    shapes = h_hex.poly(lv=recurse, kind=2)
    for (s_id, s_col, s_pts, centre) in shapes:
        pts = convert(s_pts)
        pts.append(pts[0])
        for i in range(len(pts) - 1):
            # These are UVX.
            (p1a, p1o), (p2a, p2o) = sp.xyz_ll(pts[i]), sp.xyz_ll(pts[i+1])
            try:
                base.drawgreatcircle(p1o, p1a, p2o, p2a, color='black', linewidth=3.)
            except:
                print(f'[{p1a}, {p1o}] to [{p2a}, {p2o}] failed.')


def depict(base, h_hex, recurse):
    shapes = h_hex.fract(lv=recurse, kind=0)
    for (s_id, s_col, s_pts) in shapes:
        for i in range(len(s_pts) - 1):
            (p1a, p1o) = s_pts[i]
            (p2a, p2o) = s_pts[i+1]  # lon1, lat1, lon2, lat2
            base.drawgreatcircle(p1o, p1a, p2o, p2a, linewidth=2., color='k', del_s=1.0)


if __name__ == '__main__':
    # set up orthographic map projection with
    # perspective of satellite looking down at 45N, 100W.
    # use low resolution coastlines.
    map_name, depth = 'octa_v2', 2
    sphere = SphereMap(f'assets/maps/{map_name}.json')
    sp = Spherical

    mpl.rcParams['figure.frameon'] = False
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.pad_inches'] = 0
    mpl.rcParams['figure.figsize'] = (60, 60)
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.axis('off')
    #  'c': 10000
    #  'l': 1000.
    #  'i': 100.
    #  'h': 10.
    #  'f': 1.
    m = Basemap(projection='ortho', lat_0=51, lon_0=-2, resolution='l')
    # m = Basemap(projection='cyl', resolution='i')
    m.fillcontinents(color='lightgreen', lake_color='cornflowerblue')
    m.drawmapboundary(fill_color='cornflowerblue')

    # This uses the *octagon* as a fundamental shape.
    sphere.generate(depth)
    for side in sphere.sides.values():  # each face.
        for hh in range(3):
            depict_gon(m, side.get_hh(hh), depth)

    plt.savefig('6kc.png', dpi=100, format='png')
