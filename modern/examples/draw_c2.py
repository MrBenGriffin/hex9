import numpy as np
from matplotlib.collections import PolyCollection
from modern.grid_h9 import GridH9
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Display the C1&C2 colours of each of the sides of the Octahedral Map.
# The ongoing issue is why there's an adjustment needed. This might
# Need to be looked at some point.

if __name__ == '__main__':
    #         0            1         2       3           4         5          6          7           8
    h8 = ['#0d0887', '#6100a7', '#8e0ca4', '#b42e8d', '#d24f71', '#e97257', '#f99a3e', '#fdc627', '#f0f921']
    adj = {'NEP': 1, 'NWP': 2, 'SEA': 2, 'SEP': 1, 'SWA': 0, 'SWP': 2, 'NWA': 0, 'NEA': 2}
    o = H9Octahedron()
    u = Util()
    g = GridH9()
    # s, a, b = o.h9side('NAΛ')
    # side = o.sides[s]
    # xy = side.decode_xy(a, '2')
    # poly = g.enmesh(xy, side.c2s, 1, True)
    all_polys = []
    ro = 2*np.pi/3.  # 2π/3 = 120
    # for key in o.sides.keys():
    #     side = o.sides[key]  # side.grid_theta +
    #     b0 = np.array([g.poly(c1, side.ud) for c1 in range(3)])  # Polys are reached via c1 (0,1,2)
    #     b2 = b0 @ u.r2d(side.grid_theta + np.pi + adj[key]*ro) + side.offs  # On the map: V and L are inverted.
    #     for c1 in range(3):
    #         hh = b2[c1]
    #         c2 = side.c2[c1]
    #         # hx = c2 * 3 + side.c1[c1]
    #         hx = c2
    #         # print(key, c1, c2, hx, h8[hx])
    #         px = PolyCollection([hh], alpha=.20, edgecolor='k', linewidth=1)
    #         px.set_facecolor(h8[hx])
    #         all_polys.append(px)
    k8 = ['#ff0000', '#ffff00', '#ff00ff']
    for key in o.sides.keys():
        side = o.sides[key]  # side.grid_theta + side.c2[c1]
        for c1 in range(3):
            xy = side.decode_xy(side.hx[c1])
            poly = g.enmesh(xy, side.c2s, 1, True)
            emp = np.array([poly]) @ u.r2d(side.grid_theta + np.pi) + side.offs
            px = PolyCollection(emp, alpha=.60, edgecolor='k', linewidth=1)
            px.set_facecolor(k8[c1])
            all_polys.append(px)
    Display.poly_2d(all_polys, o.glx, o.gly, None, 'c2.svg')
