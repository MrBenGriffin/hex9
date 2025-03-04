import numpy as np
from matplotlib.collections import PolyCollection
from modern.grid_h9 import GridH9
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Display the C1&C2 colours of each of the sides of the Octahedral Map.
# This has two ways of generating sides - one via enmesh, and the other via poly.
# There seems to be a 120º difference coming in somewhere.

if __name__ == '__main__':
    #         0            1         2       3           4         5          6          7           8
    h8 = ['#0d0887', '#6100a7', '#8e0ca4', '#b42e8d', '#d24f71', '#e97257', '#f99a3e', '#fdc627', '#f0f921']
    o = H9Octahedron()
    u = Util()
    g = GridH9()
    all_polys = []
    ro = 2*np.pi/3.  # 2π/3 = 120
    for key in o.sides.keys():
        side = o.sides[key]  # side.grid_theta +
        b0 = np.array([g.poly(c1, side.ud) for c1 in range(3)])  # Polys are reached via c1 (0,1,2)
        b2 = b0 @ u.r2d(side.theta + ro) + side.offs  # On the map: V and L are inverted.
        for c1 in range(3):
            hh = b2[c1]
            hx = side.c2[c1]
            px = PolyCollection([hh], alpha=.40, edgecolor='k', linewidth=1)
            px.set_facecolor(h8[hx*2])
            all_polys.append(px)
    k8 = ['#d24f71', '#8e0ca4', '#f99a3e']
    for key in o.sides.keys():
        side = o.sides[key]  # side.grid_theta + side.c2[c1]
        for c1 in range(3):
            xy = side.decode_xy(side.hx[c1])  # hx is in c2 order.
            poly = g.enmesh(xy, side.c2s, 1, True)
            emp = np.array([poly]) @ u.r2d(side.theta) + side.offs + [0.02, 0.02]
            px = PolyCollection(emp, alpha=.40, edgecolor='k', linewidth=1)
            px.set_facecolor(k8[c1])
            all_polys.append(px)
    Display.poly_2d(all_polys, o.glx, o.gly, None, 'c2.svg')
