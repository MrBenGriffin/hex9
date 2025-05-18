import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from modern.hhg_tetrahedral import HHGHH
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display
from modern.ak_projection import AK
from modern.util import Util

# Work in progress - generating live animation of a rotating triangle.
# This will be used for populating a grid with new stuff.


def r2d(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st], [st, ct]])


def init():
    return patch,


def animate(i):
    global vx, matrix
    vx = vx @ matrix
    patch.set_xy(vx)
    return [patch]


if __name__ == '__main__':
    o = H9Octahedron()
    u = Util()
    grid = HHGHH()
    ak = AK()
    ex = u.json_load('locations.json')
    cols = Display.colours(20, 'tab20')
    all_polys = []
    names = []
    i = 0
    for key, values in ex.items():
        side = o.sides[key]
        places = list(values.keys())
        uvw = ak.so(u.ll_xyz(list(values.values())))  # converted from gcd to xyz
        for (place, pt) in zip(places, uvw):
            a_pt = side.addr_pt(pt)
            polys = grid.enmesh(a_pt, side.c2s, 10, False)
            mapped = np.array(polys) @ u.r2d(side.grid_theta + np.pi) + side.offs
            pc = PolyCollection(mapped, alpha=.30, edgecolor='k', linewidth=0.1)
            pc.set_facecolors([cols[i]])
            names.append(place)
            i += 1
            all_polys.append(pc)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    matrix = r2d(0.01)
    vx = np.array([
        [0., -0.25],
        [0.5, 0.],
        [0., 0.25]
    ])
    vx[:, 0] += 5.2

    patch = patches.Polygon(vx, closed=True, fc='r', ec='r')
    ax.add_patch(patch)
    for x in all_polys:
        ax.add_collection(x)
    ax.set_aspect('equal', adjustable='box')

    # Must turn off showing plots in tool window!
    ani = animation.FuncAnimation(fig, animate, None, save_count=0, init_func=init, interval=10, blit=True)
    plt.show()
