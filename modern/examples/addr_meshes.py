import numpy as np
from matplotlib.collections import PolyCollection
from modern.grid_h9 import GridH9
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display
from modern.ak import AK
from modern.util import Util

# Using the json file 'examples'
# generate the H9 enmeshed nest of half-hexagons that represent the location
# of each address accordingly, These are then rotated and offset onto the 2D octahedral map projection
# such that they may be overlaid on the relevant map.

if __name__ == '__main__':
    o = H9Octahedron()
    u = Util()
    grid = GridH9()
    ak = AK()
    ex = u.json_load('examples.json')
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
    Display.poly_2d(all_polys, o.glx, o.gly, names, 'meshes.svg')

