import numpy as np
from matplotlib.collections import PolyCollection
from modern.hhg_tetrahedral import HHGTetrahedral
from modern.octahedron import Octahedron
from modern.octahedron_net import OctahedronNet
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display
from modern.ak_projection import AKProjection
# from modern.cn_projection import CNProjection
from modern.util import Util

# Using the json file 'examples'
# generate the H9 enmeshed nest of half-hexagons that represent the location
# of each address accordingly, These are then rotated and offset onto the 2D octahedral map projection
# such that they may be overlaid on the relevant map.
# Last run 11 May 2025 - looked ok?!

if __name__ == '__main__':
    o = Octahedron()
    net = OctahedronNet(o)
    h9 = H9Octahedron(o)
    u = Util()
    grid = HHGTetrahedral()
    ak = AKProjection()
    # cn = CNProjection()
    ex = u.json_load('locations.json')
    cols = Display.colours(20, 'tab20')
    all_polys = []
    names = []
    i = 0
    for key, values in ex.items():
        h_side = h9.sides[key]
        side = net.faces[key]
        places = list(values.keys())
        uvw = ak.so(u.ll_xyz(list(values.values())))  # converted from gcd to xyz
        for (place, pt) in zip(places, uvw):
            a_pt = h_side.addr_pt(pt)
            polys = grid.enmesh(a_pt, h_side.c2s, 10, False)
            if len(polys) > 0:
                mapped = side.place(polys)
                # mapped = np.array(polys) @ u.r2d(side.grid_theta + np.pi) + side.offs
                pc = PolyCollection(mapped, alpha=.30, edgecolor='k', linewidth=0.1)
                pc.set_facecolors([cols[i]])
                names.append(place)
                i += 1
                all_polys.append(pc)
            else:
                print(f'{place}, {pt} returned address {a_pt} but no polys were made.')
    Display.poly_2d(all_polys, net.glx, net.gly, names, 'meshes.svg')

