import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Depict OctahedronMap made of 'F' Flats


def side_f(o, side, _n):  # octahedral-F for side.
    p2f = Util.tri_eff(_n)
    _ud = o.side_ud[side]
    spt = Util.d2_3(p2f, o.i3)
    rpt = spt if _ud == 'V' else spt @ Util.rz(np.pi)
    _th = o.oct_th[_ud]           # the transpose of a rotation is it's inverse.
    _mat = o.matrices[side]
    return rpt @ (_mat.T @ Util.rz(_th)).T  # rotate and adjust.


if __name__ == '__main__':
    octa = H9Octahedron()
    n = 10000
    s = {face: side_f(octa, face, n) for face in octa.matrices}
    fx = []
    for face, pts in s.items():
        obj = octa.sides[face]
        th = obj.grid_theta + obj.theta
        xy = obj.offs
        ud = obj.ud     # use this for theta
        r_mat = octa.matrices[face]  # get the face rotation.
        px = Util.d3_2(pts @ (r_mat.T @ Util.rz(th))) + xy  # rotate and adjust to plane
        fx.append(px)
    pts = np.vstack(fx)
    Display.show_pts_2d(pts, octa.glx, octa.gly, 'F Map')
