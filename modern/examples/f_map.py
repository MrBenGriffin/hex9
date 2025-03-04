import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Depict OctahedronMap made of 'F' Flats
# This depends upon rotation theta sets for each side.
# The side_f function uses the face up/down to orient the points
# and likewise inverts the normal rotation in z accordingly.


def side_f(o, side, _n):  # octahedral-F for side.
    face = o.sides[side]
    spt = Util.d2_3(Util.tri_eff(_n), o.i3)
    rpt = spt if face.ud == 'V' else spt @ Util.rz(np.pi)  # why not 'Λ'?
    matrix = o.matrices[side]
    return rpt @ (matrix.T @ Util.rz(face.theta)).T  # rotate and adjust.


if __name__ == '__main__':
    octa = H9Octahedron()
    n = 10000
    s = {face: side_f(octa, face, n) for face in octa.matrices}
    fx = []
    ro = 2*np.pi/3.
    for face, pts in s.items():
        obj = octa.sides[face]
        o_th = obj.grid_theta + obj.theta  # use for orientation.
        xy = obj.offs
        r_mat = octa.matrices[face]  # get the face rotation.
        px = Util.d3_2(pts @ (r_mat.T @ Util.rz(o_th))) + xy  # rotate and adjust to plane
        fx.append(px)
    pts = np.vstack(fx)
    Display.show_pts_2d(pts, octa.glx, octa.gly, 'F Map')

