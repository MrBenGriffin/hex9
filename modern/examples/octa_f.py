import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Depict an Octahedron made of 'F'


def side_f(o, side, n):  # octahedral-F for side.
    p2f = Util.tri_eff(n)
    ud = o.side_ud[side]
    spt = Util.d2_3(p2f, o.i3)
    rpt = spt if ud == 'V' else spt @ Util.rz(np.pi)
    o_th = o.oct_th[ud]           # the transpose of a rotation is it's inverse.
    r_mat = o.matrices[side]
    return rpt @ (r_mat.T @ Util.rz(o_th)).T  # rotate and adjust.


if __name__ == '__main__':
    lm = (-1.1, 1.1)
    octa = H9Octahedron()
    n = 10000
    s = {face: side_f(octa, face, n) for face in octa.matrices}
    pts = np.vstack(list(s.values()))
    Display.show_pts_3d(pts, lm, lm, lm)
