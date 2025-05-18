import numpy as np
from modern.octahedron import Octahedron
from modern.util import Util
from modern.display import Display

# This generates a set of octahedral points and then
# Displays them in 3D. It shows that the sides
# match correctly to the 3D octahedron.
# Util: oct0_rnd(), ball_rnd()
# Octahedron: signs[], ov[] (and explicit keys)
# Display: show_pts_3d()

# def rnd_pts_for_side(octal, n):
#     """ This uses the Octant class """
#     from modern.util import Util
#     opt = Util.oct0_rnd(n)  # random points on face.0
#     spt = np.copysign(opt, octal.signs)  # opt is corrected.
#     bl = Util.ball_rnd(octal.apex, 500)
#     return np.vstack([spt, bl])  # add to the points.


def rnd_pts_for_side(o, side, n):
    opt = Util.oct0_rnd(n)  # random points on face.0
    sign = o.signs[side]  # find orientation.
    spt = np.copysign(opt, sign)  # opt is corrected.
    if side[0] == 'N':  # Add a ball to the N/S face.
        bl = Util.ball_rnd(o.ov['N'], 500)
    else:
        bl = Util.ball_rnd(o.ov['S'], 500)
    return np.vstack([spt, bl])  # add to the points.


if __name__ == '__main__':
    octa = Octahedron()
    sides = [rnd_pts_for_side(octa, face, 4000) for face in octa.matrices]
    pts = np.vstack(sides)
    Display.show_pts_3d(pts, (-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1))
