import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display


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
    octa = H9Octahedron()
    nwa = rnd_pts_for_side(octa, 'NWA', 1000)
    nea = rnd_pts_for_side(octa, 'NEA', 1000)
    nwp = rnd_pts_for_side(octa, 'NWP', 1000)
    nep = rnd_pts_for_side(octa, 'NEP', 1000)
    swa = rnd_pts_for_side(octa, 'SWA', 1000)
    sea = rnd_pts_for_side(octa, 'SEA', 1000)
    swp = rnd_pts_for_side(octa, 'SWP', 1000)
    sep = rnd_pts_for_side(octa, 'SEP', 1000)
    pts = np.vstack([nwa, nea, nwp, nep, swa, sea, swp, sep])
    Display.show_pts_3d(pts, (-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1), 'quick')
