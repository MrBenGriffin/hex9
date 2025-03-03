import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# For each 'side' of the Octahedron...
# =>rnd_pts_for_side
# Generate points for a random Octahedron for side 0,0,0.
# Find the signs of 'side' and convert the points to this.
# Identify vertex of 'side' representing z=±1 (N/S)
# Accordingly add a ball to that vertex.
# =>side_to_plane
# using the plane matrix and orientation,
# Rotate the points to the z-plane (all z are the same).
# main
# flatten the points and display.


def rnd_pts_for_side(o, side, n):
    opt = Util.oct0_rnd(n)  # random points on face.0
    sign = o.signs[side]  # find orientation.
    spt = np.copysign(opt, sign)  # opt is corrected.
    if side[0] == 'N':  # Add a ball to the N/S face.
        bl = Util.ball_rnd(o.ov['N'])
    else:
        bl = Util.ball_rnd(o.ov['S'])
    return np.vstack([spt, bl])  # add to the points.


def side_to_plane(o, side, pts):  # given a side rotate to Z
    ud = o.side_ud[side]  # use this for theta
    o_th = o.oct_th[ud]  # use for orientation.
    r_mat = o.matrices[side]  # get the face rotation.
    return pts @ (r_mat.T @ Util.rz(o_th))  # rotate and adjust.


if __name__ == '__main__':
    octa = H9Octahedron()
    for face in octa.matrices:
        srd = rnd_pts_for_side(octa, face, 50000)
        trx = side_to_plane(octa, face, srd)
        # print(face, trx[0])
        pts = Util.d3_2(trx)
        Display.show_pts_2d(pts, (-.9, .9), (-.9, .9), face)
