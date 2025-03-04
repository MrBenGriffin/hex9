import numpy as np
from modern.octahedron import Octahedron
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
# flatten the points and display.
# This is better facilitated with EFFs, but it was useful also.


def rnd_pts_for_side(o, side, n):
    opt = Util.oct0_rnd(n)  # random points on face.0
    sign = o.signs[side]  # find orientation.
    spt = np.copysign(opt, sign)  # opt is corrected.
    if side[0] == 'N':  # Add a ball to the N/S face.
        bl = Util.ball_rnd(o.ov['N'])
    else:
        bl = Util.ball_rnd(o.ov['S'])
    return np.vstack([spt, bl])  # add to the points.


def side_to_plane(matrix, points):  # given a side rotate to Z
    return points @ (matrix.T @ Util.rz(-np.pi/3.))  # rotate and adjust.


if __name__ == '__main__':
    octa = Octahedron()
    for face in octa.matrices:
        srd = rnd_pts_for_side(octa, face, 50000)
        trx = side_to_plane(octa.matrices[face], srd)
        pts = Util.d3_2(trx)
        Display.show_pts_2d(pts, (-.9, .9), (-.9, .9), face)
