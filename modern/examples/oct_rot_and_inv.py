import numpy as np
from modern.octahedron import Octahedron
from modern.util import Util
from modern.display import Display

# Use a flat F triangle
# Convert to side 0 of Octahedron
# Demonstrate it's legal.
# Convert it back to F triangle.
# This is handling 2d<=>3d conversions and ensuring that objects are correct.
# Not 100% why the rotation of -120º is needed here.
# Util: d2_3(), tri_eff(), rz()
# Octahedron: i3, matrices[]
# Display: show_pts_3d(), show_pts_2d()


def plane_to_side(matrix, points):  # octahedral-F for side.
    return points @ (matrix.T @ Util.rz(-np.pi/3.)).T  # rotate and adjust.


def side_to_plane(matrix, points):  # given a side rotate to Z
    return points @ (matrix.T @ Util.rz(-np.pi/3.))  # rotate and adjust.


if __name__ == '__main__':
    octa = Octahedron()
    eff = Util.d2_3(Util.tri_eff(15000), octa.i3)
    sides = {side: plane_to_side(octa.matrices[side], eff) for side in octa.matrices}
    pts = np.vstack(list(sides.values()))
    Display.show_pts_3d(pts, (-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1))
    nwa = side_to_plane(octa.matrices['NWA'], sides['NWA'])
    Display.show_pts_2d(nwa, (-.9, .9), (-.9, .9), 'NWA')
