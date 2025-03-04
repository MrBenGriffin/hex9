import numpy as np
from modern.octahedron import Octahedron
from modern.util import Util
from modern.display import Display

# Depict an Octahedron made of 'F'
# Having got the flat F, we can now project it
# into a 3D setup and show that the 2D<=>3D
# conversions are ok.
# Importantly, the Fs should all be correctly oriented
# either upright (in the north) or upside-down (in the south).
# Util: d2_3(), tri_eff(), rz()
# Octahedron: i3, matrices[]
# Display: show_pts_3d()


def side_f(matrix, z, _n):  # octahedral-F for side.
    rotate = -np.pi/3.  # -120; because(?)
    spt = Util.d2_3(Util.tri_eff(_n), z)
    return spt @ (matrix.T @ Util.rz(rotate)).T  # rotate and adjust.


if __name__ == '__main__':
    lm = (-1.1, 1.1)
    octa = Octahedron()
    n = 10000
    s = {face: side_f(octa.matrices[face], octa.i3, n) for face in octa.matrices}
    pts = np.vstack(list(s.values()))
    Display.show_pts_3d(pts, lm, lm, lm)
