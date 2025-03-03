import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from modern.display import Display

# Use a flat F triangle
# Convert to side 0 of Octahedron
# Demonstrate it's legal.
# Convert it back to F triangle.


def make_3df(o, ud, n):
    p2f = Util.tri_eff(n)
    if ud == 'V':
        return Util.d2_3(p2f, o.i3)
    else:
        return Util.d2_3(p2f, o.i3)  @ Util.rz(np.pi)


def plane_to_side(o, side, pts):  # given a side rotate to Z
    ud = o.side_ud[side]
    o_th = o.oct_th[ud]           # the transpose of a rotation is it's inverse.
    r_mat = o.matrices[side]
    return pts @ (r_mat.T @ Util.rz(o_th)).T  # rotate and adjust.


def side_to_plane(o, side, pts):  # given a side rotate to Z
    ud = o.side_ud[side]  # use this for theta
    o_th = o.oct_th[ud]   # use for orientation.
    r_mat = o.matrices[side]  # get the face rotation.
    return Util.d3_2(pts @ (r_mat.T @ Util.rz(o_th)))  # rotate and adjust.


if __name__ == '__main__':
    octa = H9Octahedron()
    f_u = make_3df(octa, 'Λ', 25000)  #
    f_d = make_3df(octa, 'V', 25000)  #
    nwa = plane_to_side(octa, 'NWA', f_u)
    nea = plane_to_side(octa, 'NEA', f_d)
    nwp = plane_to_side(octa, 'NWP', f_d)
    nep = plane_to_side(octa, 'NEP', f_u)
    sea = plane_to_side(octa, 'SEA', f_u)
    swa = plane_to_side(octa, 'SWA', f_d)
    swp = plane_to_side(octa, 'SWP', f_u)
    sep = plane_to_side(octa, 'SEP', f_d)
    pts = np.vstack([nwa, nea, nwp, nep, swa, sea, swp, sep])
    # Display.show_pts_3d(pts,
    #                     (-1.1, 1.1),
    #                     (-1.1, 1.1),
    #                     (-1.1, 1.1), 'NS')
    nwa2 = side_to_plane(octa, 'NWA', nwa)
    Display.show_pts_2d(nwa2, (-.9, .9), (-.9, .9), 'NWA')
    nea2 = side_to_plane(octa, 'NEA', nea)
    Display.show_pts_2d(nea2, (-.9, .9), (-.9, .9), 'NEA')
