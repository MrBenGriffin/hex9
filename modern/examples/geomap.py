import numpy as np
from modern.ak import AK
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display
from modern.util import Util
from photo import Photo

# Depict (in mpl) Blue Marble as Octahedral Map
# This is a little slow, because it goes through every pixel when generating.
# It is NOT efficient, but does show how arbitrary pixel values can be converted
# To grid coordinates.
# Last run on March3/25 it looked ok.

if __name__ == '__main__':
    octa = H9Octahedron()
    ak = AK()
    ps = Photo()   # source image
    ps.load('../../preparatory/world.topo.bathy.200406.3x5400x2700.png', False)
    ps.set_latlon([-90., 90.], [-180., 180.])
    x = np.linspace(*octa.glx, num=1000)
    y = np.linspace(*octa.gly, num=1000)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack((xx.ravel(), yy.ravel()), axis=1)
    ipt, cpt = [], []
    for xy in pts:
        face = octa.xy_side(xy[0], xy[1])
        if face is not None:
            ipt.append(xy)
            th = octa.grid_th[face]
            ud = octa.side_ud[face]  # use this for theta
            o_th = th + octa.oct_th[ud]  # use for orientation.
            r_mat = octa.matrices[face]  # get the face rotation.
            xya = Util.d2_3(np.array([xy - octa.grid_xy[face]]), octa.i3)
            uv = xya @ (r_mat.T @ Util.rz(o_th)).T
            xyz = ak.os(uv)
            ll = Util.xyz_ll(xyz)
            la, lo = ll[0]
            bgr = ps.col(la, lo, True)
            c = tuple([bgr[i]/255. for i in [2, 1, 0]])
            cpt.append(c)
    Display.col_pts_2d(np.array(ipt), cpt, octa.glx, octa.gly)
