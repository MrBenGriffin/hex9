import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import numpy as np
from modern.ak import AK
from modern.octahedron_h9 import H9Octahedron
from modern.util import Util
from photo import Photo

# Save Blue Marble as Octahedral Map PNG
# 5400x2700 => 4725x3507:  315º by 233.826859º (√3/2*3*90)
# 86400x43200 =>: 75600x56118:  315º by 233.826859º (√3/2*3*90)

if __name__ == '__main__':
    octa = H9Octahedron()
    ak = AK()
    ps = Photo()   # source image
    ps.load('w86400x43200.png', False)
    ps.set_latlon([-90., 90.], [-180., 180.])
    pd = Photo()
    dw, dh = 75600, 56118  # 315º by 233.826859º (√3/2*3*90) for 15px / degree.
    dws = octa.glx[1] / dw
    dhs = octa.gly[1] / dh
    pd.new(dw, dh)  # photo-pixels.
    for wx in range(dw):  # This gives us the pt.
        for iy in range(dh):
            wy = dh - iy - 1
            ax = wx * dws
            ay = wy * dhs
            face = octa.xy_side(ax, ay)
            if face is not None:
                th = octa.grid_th[face]
                ud = octa.side_ud[face]  # use this for theta
                o_th = th + octa.oct_th[ud]  # use for orientation.
                r_mat = octa.matrices[face]  # get the face rotation.
                xy = np.array([ax, ay])
                xya = Util.d2_3(np.array([xy - octa.grid_xy[face]]), octa.i3)
                uv = xya @ (r_mat.T @ Util.rz(o_th)).T
                xyz = ak.os(uv)
                ll = Util.xyz_ll(xyz)
                la, lo = ll[0]
                pd.img[iy, wx] = ps.col(la, lo, True)
    pd.save(f'huge_map')
