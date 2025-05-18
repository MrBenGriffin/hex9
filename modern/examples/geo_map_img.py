import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import numpy as np
from modern.util import Util
from photo import Photo
from examples.addressing import OctahedralCartesian, Registrar, SphericalGCD, SphericalCartesian, OctahedralBarycentric, OctahedralNet, \
    OctahedralSpherical, AKOctahedralSpherical

# Save Blue Marble as Octahedral Map PNG
# 5400x2700 => 4725x3507:  315º by 233.826859º (√3/2*3*90)
# 86400x43200 =>: 75600x56118:  315º by 233.826859º (√3/2*3*90)
# There are two versions of this - 'huge'  takes days.
# The smaller version takes about ten minutes.
# 25-05-18 √ rgba x

huge = False

if __name__ == '__main__':
    reg = Registrar()  # Manage coordinate sets & projections
    # Coordinate sets
    gcd = SphericalGCD(reg)  # 3D Spherical (2-value Polar)
    esp = SphericalCartesian(reg)  # 3D Spherical (3-value Euclidean)
    hed = OctahedralCartesian(reg)  # 3D Octahedral (surface) coordinate set
    bry = OctahedralBarycentric(reg, hed)  # 2d Flat coordinate set.
    net = OctahedralNet(reg, hed)  # 2d Flat coordinate set.
    # Projections
    es = OctahedralSpherical(reg)    # crt<=>gcd 3D to Polar
    ak = AKOctahedralSpherical(reg)  # crt<=>hed 3D
    reg.register_projection('chain', [gcd, esp, hed, bry, net])  # latlon to octahedral via spherical_euclidean

    octa = hed
    u = Util()
    ps = Photo()   # source image
    if huge:
        ps.load('w86400x43200.png', False)
        dw, dh = 75600, 56118  # 315º by 233.826859º
    else:
        ps.load('h6930x3465.png', False)
        ps.convert_a(True)
        # ps.load('w5400x2700.png', False)
        dw, dh = 4725, 3507  # 315º by 233.826859º
    ps.set_latlon([-90., 90.], [-180., 180.])

    pd = Photo()
    dws = net.glx[1] / dw
    dhs = net.gly[1] / dh
    sc = [dw / net.glx[1], dh / net.gly[1]]
    pd.new(dw, dh)  # photo-pixels.
    pd.convert_a(False)
    x = np.linspace(*net.glx, num=dw)
    y = np.linspace(*net.gly, num=dh)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack((xx.ravel(), yy.ravel()), axis=1)
    bob = net.binning(pts)
    ipt = np.vstack([(v * sc).astype(int) for v in bob.values()])
    gcd_pts = reg.project(bob, [net, gcd])
    for (px, py), (la, lo) in zip(ipt, gcd_pts):
        pd.img[dh - py - 1, px] = ps.col(la, lo)
    pd.save(f'{dw}x{dh}gr')
