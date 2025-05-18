"""
Part of the H9 project

here we find the x,y of the resultant map and
attempt to find out what it's latitude and longitude is
from which we take a sample.
"""
from hhg9 import (
    Registrar, Points, Projection, SphericalGCD,
    SphericalCartesian, OctahedralCartesian, OctahedralBarycentric,
    OctahedralNet, DMS, DecimalDegrees,
    AKOctahedralSpherical, CartesianGCD
)
from support import Util, Display, Photo


if __name__ == '__main__':
    u = Util()
    reg = Registrar()  # Manage coordinate sets & projections
    g_sph = SphericalGCD(reg)  # 3D Spherical (2-value Polar)
    c_sph = SphericalCartesian(reg)  # 3D Spherical (3-value Euclidean)
    c_oct = OctahedralCartesian(reg)  # 3D Octahedral (surface) coordinate set
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat coordinate set.
    n_oct = OctahedralNet(reg, c_oct, b_oct)  # 2d Flat coordinate set.

    map_w, map_h = n_oct.width, n_oct.height
    scale = 200.
    ps = Photo()   # source image
    ps.load('../preparatory/world.topo.bathy.200406.3x5400x2700.png')
    ps.set_latlon([-90., 90.], [-180., 180.])
    dw, dh = int(scale*map_w), int(scale*map_h)
    pd = Photo()  # destination image.
    pd.new(dw, dh)  # photo-pixels.

    for wx in range(dw):  # This gives us the pt.
        for iy in range(dh):
            wy = dh - iy - 1
            ax = wx / scale
            ay = wy / scale
            sk = _o.xy_side(ax, ay)  # This looks to be good.
            if sk is not None:
                geo = _o.side(sk)[0]
                px, py, pz = geo.offs
                mpt = np.array([ax-px, ay-py])
                pt = mpt @ geo.map_r2d.T
                p3 = np.append(pt, pz)
                # p3 = (np.array([ax, ay, 0]) - geo.offs) * [1., 1., -1.]
                po = p3 @ geo.matrix.T
                # oso = _o.xyz_side(po)
                # df = np.sum(np.abs(po))
                sph = _o.o_s(po)
                sso = _o.xyz_side(sph)

                ll = u.xyz_ll(np.array([sph]))
                la, lo = ll[0]
                c = ps.col(la, lo, False)
                pd.img[iy, wx] = c
    pd.convert()
    pd.save(f'direct_map')
