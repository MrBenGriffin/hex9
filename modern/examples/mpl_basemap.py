import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from examples.addressing import Registrar, SphericalGCD, OctahedralCartesian, AKOctahedralSpherical, SphericalCartesian
from modern.octahedron import Octahedron
from modern.util import Util

if __name__ == '__main__':
    o = Octahedron()
    u = Util()
    reg = Registrar()
    # Coordinate Systems
    gcd = SphericalGCD(reg)
    crt = SphericalCartesian(reg)
    hed = OctahedralCartesian(reg)
    # Projections
    ak = AKOctahedralSpherical(reg)
    cs = SphericalCartesian(reg)
    reg.register_projection('chain', [gcd, crt, hed])  # should this be [ak, cs] ?

    eff = u.tri_eff(10000)
    effs = {f: octant.adopt(eff) for f, octant in o.faces.items()}
    pts = np.vstack(list(effs.values()))
    loc = reg.project(pts, [hed, gcd])
    fig = plt.figure(figsize=(12, 12), dpi=150, frameon=False)
    m = Basemap(projection='ortho', lon_0=-105, lat_0=40, resolution='l')
    m.fillcontinents(color='coral', lake_color='aqua')
    xpt, ypt = m(loc[..., 1], loc[..., 0])
    m.scatter(xpt, ypt)
    plt.show()
