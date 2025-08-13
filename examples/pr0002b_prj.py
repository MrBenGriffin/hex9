"""
Part of the H9 project - Preparation 0002
(1) Load the Population numpy data file,
(2) Project it onto the Barycentric Octahedron.
"""
import numpy as np

from hhg9 import Registrar, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid

if __name__ == '__main__':
    # Kyushu
    min_lat_lon = np.array([30.973930071123313, 129.37257774562536])
    max_lat_lon = np.array([34.867227645401115, 132.06213814072615])

    file = 'jpn'
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    eg = EllipsoidGCD(reg)              # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)     # c_sph <=> (c_oct <=> b_oct)
    ak.set_accuracy(0.01)  # nanometre

    # Project the source gcd points onto octahedral.
    np_file = f'src/{file}_lon_lat_pop.npy'
    print(f'Loading GCD/Population numpy data file {np_file}')
    g_data = np.load(np_file)
    lon_lat = g_data[:, :2]
    size = g_data.shape[0]
    print(f'Found {size} rows. Converting to lat/lon arrays for projection.')
    lat_lon = lon_lat[:, [1, 0]]  # switch long/lat.
    mask = (
            (lat_lon[:, 0] >= min_lat_lon[0]) &
            (lat_lon[:, 0] <= max_lat_lon[0]) &
            (lat_lon[:, 1] >= min_lat_lon[1]) &
            (lat_lon[:, 1] <= max_lat_lon[1])
    )
    lat_lon = lat_lon[mask]
    pop = g_data[:, 2][mask]
    pop_file = f'src/{file}_pop_data.npy'
    np.save(pop_file, pop)

    size = lat_lon.shape[0]
    print(f'Restricted to {size} rows. Calculating boundaries')

    bd_file = f'src/{file}_lat_lon_bounds.npy'
    print(f'Calculating the GCD boundary and saving it into {bd_file}')
    min_lat_lon, max_lat_lon = lat_lon.min(axis=0), lat_lon.max(axis=0)
    # This is very tight on the edges, so we want to extend them.
    lat_span = max_lat_lon[0] - min_lat_lon[0]  # lat
    lon_span = max_lat_lon[1] - min_lat_lon[1]  # lon
    max_span = max(lat_span, lon_span)  # lat_lon
    padding = max_span * 0.025  # pad 2.5% of maximum span to each edge.
    min_lat_lon -= padding
    max_lat_lon += padding
    min_max = np.vstack((min_lat_lon, max_lat_lon))
    bounds = min_max.T
    extent = bounds.reshape(-1)
    # GCD bounds: min_lat/max_lat/min_lon/max_lon
    np.save(bd_file, extent)
    br_file = f'src/{file}_bounds_bry.npy'
    brc_file = f'src/{file}_bounds_bry_cmp.npy'
    print(f'Adopting GCD boundary, projecting to barycentric, and storing in {br_file} and {brc_file}')
    mm_gcd = Points(min_max, g_gcd)
    mm_data = reg.project(mm_gcd, [g_gcd, c_ell, c_oct, b_oct])
    mm_bry = mm_data.coords.T.reshape(-1)
    mm_bry_c = mm_data.components
    np.save(br_file, mm_bry)     # stores min-x/max-x, min-y/max-y
    np.save(brc_file, mm_bry_c)  # stores lower, upper point components.

