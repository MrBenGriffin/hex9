"""
Part of the H9 project - Preparation 0004
Project the imagemap onto barycentric via sampling.
Last Tested 07 August 2025 √
"""
import os
import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Grid, Points
from hhg9.domains import PlatePixel, GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import PlatePixelGCD, EllipsoidGCD, AKOctahedralEllipsoid
from support import Util
from PIL import Image  # Pillow for clean image saving


if __name__ == '__main__':
    util = Util()
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    p_plt = PlatePixel(reg)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.

    # Projections/Transforms. Bary and Net are loaded by the domains.
    eg = EllipsoidGCD(reg)            # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)   # c_sph <=> (c_oct <=> b_oct)
    pp = PlatePixelGCD(reg)
    layers = ak.set_accuracy(0.0000000001)  # nanometre
    grid = Grid()

    file = 'jpn'
    bd_file = f'src/{file}_lat_lon_bounds.npy'
    lat_lon = np.load(bd_file)  # GCD bounds: min_lat/max_lat/min_lon/max_lon
    min_lat, max_lat, min_lon, max_lon = lat_lon

    b_rect = np.load(f'src/{file}_bry_border.npy')
    b_rcmp = np.load(f'src/{file}_bounds_bry_cmp.npy')
    cmp = tuple(b_rcmp[0])
    sdo = b_oct.components[cmp]
    theta = np.load(f'src/{file}_theta.npy')
    centroid = np.load(f'src/{file}_centroid.npy')
    grid_shape = np.load(f'src/{file}_rot_bry_border.npy')
    img_w = 6000
    ww, hh, pxl, msk, (grid_org, grid_scale) = grid.qa_grid(grid_shape, img_w)
    np.save(f'src/{file}_bg_extent.npy', grid_org)
    data = pxl[msk]  # These are our reference points.
    c_th, s_th = np.cos(theta), np.sin(theta)
    rot = np.array([[c_th, -s_th], [s_th, c_th]])
    bary = (data-centroid) @ rot.T + centroid
    refs = Points(bary, sdo, components=np.array(cmp))
    sp1 = reg.project(refs, [b_oct, c_oct, c_ell, g_gcd])
    img_file = f'src/{file}_gcd.png'
    if os.path.isfile(img_file):  # must be plate carree, ideally with an alpha channel.
        img = image.imread(img_file, 'png')
    else:
        raise FileNotFoundError(f'{img_file} not found.')
    pc_px = p_plt.adopt(img)  # min_lat, max_lat, min_lon, max_lon

    pp.set_dim(pc_px, (min_lon, max_lon, min_lat, max_lat))  # (lon_min, lon_max, lat_min, lat_max)
    pc_sp = reg.project(pc_px, [p_plt, g_gcd])
    src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.
    _, idx = src.query(sp1.coords, workers=-1)  # query KDTree and return indices of pc_sp
    samples = pc_sp.samples[idx]  # Grab the colours at each point.
    if samples.shape[1] == 3:
        rgba = np.hstack((samples, np.ones((samples.shape[0], 1), dtype=samples.dtype)))
    else:
        rgba = samples
    pv = np.round((pxl - grid_org[:2]) * grid_scale).astype(int)
    pv[:, 1] = hh - pv[:, 1]  # convert Cartesian-up y to raster-down y
    image = np.zeros((hh + 1, ww + 1, 4))
    image[pv[:, 1][msk], pv[:, 0][msk]] = rgba
    image_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(image_uint8)
    pil_img.save(f'src/{file}_grid.png')

