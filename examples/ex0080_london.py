"""
Part of the H9 project
Compose an image of central London with a grid overlay.
"""
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib import image, pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import KDTree
from hhg9 import Registrar, H9Engine, Step, Grid
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric, PlatePixel
from hhg9.projections import EllipsoidGCD, PlatePixelGCD, AKOctahedralEllipsoid
from hhg9.formats import OctahedralH9, DMS, DecimalDegrees, DecimalCartesian
from support import Util, Display


def reverse_address(addr: str):  # -> List[Tuple[int, str, int]]:
    """
    Parse hierarchical hex address into list of (subhex, mode, rotation) tuples.
    Final segment is something like Λ1 or V2.
    """
    path = []
    i = 0
    while i < len(addr) - 2:
        path.append(int(addr[i]))
        i += 1
    final_mode = addr[-2]
    final_rot = int(addr[-1])
    # Generate full sequence with implied modes/rotations
    descent = []
    mode = final_mode
    rot = final_rot
    for digit in reversed(path):
        descent.append((digit, mode, rot))
        if digit < 6:
            # Simple: flip Λ/V, flip rotation
            mode = 'Λ' if mode == 'V' else 'V'
            # rot = ROT_FLIP[rot]
        else:
            # Special handling below
            pass
    return descent


if __name__ == '__main__':
    reg = Registrar()  # Manage Domains & Projections
    g_gen = GeneralGCD(reg)  # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)  # Cartesian Geodesic (xyz)
    c_oct = OctahedralCartesian(reg)  # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    p_plt = PlatePixel(reg)  # 2D Pixel Cartesian Domain

    h9 = OctahedralH9()  # formatter.
    g_gen.register_format(DMS())
    g_gen.register_format(DecimalDegrees())
    c_ell.register_format(DecimalCartesian())
    b_oct.register_format(h9)

    # Projections/Transforms. Bary and Net are loaded by the domains.
    EllipsoidGCD(reg)  # [g_gen, c_ell]
    ak = AKOctahedralEllipsoid(reg)  # [c_ell, c_oct]
    ppg = PlatePixelGCD(reg)

    # Support Classes
    u = Util()
    h9 = H9Engine()
    g = Grid()
    d = Display()
    imagery = OSM()

    acc = ak.set_accuracy(0.000000000000000001)
    spot = np.array([
        [51.50676864763299, -0.1330302972995536],       # Traveller's Club
        # [51.508777843989016, 0.02452302507469842],   # Passport Office
        # [51.477270526325135,  0.00000000000000001],  # greenwich park east
        # [51.477270526325135, -0.00000000000000001],  # greenwich park west
        # [+0.0000000000000001, 22.1233123456564561],  # equator east 22.12
        # [-0.0000000000000001, 22.1233123456564561]   # equator west 22.12
    ])
    ll0 = g_gen.adopt(spot)
    sp0 = reg.project(ll0, [g_gen, c_ell])  # spherical cart
    oc0 = reg.project(sp0, [c_ell, c_oct])
    bc0 = reg.project(oc0, [c_oct, b_oct])

    club = bc0.coords[0]  # [0.30298694 0.28958754]
    # pasp = bc0.coords[1]  # [0.30298694 0.28958754]
    cmp = tuple(bc0.components[0])
    sdo = bc0.domain.components[cmp]
    loc = sdo.tr

    plx = h9.enmesh(club, loc, acc, False, True)  # This is the set of hexagons.
    # Some of the hex points will be out of scope for NWA. Here we will merely truncate them.
    # Let's just use hexagon at layer 6 for our start.
    # for i in range(2, 32):
    # e = 0.0000001
    # for k in range(9):
    #     print(f'\n{bc0:h9.32}\n{bc0:h9.c32}')
    #     # bc0.coords[0][0] += e
    #     # bc0.coords[1][0] += e
    #     bc0.coords[0][1] -= e
    #     bc0.coords[1][1] += e
    # cod6 = f'{bc0:h9.6}'
    # cod10 = f'{bc0:h9.10}'
    # codcc = f'{bc0:h9.c}'
    i = 8
    club8 = f'{bc0:h9.8}'
    # 0v2
    # dx = b_oct.decode(code)
    rt_hex = plx[i]

    hxp = sdo.adopt(rt_hex)
    hx = reg.project(hxp, [b_oct, c_oct, c_ell, g_gen])
    y0, y1 = np.min(hx.coords[..., 0]), np.max(hx.coords[..., 0])
    x0, x1 = np.min(hx.coords[..., 1]), np.max(hx.coords[..., 1])
    extent = [x0, x1, y0, y1]
    w = x1 - x0
    h = y1 - y0
    wid = 5000  # width of sample for hexagon
    hgt = int(wid * (3 ** 0.5 / 2.))  # height wid*0.8660254

    pc_file = Path(f'sh/LDN_{i}_pc.png')
    if not pc_file.exists():
        scale = 17
        lon_min, lon_max, lat_min, lat_max = extent
        fw, fh, dpi = wid / 72, hgt / 72, 72
        fig = plt.figure(figsize=(fw, fh), dpi=dpi, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
        # PlateCarree.
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_image(imagery, scale)
        fig.savefig(pc_file, format='png', bbox_inches='tight', pad_inches=0)
    # Load and convert the LDN PlateCarree to a sample source in GCD coordinates.
    img = image.imread(pc_file, 'png')
    pc_px = p_plt.adopt(img)
    ppg.set_dim(pc_px, extent)
    pc_sp = reg.project(pc_px, [p_plt, g_gen])
    # Create a pixel-grid from layer root_hexagon.
    hh1 = rt_hex[:4]  # First 4 is a half-hex.
    hh2 = np.vstack([rt_hex[3:], rt_hex[:1]])
    map_size = 4000
    x_lim = np.min(rt_hex[:, 0]), np.max(rt_hex[:, 0])
    y_lim = np.min(rt_hex[:, 1]), np.max(rt_hex[:, 1])
    i_wid = map_size  # width of hexagon is 2A
    i_hgt = int(i_wid * (3 ** 0.5 / 2.))  # height wid*0.8660254

    w1, h1, pxl1, msk1 = g.qa_grid(hh1, map_size)
    w2, h2, pxl2, msk2 = g.qa_grid(hh2, map_size)
    pxl = np.unique(np.vstack([pxl1, pxl2]), axis=0)
    pts = sdo.adopt(pxl)
    src = KDTree(pc_sp.coords)  # KDTree of plate_carrée projected onto unit sphere.
    sp1 = reg.project(pts.copy(), [b_oct, c_oct, c_ell, g_gen])
    _, idx = src.query(sp1.coords, workers=-1)  # query KDTree and return indices of pc_sp
    pts.samples = pc_sp.samples[idx]
    img = pts.image((i_wid, i_hgt))
    fw, fh, dpi = i_wid / 72, i_hgt / 72, 72
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.imshow(img, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
    # So that's the base layer.
    ldn_x, ldn_y = bc0.coords[0]  # [0.30301429 0.28962174]
    step = Step(loc, ldn_x, ldn_y)
    for j in range(5):
        step, x = h9.encode_step(step)
    steps = h9.branch_step(step)
    for layer in range(3):
        nxt = []
        # if layer == 3:
        #     l_xy = np.array([(s.x, s.y) for s in steps])
        #     labs = sdo.adopt(l_xy)
        pols = []
        for st in steps:
            subs = h9.branch_step(st)
            for sub in subs:
                hp = h9.poly_step(sub, True)
                if np.any(Grid.in_convex_poly(hp, rt_hex)):
                    pols.append(hp)
            nxt.extend(subs)
        if len(pols) > 0:
            # print(f'layer {layer}: {len(pols)} Hexagons')
            pc = PolyCollection(pols, alpha=0.24, fc='none', edgecolor='k', linewidth=2.5)
            ax.add_collection(pc)
        steps = nxt
    plt.show()
