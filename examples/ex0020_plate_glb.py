# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Supporting System Test - No octahedral projection or H9
This follows ex0010_plate_px.py (which loaded a png and displayed it).
This loads a Plate Carrée png, converts it to latitude/longitude, and displays it.
Needless to say, in this case, we could just use Basemap - but the point is to demonstrate
13 Mar 2026 0.1.1a1 (passed)
28 Feb 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
from matplotlib import image, pyplot as plt
from mpl_toolkits.basemap import Basemap
from hhg9 import Registrar, Points
from PIL import Image  # Pillow for clean image saving
import numpy as np


def show_global(pts: Points, proj='ortho', alpha=1.0):
    """Display GCD points on the globe"""
    lat, lon = pts.coords[:, 0], pts.coords[:, 1]
    cols = pts.samples
    cols = cols.astype(np.float64) / 255 # matplotlib wants RGBA within 0..1
    """Project GCD points onto global space."""
    fig = plt.figure(figsize=(12, 12), dpi=150, frameon=False)
    m = Basemap(projection=proj, lon_0=22.5, lat_0=40)
    m.scatter(x=lon, y=lat, latlon=True, c=cols, s=2, alpha=alpha)
    fig.savefig(f"output/ex0020_global.png", dpi=100)
    print(f'fig saved at output/ex0020_global.png')


def run():
    """
    Load a photo, adopt into PlatePixel points,
    transform to Sphere via GCD (latitude/longitude).
    Then display onto globe
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')  # PlatePixel (configured as Plate Carrée here)

    # Load as uint8 RGBA (matplotlib.imread often yields float32 0..1, which breaks roundtrips/saving)
    pil_img = Image.open('src/tissot_3600x1800.png').convert('RGBA')
    img = np.array(pil_img)
    print('img loaded', img.shape, img.dtype)

    # Adopt with an explicit full-sphere Plate Carrée extent so lon/lat mapping is correct.
    # extent ordering is (lon_min, lat_min, lon_max, lat_max) == (xmin, ymin, xmax, ymax)
    pc_extent = (-180.0, -90.0, 180.0, 90.0)
    pc_px = p_pix.adopt(img, extent=pc_extent, y_up=True, center=True)

    print(f'img adopted to Points')
    sp_ll = reg.project(pc_px, ['p_pix', 'g_gcd'])  # project image as GCD
    print(f'img projected to g_gcd')
    show_global(sp_ll)  # Display it as an overlay on the globe.
    sp_pl = reg.project(sp_ll, ['g_gcd', 'p_pix'])  # Roundtrip back to Pixels
    out = p_pix.image(sp_pl)

    # Ensure uint8 for saving.
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    mode = 'RGBA' if (out.ndim == 3 and out.shape[2] == 4) else 'RGB'
    pic = Image.fromarray(out, mode=mode)
    pic.save('output/ex0020_flat.png')  # PNG keeps alpha if present


if __name__ == '__main__':
    run()
