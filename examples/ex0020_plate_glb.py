# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Supporting System Test - No octahedral projection or H9
This follows ex0010_plate_px.py (which loaded a png and displayed it).
This loads a Plate Carrée png, converts it to latitude/longitude, and displays it.
Needless to say, in this case, we could just use Basemap - but the point is to demonstrate
Last Tested 25 November 2025 √
"""
from matplotlib import image, pyplot as plt
from mpl_toolkits.basemap import Basemap
from hhg9 import Registrar, Points


def show_global(pts: Points, proj='ortho', alpha=1.0):
    """Display GCD points on the globe"""
    lat, lon = pts.coords[:, 0], pts.coords[:, 1]
    cols = pts.samples
    """Project GCD points onto global space."""
    fig = plt.figure(figsize=(12, 12), dpi=150, frameon=False)
    m = Basemap(projection=proj, lon_0=22.5, lat_0=40)
    m.fillcontinents(color='coral')
    xpt, ypt = m(lon, lat)
    m.scatter(xpt, ypt, c=cols, s=3, alpha=alpha)
    plt.show()


def run():
    """
    Load a photo, adopt into PlatePixel points,
    transform to Sphere via GCD (latitude/longitude).
    Then display onto globe
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')  # PlatePixel: Plate Carrée
    img = image.imread(f'src/tissot_2560x1280.png', 'png')
    pc_px = p_pix.adopt(img)     # ps.img shape is [1280,2560,3]
    sp_ll = reg.project(pc_px, ['p_pix', 'g_gcd'])  # project image as GCD
    show_global(sp_ll)  # Display it as an overlay on the globe.
    sp_pl = reg.project(sp_ll, ['g_gcd', 'p_pix'])  # Roundtrip back to Pixels
    img = p_pix.image(sp_pl)     # convert points back to an [1280,2560,3] image.

    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    run()
