"""
Part of the H9 project - Supporting System Test - No octahedral projection or H9
This follows ex0010_plate_px.py (which loaded a png and displayed it).
This loads a Plate Carrée png, converts it to latitude/longitude, and displays it.
Needless to say, in this case, we could just use Basemap - but the point is to demonstrate
Last Tested 10 August 2025 √
"""
import numpy as np
from matplotlib import image, pyplot as plt
from hhg9 import Registrar
from hhg9.domains import PlatePixel, GeneralGCD
from hhg9.projections import PlatePixelGCD
from support import Display

if __name__ == '__main__':
    """
    Load a photo, adopt into PlatePixel points, transform to Sphere via GCD (latitude/longitude).
    Then display as a Unit Sphere
    """
    reg = Registrar()  # Manage Domains & Projections
    # Domains - 2D image and GCD Spherical.
    p_plt = PlatePixel(reg)      # 2D Pixel Cartesian Domain
    g_gcd = GeneralGCD(reg)    # GCD Domain (latitude/longitude)
    PlatePixelGCD(reg)   # Transform (Pixel Cartesian <=> GCD)
    d = Display()

    img = image.imread(f'src/tissot_2560x1280.png', 'png')
    pc_px = p_plt.adopt(img)     # ps.img shape is [1280,2560,3]
    # project it into GCD
    sp_ll = reg.project(pc_px, [p_plt, g_gcd])
    llx = np.array(sp_ll.coords)
    # test g_sph can adopt ll array.
    g_gcd.adopt(llx)
    # Display it as an overlay on the globe.
    d.show_global(sp_ll, alpha=0.15)
    # Roundtrip back to Pixels
    sp_pl = reg.project(sp_ll, [g_gcd, p_plt])
    img = p_plt.image(sp_pl)     # convert points back to an [1280,2560,3] image.
    # Use matplotlib to display 2D.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(img)
    plt.show()

