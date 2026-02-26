# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Note that it converts pixel coordinates to (x,y) net coordinates.
23 February 2026 0.1.0a4

"""
import numpy as np
from hhg9 import Registrar
from PIL import Image  # Pillow for clean image saving
Image.MAX_IMAGE_PIXELS = 200_000_000  # just above 128M


def run():
    """
    Load a photo, convert to PlatePixel points, 
    show it, then convert back and save.
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    n_oct = reg.domain('n_oct:diamonds')
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    b_oct.set_warp('src/l4_polished.npz')

    # n_oct:diamonds; cmp:[1 1 1]; bbox T:1.6385598199239102, L:3.127344981023394, B:1.4391513389336896, R:3.384795720163348
    t, l, b, r = [1.6385598199239102, 3.12734498102339, 1.4391513389336896, 3.384795720163348]
    img = np.array(Image.open("output/ex0252_sn_x2_L08_Reg_10.png").convert("RGBA"))
    plate = p_pix.adopt(img, extent=(l, b, r, t), y_up=True)
    n_oct.binning(plate)
    plate.domain = n_oct
    gplt = reg.project(plate, [n_oct, b_oct, g_gcd])
    # Origin = (14.023357627241339,33.797108194063014)
    # Pixel Size = (0.001916879003323,-0.001916879003323)
    # Corner Coordinates:
    # Upper Left  (  14.0233576,  33.7971082) ( 14d 1'24.09"E, 33d47'49.59"N)
    # Lower Left  (  14.0233576,  19.3630093) ( 14d 1'24.09"E, 19d21'46.83"N)
    # Upper Right (  31.0337419,  33.7971082) ( 31d 2' 1.47"E, 33d47'49.59"N)
    # Lower Right (  31.0337419,  19.3630093) ( 31d 2' 1.47"E, 19d21'46.83"N)
    # Center      (  22.5285498,  26.5800587) ( 22d31'42.78"E, 26d34'48.21"N)
    # PlatePixel expects extent as (xmin, ymin, xmax, ymax) = (lon_min, lat_min, lon_max, lat_max)
    # extent = (14.023357627241339, 19.363009299040208, 31.033741902730366, 33.797108194063014)
    # optional padding (in degrees)
    extent = (14.0233576, 19.3630093, 31.0337419, 33.7971082)  # (xmin,ymin,xmax,ymax) == (lon,lat)

    # Match the original WGS84 raster grid from gdalinfo (Size 8874x7530).
    # Note: GDAL reports pixel_size Y negative because rows go downward; in PlatePixel we keep
    # a positive `pixel_size` and use `y_up=True` to express north-up.
    p_pix.set_grid(width=8874*0.9, height=7530*0.9, extent=extent, y_up=True)
    pix = reg.project(gplt, ['g_gcd', 'p_pix'])  # Roundtrip back to Pixels
    flat = p_pix.image(pix)     # convert points back to an image.
    img = Image.fromarray(flat, mode="RGBA")
    img.save("output/ex0011_08.png")  # PNG keeps alpha


if __name__ == '__main__':
    run()
