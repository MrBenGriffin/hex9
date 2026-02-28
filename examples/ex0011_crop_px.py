# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Here we load a photo of the world, and then extract a crop of it, set by a digital lat/lon extent.
28 February 2026 0.1.1a1 (passed)


"""
import numpy as np
from hhg9 import Registrar
from hhg9.domains import PlatePixelCarree
from PIL import Image  # Pillow for clean image saving


def run():
    """
    Load a photo, convert to PlatePixel points, 
    crop to extent and save
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')
    base = np.array(Image.open("src/bm3600x1800.jpg").convert("RGBA"))
    height, width, channels = base.shape
    fs = PlatePixelCarree.full_sphere(reg, width, height)
    zone = [-11, 49.5, 2.5, 61]  # Rough UK boundary box
    # crop can be split - e.g, an area which crosses the dateline. Here it's not.
    arr, ext = fs.crop(base, zone)[0]
    plate = p_pix.adopt(arr, extent=ext)
    flat = p_pix.image(plate)     # convert points back to an image.
    img = Image.fromarray(flat, mode="RGBA")
    img.save("output/ex0011_UK.png")  # PNG keeps alpha


if __name__ == '__main__':
    run()
