# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Supporting System Test - No octahedral projection or H9
Roundtrip of a Plate Carrée <=> PlatePixel
Note that it converts pixel coordinates to (x,y) cartesian coordinates.
26 February 2026 0.1.1a1 (passed)
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed)
25 November 2025 (passed)
"""
from matplotlib import image, pyplot as plt
from hhg9 import Registrar

def run():
    """
    Load a photo, convert to PlatePixel points, 
    show it, then convert back and save.
    """
    reg = Registrar()  # Manage Domains & Projections
    p_pix = reg.domain('p_pix')

    img = image.imread(f'src/bm_3600x1800.jpg', 'jpg')
    plate = p_pix.adopt(img)     # ps.img shape is [675,1350,4]
    img = p_pix.image(plate)     # convert points back to an [675,1350,4] image.

    # Use matplotlib to display.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(img)
    fig.savefig(f"output/ex0010.png", dpi=400)
    print(f'fig saved at output/ex0010.png')


if __name__ == '__main__':
    run()
