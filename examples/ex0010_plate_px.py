"""
Part of the H9 project - Supporting System Test - No octahedral projection or H9
Roundtrip of a Plate Carrée <=> PlatePixel
Note that it converts pixel coordinates to (x,y) cartesian coordinates.
Last Tested 10 August 2025 √
"""
from matplotlib import image, pyplot as plt
from hhg9 import Registrar
from hhg9.domains import PlatePixel

if __name__ == '__main__':
    """
    Load a photo, convert to PlatePixel points, 
    show it, then convert back and save.
    """
    reg = Registrar()  # Manage Domains & Projections
    p_plt = PlatePixel(reg)

    img = image.imread(f'src/world1350x675.png', 'png')
    plate = p_plt.adopt(img)     # ps.img shape is [675,1350,4]
    img = p_plt.image(plate)     # convert points back to an [675,1350,4] image.

    # Use matplotlib to display.
    fig = plt.figure(figsize=(18, 9), dpi=100, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    plt.imshow(img)
    plt.show()

