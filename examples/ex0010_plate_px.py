"""
Part of the H9 project
First example :Round trip of a Plate Carrée png.
Note that it converts pixel coordinates to (x,y) cartesian coordinates.
"""

from hhg9 import Registrar
from hhg9.domains import PlatePixel
from support import Photo, Display

if __name__ == '__main__':
    """
    Load a photo, convert to PlatePixel points, 
    show it, then convert back and save.
    """
    reg = Registrar()  # Manage Domains & Projections
    # Register the PlatePixel Domain
    p_plt = PlatePixel(reg)

    d = Display()  # simple support display class
    ps = Photo()   # simple support photo class
    ps.load('../preparatory/world1350x675.png')  # RGB image

    plate = p_plt.adopt(ps.img)  # ps.img shape is [675,1350,4]
    d.show_pts_2d(plate)         # plate shape is [675*1350,6] (the final columns being the x,y)
    img = p_plt.image(plate)     # convert points back to an [675,1350,4] image.
    ps.img = img
    ps.save('px_map')

