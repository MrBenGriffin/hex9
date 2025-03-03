import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display

# Fill a grid with points.

if __name__ == '__main__':
    octa = H9Octahedron()
    x = np.linspace(*octa.glx, num=1000)
    y = np.linspace(*octa.glx, num=1000)
    xx, yy = np.meshgrid(x, y)
    pts = np.stack((xx.ravel(), yy.ravel()), axis=1)
    Display.show_pts_2d(pts, octa.glx, octa.gly)
