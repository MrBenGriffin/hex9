import numpy as np
from modern.octahedron_h9 import H9Octahedron
from modern.display import Display

# Depict Octahedron Side Colours from xy grid test.
# This uses an x,y test across the entire rectilinear grid
# And then colours anything that is found to be lying within the grid.
# It is NOT efficient, but does show how arbitrary pixel values can be converted
# To grid coordinates.

if __name__ == '__main__':
    octa = H9Octahedron()
    n = 10000
    x = np.linspace(*octa.glx, num=1000)
    y = np.linspace(*octa.gly, num=1000)
    xx, yy = np.meshgrid(x, y)
    cols = Display.colours(9, 'tab10')
    ci = {face: cols[i] for i, face in enumerate(octa.matrices.keys())}
    pts = np.stack((xx.ravel(), yy.ravel()), axis=1)
    cx = []
    for x, y in pts:
        s = octa.xy_side(x, y)
        cx.append('k' if s is None else ci[s])
    Display.col_pts_2d(pts, cx, octa.glx, octa.gly)
