# Import necessary modules
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (36, 18)
    mpl.rcParams['axes.xmargin'] = 0.0
    mpl.rcParams['axes.ymargin'] = 0.0
    mpl.rcParams['figure.frameon'] = False
    mpl.rcParams['figure.constrained_layout.use'] = True
    # mpl.rcParams['figure.constrained_layout.use'] = True

    plt.axis('off')
    fig, ax = plt.subplots(dpi=150)
    ax.use_sticky_edges = True
    ax.set_aspect('equal')
    ax.set_xlim([-180., 180.])
    ax.set_ylim([-90., 90.])
    ax.set_axis_off()
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    fp = "assets/landmasses/land_polygons.shp"
    data = gpd.read_file(fp)
    data.plot(facecolor='green', ax=ax)
    plt.show()
