import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
import math


class Plotter:
    def __init__(self, limits: tuple, screen_dim: tuple = None):
        self.w, self.h = limits
        if screen_dim:
            self.fig_w, self.fig_h = screen_dim
        else:
            w, h = self.w[1] - self.w[0], self.h[1] - self.h[0]
            self.fig_w, self.fig_h = math.ceil(w), math.ceil(h)
        self.fig, self.ax = plt.subplots(figsize=(self.fig_w, self.fig_h), dpi=100, layout='tight', frameon=False)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.w)
        self.ax.set_ylim(self.h)
        self.ax.set_axis_off()
        self.axs = plt.gca()
        self.axs.invert_yaxis()
        plt.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    def save(self, filename: str):
        self.fig.savefig(f'output/{filename}.png', bbox_inches='tight', pad_inches=0)


class HexPlot(Plotter):
    def __init__(self, limits: tuple):
        self.xm = np.sqrt(3.) * 0.5  # x multiplier for cartesian x
        self.r = np.sqrt(3.) / 3.  # radius for hexagon in matplotlib for hex 1.0 0.568
        w, h = limits
        plot_lims = [-self.r, w], [-1, h]
        screen_dim = math.ceil(w * 0.75), math.ceil((h-1) * 0.75)
        super().__init__(plot_lims, screen_dim)
        self.r30 = np.radians(30.)

    def hex(self, where, c: tuple):
        r, g, b = c
        px, py = where
        col = f'#{int(r):02X}{int(g):02X}{int(b):02X}'
        xp = px * self.xm
        yp = py if px & 1 == 0 else py - 0.5
        hx = RegularPolygon((xp, yp), numVertices=6, radius=self.r,
                            linewidth=0.5, orientation=self.r30,
                            facecolor=col, alpha=None, edgecolor=col, aa=True)
        self.ax.add_patch(hx)
