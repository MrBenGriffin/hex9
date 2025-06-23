import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import numpy as np

from hhg9 import Points


class Display:
    """Show stuff"""

    @classmethod
    def get_colours(cls, samples, clip: bool=False):
        """
        Given a set of samples
        Return them as colours.
        """
        if samples is not None:
            cols = samples
            if clip:
                cols = np.clip(cols, 0, 1)
            else:
                _max = np.max(cols)
                if _max > 1.0:
                    top = 1.0 / _max
                    cols = np.array(cols) * top
        return cols

    @classmethod
    def colours(cls, n, cmap='tab20'):
        return mpl.colormaps[cmap](np.linspace(0, 1, n))

    @classmethod
    def show_pts_2d(cls, arr: Points, x_lim=None, y_lim=None, label=None, clip=False, extra=0.1, ratio=None):
        if isinstance(arr, Points):
            xx, yy = arr.coords[:, 0], arr.coords[:, 1]
            if arr.samples is not None:
                cols = cls.get_colours(arr.samples, clip)
        else:
            xx, yy = arr[:, 0], arr[:, 1]
            cols = None
        if ratio is not None:
            fig_x = 16
            fig_y = fig_x/ratio
            fig = plt.figure(figsize=(fig_x, fig_y), dpi=150, frameon=False)
        else:
            fig = plt.figure(figsize=(10, 10), dpi=150, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        if x_lim is None:
            # xmi, xma = xx.min(), xx.max()
            # extra = (xma-xmi) * extra
            x_lim = (xx.min()-extra, xx.max()+extra)
        if y_lim is None:
            y_lim = (yy.min()-extra, yy.max()+extra)
        if label is not None:
            ofx = (x_lim[1] - x_lim[0]) / 20
            ofy = (y_lim[1] - y_lim[0]) / 20
            ax.text(x_lim[0] + ofx, y_lim[1] - ofy, label, fontsize=20)
        if x_lim is not None:
            ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(xx, yy, marker='.', s=0.5, alpha=0.5, c=cols)
        if isinstance(arr, Points):
            if arr.samples is not None and arr.samples.ndim == 1:
                _, idx = np.unique(arr.samples, return_index=True)
                for i in idx:
                    ax.annotate(arr.samples[i], (arr.coords[i][0], arr.coords[i][1]))
        ax.set_aspect('equal', adjustable='box')
        a = plt.gca()
        plt.box(False)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        plt.show()

    @classmethod
    def col_pts_2d(cls, pts, cols, x_lim=None, y_lim=None, label=None):
        fig = plt.figure(figsize=(10, 10), dpi=150, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        if label is not None:
            ofx = (x_lim[1] - x_lim[0]) / 20
            ofy = (y_lim[1] - y_lim[0]) / 20
            ax.text(x_lim[0] + ofx, y_lim[0] + ofy, label, fontsize=40)
        if x_lim is not None:
            ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(pts[:, -2], pts[:, -1], c=cols, s=2.5)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    @classmethod
    def poly_2d(cls, collections, x_lim=None, y_lim=None, names=None, filename=None):
        fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        for c in collections:
            ax.add_collection(c)
        if x_lim is not None:
            ax.set(xlim=x_lim, ylim=y_lim)
        ax.set_aspect('equal', adjustable='box')
        if names is not None:
            ax.legend(names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if filename is not None:
            plt.savefig(f'{filename}')
        plt.show()

    @classmethod
    def poly_3d(cls, collections, x_lim=None, y_lim=None, z_lim=(-0.01, 0.01)):
        fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        ax.set_proj_type('ortho')  # FOV = 0 deg
        ax.view_init(90, -90, 0)  # √ x,y top down.
        for c in collections:
            ax.add_collection(c)
        ax.auto_scale_xyz(x_lim, y_lim, z_lim)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    @classmethod
    def show_pts_3d(cls, arr: Points, x_lim=None, y_lim=None, z_lim=None, label=None, clip=False):
        xx, yy, zz = arr.coords[:, 0], arr.coords[:, 1], arr.coords[:, 2]
        cols = None
        if arr.samples is not None:
            cols = cls.get_colours(arr.samples, clip)
        fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(-1, -1, 0., label, 'x', fontsize=25)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        ax.set_proj_type('ortho')  # FOV = 0 deg
        ax.auto_scale_xyz(x_lim, y_lim, z_lim)
        ax.scatter(xx, yy, zz, marker=',', s=0.5, c=cols)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    @classmethod
    def show_global(cls, pts: Points, proj='ortho', alpha=1.0):
        lat, lon = pts.coords[:, 0], pts.coords[:, 1]
        cols = None
        if pts.samples is not None:
            cols = cls.get_colours(pts.samples)
        """Project GCD points onto global space."""
        fig = plt.figure(figsize=(12, 12), dpi=150, frameon=False)
        m = Basemap(projection=proj, lon_0=22.5, lat_0=40, resolution='c')
        m.fillcontinents(color='coral')
        xpt, ypt = m(lon, lat)
        m.scatter(xpt, ypt, c=cols, s=3, alpha=alpha)
        plt.show()

