import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import numpy as np


class Display:
    """Show stuff"""

    @classmethod
    def colours(cls, n, cmap='tab20'):
        return mpl.colormaps[cmap](np.linspace(0, 1, n))

    @classmethod
    def show_pts_2d(cls, pts, x_lim=None, y_lim=None, label=None, clip=False):
        if isinstance(pts, tuple):
            pts = np.vstack(pts)
        pts = np.asarray(pts)
        cols = None
        if pts.shape[1] > 2:
            cols = np.array(pts[:, :-2])
            if clip:
                cols = np.clip(cols, 0, 1)
            else:
                if np.max(cols) > 1.0:
                    cols = cols / 256
        fig = plt.figure(figsize=(10, 10), dpi=150, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        if x_lim is None:
            x_lim = (pts[:, -2].min()-0.1, pts[:, -2].max()+0.1)
        if y_lim is None:
            y_lim = (pts[:, -1].min()-0.1, pts[:, -1].max()+0.1)
        if label is not None:
            ofx = (x_lim[1] - x_lim[0]) / 20
            ofy = (y_lim[1] - y_lim[0]) / 20
            ax.text(x_lim[0] + ofx, y_lim[1] - ofy, label, fontsize=20)
        if x_lim is not None:
            ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(pts[:, -2], pts[:, -1], s=1.5, c=cols)
        ax.set_aspect('equal', adjustable='box')
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
    def show_pts_3d(cls, pts, x_lim=None, y_lim=None, z_lim=None, label=None, clip=False):
        cols = None
        if pts.shape[1] > 3:
            cols = np.array(pts[:, :-3])
            if clip:
                cols = np.clip(cols, 0, 1)
            else:
                if np.max(cols) > 1.0:
                    cols /= 256.
        fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(-1, -1, 0., label, 'x', fontsize=25)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        ax.set_proj_type('ortho')  # FOV = 0 deg
        ax.auto_scale_xyz(x_lim, y_lim, z_lim)
        ax.scatter(pts[:, -3], pts[:, -2], pts[:, -1], marker='o', s=0.5, c=cols)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    @classmethod
    def show_global(cls, gcd_pts, proj='ortho', alpha=1.0):
        cols = None
        if gcd_pts.shape[1] > 2:
            cols = gcd_pts[:, :-2] / 255.0
        """Project GCD points onto global space."""
        fig = plt.figure(figsize=(18, 9), dpi=150, frameon=False)
        # m = Basemap()
        m = Basemap(projection=proj, lon_0=22.5, lat_0=40, resolution='c')
        m.fillcontinents(color='coral')
        xpt, ypt = m(gcd_pts[..., -1], gcd_pts[..., -2])
        m.scatter(xpt, ypt, c=cols, s=3, alpha=alpha)
        plt.show()

