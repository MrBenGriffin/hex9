import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class Display:

    @classmethod
    def colours(cls, n, cmap='tab10'):
        return mpl.colormaps[cmap](np.linspace(0, 1, n))

    @classmethod
    def show_pts_2d(cls, pts, x_lim=None, y_lim=None, label=None):
        fig = plt.figure(figsize=(10, 10), dpi=150, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111)
        if label is not None:
            ofx = (x_lim[1] - x_lim[0]) / 20
            ofy = (y_lim[1] - y_lim[0]) / 20
            ax.text(x_lim[0] + ofx, y_lim[0] + ofy, label, fontsize=40)
        if x_lim is not None:
            ax.set(xlim=x_lim, ylim=y_lim)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.01)
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
        ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=0.01)
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
    def show_pts_3d(cls, pts, x_lim=None, y_lim=None, z_lim=None, label=None):
        fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax = fig.add_subplot(111, projection='3d')
        ax.text(-1, -1, 0., label, 'x', fontsize=25)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        ax.set_proj_type('ortho')  # FOV = 0 deg
        # ax.view_init(10, 45+90, 0)  #45 SEA 225 SWP
        # ax.view_init(90, -90, 0)  # √ x,y top down.
        ax.auto_scale_xyz(x_lim, y_lim, z_lim)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', s=0.5)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
