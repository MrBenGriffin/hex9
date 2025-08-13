"""
Part of the H9 project
This is the Plate Carrée Domain in 2D Cartesian (X/Y)
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base import Domain, Points


class PlatePixel(Domain):

    def __init__(self, registrar):
        super().__init__(registrar, 'p_plt', 2)
        self.height = 180
        self.width = 360
        self.type = np.uint8

    def adopt(self, img: NDArray):
        """
        Take an array and adopt as this domain.
        Have to override as this has a pixel structure.
        :returns: Points
        """
        if len(img.shape) == 3:
            (h, w, c), t = img.shape, img.dtype
            y, x = np.meshgrid(np.arange(h)[::-1], np.arange(w), indexing='ij')
            pts = np.concatenate([img, x[..., np.newaxis], y[..., np.newaxis]], axis=-1)
            self.height, self.width, self.type = h, w, t
            arr = pts.reshape(-1, c+2)  # This now has the colours, followed by the indices.
            coords = arr[:, -2:]
            cols = (arr[:, 0:3]).astype(img.dtype)
            return Points(coords, self, samples=cols)
        else:
            raise ValueError(f'{img.shape} does not seem to represent a 2D image.')

    def image(self, pts: Points, dims=None) -> NDArray:
        """
        return the image that these points represent.
        """
        if dims is None:
            xs, ys = pts.coords[:, 0], pts.coords[:, 1]
            ux, uy = np.unique(xs, axis=0), np.unique(ys, axis=0)
            w = ux.size
            h = uy.size
        else:
            w, h = dims
        img = pts.samples.reshape(h, w, -1)  # This now has the colours, followed by the indices.
        return img

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of GCD points
        """
        return True
