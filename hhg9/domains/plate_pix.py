"""
Part of the H9 project
This is the Plate Carrée Projection in 2D Cartesian (X/Y)
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base import Domain, Points


class PlatePixel(Domain):

    def __init__(self, registrar):
        super().__init__(registrar, 'p_plt')
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

    def image(self, pts: Points, width: int=None, height: int=None) -> NDArray:
        """
        return the image that these points represent.
        """
        h = self.height if not height else int(height)
        w = self.width if not width else int(width)
        xs, ys = pts.coords[:, 0], pts.coords[:, 1]
        x0 = np.min(xs)
        y0 = np.min(ys)
        y_adj = (h-1e-6)/(np.max(ys)-y0)
        x_adj = (w-1e-6)/(np.max(xs)-x0)
        yy = np.floor(y_adj*(ys-y0)).astype(np.uint64)
        xx = np.floor(x_adj*(xs-x0)).astype(np.uint64)

        ch = pts.samples.astype(self.type)
        y = (h - 1) - yy.astype(np.uint64)  # still in cartesian (ie, 0 is bottom left).
        x = xx.astype(np.uint64)
        channels = 1 if ch.ndim == 1 else ch.shape[1]
        ch = ch.reshape(-1, channels)
        img = np.zeros((h, w, channels), dtype=self.type)
        img[y, x] = ch
        return img

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of GCD points
        """
        return True
