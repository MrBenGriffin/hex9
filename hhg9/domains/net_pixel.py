# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
NetPixel domain — a raster pixel grid aligned to an OctahedralNet layout.

Pixel coordinates (integer column j, row i, origin top-left) are mapped to
n_oct world coordinates via the net's wi × he extent:

    x = wi * (j + 0.5) / pix_w
    y = he * (pix_h - i - 0.5) / pix_h

This is the pixel counterpart to OctahedralNet: it owns all image-sizing
and pixel↔world-coordinate concerns so that OctahedralNet can remain a
purely geometric domain.
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base import Points
from hhg9.base.domain import Domain


class NetPixel(Domain):
    """Pixel raster domain for an OctahedralNet layout.

    Parameters
    ----------
    registrar :
        The shared Registrar instance.
    n_oct : OctahedralNet or None
        The net whose geometry drives pixel sizing. If None, looked up lazily
        as ``n_oct:butterfly`` on first use.
    """

    def __init__(self, registrar, n_oct=None):
        super().__init__(registrar, 'n_pix', 2)
        self._n_oct = n_oct  # may be None until first use
        self._registrar = registrar
        self.width: int | None = None
        self.height: int | None = None
        self.type = np.uint8

    @property
    def n_oct(self):
        if self._n_oct is None:
            self._n_oct = self._registrar.domain('n_oct:butterfly')
        return self._n_oct

    # ------------------------------------------------------------------
    # Image-sizing methods (moved from OctahedralNet)
    # ------------------------------------------------------------------

    def ratio(self) -> float:
        """Width/height aspect ratio of the net layout."""
        return self.n_oct.wi / self.n_oct.he

    def img_adj(self) -> tuple[float, float]:
        """Fractional pixel trim to avoid rounding artifacts at layout edges."""
        l_width = self.n_oct.layout['width'] / self.n_oct.tri_w
        l_height = self.n_oct.layout['height'] / self.n_oct.tri_h
        return l_width + 0.51, l_height + 0.51

    def image_dims(self, pixels: int) -> tuple[int, int]:
        """Given a triangle side length in *pixels*, return image (W, H) in pixels."""
        from hhg9.h9 import H9K
        tri_w = pixels
        tri_h = pixels * H9K.R3 * 0.5
        l_width = self.n_oct.layout['width'] / self.n_oct.tri_w
        l_height = self.n_oct.layout['height'] / self.n_oct.tri_h
        w_a, h_a = self.img_adj()
        return int(l_width * tri_w - w_a), int(l_height * tri_h - h_a)

    def dim_from_image(self, pix_w: int, pix_h: int) -> float:
        """Given image pixel dimensions, return the triangle side length in pixels."""
        from hhg9.h9 import H9K
        tri_h = H9K.R3 * 0.5
        l_width = self.n_oct.layout['width'] / self.n_oct.tri_w
        l_height = self.n_oct.layout['height'] / self.n_oct.tri_h
        w_a, h_a = self.img_adj()
        img_w = (float(pix_w) + w_a) / (l_width * 1.0)
        img_h = (float(pix_h) + h_a) / (l_height * tri_h)
        return float(np.rint((img_w + img_h) / 2.0))

    # ------------------------------------------------------------------
    # Domain interface
    # ------------------------------------------------------------------

    def valid(self, pts: NDArray) -> NDArray:
        """Delegate validity to the underlying n_oct geometry."""
        return self.n_oct.valid(pts)

    def adopt(self, img: NDArray, pixels: int | None = None) -> Points:
        """Convert a raster image to Points in n_oct world coordinates.

        Parameters
        ----------
        img : (H, W, C) uint8 array
        pixels : triangle side length in pixels. If None, derived from image width.

        Returns
        -------
        Points with coords in n_oct world space and samples from the image.
        """
        if img.ndim != 3:
            raise ValueError(f'{img.shape} does not represent a 2D image (expected H×W×C).')
        h, w, c = img.shape
        if pixels is None:
            pixels = w
        self.width, self.height, self.type = w, h, img.dtype
        pix_w, pix_h = self.image_dims(pixels)

        j, i = np.meshgrid(np.arange(w), np.arange(h))
        x = self.n_oct.wi * ((j.ravel() + 0.5) / pix_w)
        y = self.n_oct.he * ((pix_h - i.ravel() - 0.5) / pix_h)
        coords = np.stack([x, y], axis=1)
        samples = img.reshape(-1, c).astype(img.dtype, copy=False)
        return Points(coords, self, samples=samples)

    def image(self, pts: Points, dims: tuple[int, int] | None = None,
              fill_value: int = 0) -> NDArray:
        """Scatter Points back into a raster image.

        Parameters
        ----------
        pts : Points with coords in this domain's world space and samples.
        dims : (W, H). Defaults to the dimensions set by the last adopt().
        fill_value : background fill for pixels with no point.
        """
        if dims is not None:
            w, h = int(dims[0]), int(dims[1])
        elif self.width is not None:
            w, h = self.width, self.height
        else:
            raise ValueError('dims not provided and adopt() has not been called.')

        c = int(pts.samples.shape[1]) if pts.samples.ndim == 2 else 1
        out = np.full((h, w, c), fill_value, dtype=pts.samples.dtype)

        pix_w, pix_h = w, h  # output pixel dims
        x, y = pts.coords[:, 0], pts.coords[:, 1]
        j = np.rint(x / self.n_oct.wi * pix_w - 0.5).astype(np.int64)
        i = np.rint(pix_h - y / self.n_oct.he * pix_h - 0.5).astype(np.int64)
        m = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        out[i[m], j[m], :] = pts.samples[m].reshape(-1, c)
        return out
