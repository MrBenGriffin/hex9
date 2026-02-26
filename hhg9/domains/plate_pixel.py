# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""hhg9 Plate/Pixel domain.

This domain represents a 2D raster grid (pixels) but *not* necessarily
Plate Carrée (lon/lat). It is intended as an intermediary domain when
adopting / emitting raster images for arbitrary projections.

Key idea: pixel indices (i,j) are mapped to continuous 2D coordinates
(x,y) via an affine transform (origin + pixel_size), so the same raster
can represent Plate Carrée, Web Mercator, LAEA, etc.
"""

import numpy as np
from hhg9.base import Points
from hhg9.base.domain import Domain
from numpy.typing import NDArray


class PlatePixel(Domain):

    def __init__(
        self,
        registrar,
        width: int = 360,
        height: int = 180,
        dtype=np.uint8,
        origin: tuple[float, float] = (0.0, 0.0),
        pixel_size: tuple[float, float] = (1.0, 1.0),
        extent: tuple[float, float, float, float] | None = None,
        y_up: bool = True,
        center: bool = True,
    ):
        """Create a pixel grid domain.

        Parameters
        ----------
        width, height:
            Raster dimensions in pixels.
        dtype:
            Default dtype for images in this domain.
        origin:
            (x0, y0) world-coordinate origin of the pixel grid.
            Interpretation depends on `center`.
        pixel_size:
            (sx, sy) size of one pixel in world units (e.g. degrees or metres).
        extent:
            Optional world-coordinate bounds as (xmin, ymin, xmax, ymax).
            If provided, `origin`/`pixel_size` are derived from it (unless
            overridden later), so the pixel grid represents only that region.
        y_up:
            If True, y increases as i decreases (north-up convention). If False,
            y increases as i increases (image coordinate convention).
        center:
            If True, the (x,y) returned for a pixel is at its centre.
            If False, (x,y) is the pixel's corner at (j,i).
        """
        super().__init__(registrar, 'p_pix', 2)
        self.height = int(height)
        self.width = int(width)
        self.type = dtype

        self.y_up = bool(y_up)
        self.center = bool(center)

        self.extent = None if extent is None else (
            float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])
        )

        if self.extent is not None:
            xmin, ymin, xmax, ymax = self.extent
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f'Invalid extent (expected xmin<xmax and ymin<ymax): {self.extent}'
                )
            self.origin = (xmin, ymin)
            self.pixel_size = ((xmax - xmin) / self.width, (ymax - ymin) / self.height)
        else:
            self.origin = (float(origin[0]), float(origin[1]))
            self.pixel_size = (float(pixel_size[0]), float(pixel_size[1]))

    def adopt(
        self,
        img: NDArray,
        *,
        origin: tuple[float, float] | None = None,
        pixel_size: tuple[float, float] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        y_up: bool | None = None,
        center: bool | None = None,
    ) -> Points:
        """Take an image array and adopt it as points in this domain.

        This converts the raster into a `Points` cloud where:
        - `coords` are the world (x,y) for each pixel (via extent and/or origin/pixel_size)
        - `samples` are the pixel samples (RGB/RGBA/etc.)

        You can override origin/pixel_size/y_up/center per-call, which is handy
        when the raster is in some projection other than Plate Carrée.
        """
        if len(img.shape) == 3:
            (h, w, c), t = img.shape, img.dtype

            # Allow per-call overrides (without mutating defaults unless used).
            # Priority: extent > explicit origin/pixel_size > instance defaults.
            extent_eff = self.extent if extent is None else (
                float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])
            )

            if extent_eff is not None:
                xmin, ymin, xmax, ymax = extent_eff
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError(
                        f'Invalid extent (expected xmin<xmax and ymin<ymax): {extent_eff}'
                    )
                ox, oy = xmin, ymin
                sx, sy = (xmax - xmin) / w, (ymax - ymin) / h
            else:
                ox, oy = self.origin if origin is None else (float(origin[0]), float(origin[1]))
                sx, sy = self.pixel_size if pixel_size is None else (float(pixel_size[0]), float(pixel_size[1]))

            # Cache dims/dtype on the domain instance.
            self.height, self.width, self.type = h, w, t
            self.origin = (ox, oy)
            self.pixel_size = (sx, sy)
            self.extent = extent_eff
            self.y_up = self.y_up if y_up is None else bool(y_up)
            self.center = self.center if center is None else bool(center)

            # Pixel index grids (i: row, j: col)
            i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            # Map indices -> world coordinates.
            # If center=True, use pixel centres at (j+0.5, i+0.5); otherwise corners at (j, i).
            dj = j + (0.5 if self.center else 0.0)
            di = i + (0.5 if self.center else 0.0)
            x = ox + dj * sx
            if self.y_up:
                # north-up: y decreases with increasing i
                y = oy + (h - di) * sy
            else:
                # image coords: y increases with increasing i
                y = oy + di * sy

            coords = np.stack([x, y], axis=-1).reshape(-1, 2)
            samples = img.reshape(-1, c).astype(t, copy=False)
            return Points(coords, self, samples=samples)
        else:
            raise ValueError(f'{img.shape} does not seem to represent a 2D image.')

    def image(self, pts: Points, dims=None, fill_value=0) -> NDArray:
        """Reconstruct an image from `Points`.

        If `pts` represents a full dense grid created by `adopt`, this will
        return a (H, W, C) array. If `pts` is sparse, points are scattered into
        the output image using nearest pixel indices.

        Parameters
        ----------
        dims:
            Optional (W, H). Defaults to this domain's configured width/height.
        fill_value:
            Value used for pixels with no corresponding point when scattering.
        """
        if dims is None:
            w, h = int(self.width), int(self.height)
        else:
            w, h = int(dims[0]), int(dims[1])

        c = int(pts.samples.shape[1]) if pts.samples.ndim == 2 else 1
        out = np.full((h, w, c), fill_value, dtype=pts.samples.dtype)

        # Invert world coords -> indices.
        ox, oy = self.origin
        sx, sy = self.pixel_size
        x, y = pts.coords[:, 0], pts.coords[:, 1]

        if self.center:
            j = (x - ox) / sx - 0.5
            if self.y_up:
                i = (h - (y - oy) / sy) - 0.5
            else:
                i = (y - oy) / sy - 0.5
        else:
            j = (x - ox) / sx
            if self.y_up:
                i = h - (y - oy) / sy
            else:
                i = (y - oy) / sy

        jj = np.rint(j).astype(np.int64)
        ii = np.rint(i).astype(np.int64)
        m = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)

        out[ii[m], jj[m], :] = pts.samples[m].reshape(-1, c)
        return out

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of points
        """
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f'Expected (N,2) coords, got {pts.shape}.')

        # Interpret validity in world coords, based on this domain's configured extent.
        if self.extent is not None:
            x0, y0, x1, y1 = self.extent
        else:
            ox, oy = self.origin
            sx, sy = self.pixel_size
            w, h = int(self.width), int(self.height)
            x0, y0 = ox, oy
            x1, y1 = ox + w * sx, oy + h * sy

        x, y = pts[:, 0], pts[:, 1]
        return (x >= min(x0, x1)) & (x < max(x0, x1)) & (y >= min(y0, y1)) & (y < max(y0, y1))

    def set_grid(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        extent: tuple[float, float, float, float] | None = None,
        origin: tuple[float, float] | None = None,
        pixel_size: tuple[float, float] | None = None,
        y_up: bool | None = None,
        center: bool | None = None,
    ) -> 'PlatePixel':
        """Configure this domain for an output raster without adopting an image.

        This is useful when you know the world-space bounds you want to render
        into, but the current instance dimensions were set by a previous adopt.

        Rules / precedence
        ------------------
        - If `extent` is provided, it is interpreted as (xmin, ymin, xmax, ymax)
          and must satisfy xmin<xmax and ymin<ymax.
        - If `width`/`height` are not provided but `pixel_size` is, they are
          derived by rounding (extent span / pixel_size).
        - If `extent` is not provided, you must provide `origin` and `pixel_size`
          along with `width` and `height`.
        """
        if y_up is not None:
            self.y_up = bool(y_up)
        if center is not None:
            self.center = bool(center)

        if extent is not None:
            xmin, ymin, xmax, ymax = map(float, extent)
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f'Invalid extent (expected xmin<xmax and ymin<ymax): {(xmin, ymin, xmax, ymax)}'
                )

            # Determine output dimensions.
            if width is None or height is None:
                if pixel_size is None:
                    raise ValueError('set_grid: provide width/height or pixel_size when using extent.')
                sx, sy = float(pixel_size[0]), float(pixel_size[1])
                if sx == 0.0 or sy == 0.0:
                    raise ValueError(f'set_grid: invalid pixel_size: {(sx, sy)}')
                width = int(np.rint((xmax - xmin) / sx))
                height = int(np.rint((ymax - ymin) / sy))

            self.width = int(width)
            self.height = int(height)
            self.extent = (xmin, ymin, xmax, ymax)
            self.origin = (xmin, ymin)
            self.pixel_size = ((xmax - xmin) / self.width, (ymax - ymin) / self.height)
            return self

        # No extent: require explicit grid spec.
        if width is None or height is None:
            raise ValueError('set_grid: provide width and height when extent is not provided.')
        if origin is None or pixel_size is None:
            raise ValueError('set_grid: provide origin and pixel_size when extent is not provided.')

        self.width = int(width)
        self.height = int(height)
        self.origin = (float(origin[0]), float(origin[1]))
        self.pixel_size = (float(pixel_size[0]), float(pixel_size[1]))
        self.extent = None
        return self