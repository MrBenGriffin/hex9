# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""hhg9 Plate Carrée <-> pixel utilities.

`PlatePixel` is projection-agnostic (an affine map between (i,j) and (x,y)).
`PlateCarreePixel` is the thin Plate Carrée specialisation that:
- establishes the conventional full-sphere lon/lat extent (by default)
- converts lon/lat bounds -> pixel windows (with optional dateline wrap handling)
- converts pixel windows -> lon/lat extents

Conventions
-----------
- `extent` is (xmin, ymin, xmax, ymax) in world units.
- For Plate Carrée, world units are (lon, lat) in decimal degrees.
- Windows are (i0, i1, j0, j1) in half-open slicing form, suitable for:
    img[i0:i1, j0:j1]
- If `wrap='split'` and bounds cross the dateline, two windows are returned.
"""

from __future__ import annotations
import numpy as np
from hhg9.domains.plate_pixel import PlatePixel


class PlatePixelCarree(PlatePixel):

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
        super().__init__(
            registrar,
            width=width,
            height=height,
            dtype=dtype,
            origin=origin,
            pixel_size=pixel_size,
            extent=extent,
            y_up=y_up,
            center=center,
        )

    @classmethod
    def full_sphere(
        cls,
        registrar,
        width: int = 3600,
        height: int = 1800,
        dtype=np.uint8,
        *,
        lon0: float = -180.0,
        lat0: float = -90.0,
        lon1: float = 180.0,
        lat1: float = 90.0,
        y_up: bool = True,
        center: bool = True,
    ) -> "PlatePixelCarree":
        """Convenience constructor for a full-sphere Plate Carrée raster."""
        extent = (float(lon0), float(lat0), float(lon1), float(lat1))
        return cls(
            registrar,
            width=width,
            height=height,
            dtype=dtype,
            extent=extent,
            y_up=y_up,
            center=center,
        )

    def bounds_to_windows(
        self,
        bounds: tuple[float, float, float, float],
        *,
        clamp: bool = True,
        wrap: str = "split",
    ) -> list[tuple[int, int, int, int]]:
        """Convert lon/lat bounds -> one or more pixel windows.

        Parameters
        ----------
        bounds:
            (lon_min, lat_min, lon_max, lat_max) in decimal degrees.
            Bounds are interpreted as half-open in world space:
                lon in [lon_min, lon_max), lat in [lat_min, lat_max)
        clamp:
            If True, clamp windows to the raster bounds.
        wrap:
            How to handle dateline-crossing bounds where lon_max < lon_min:
            - 'split': return two windows split at the dateline
            - 'error': raise ValueError

        Returns
        -------
        A list of windows, each (i0, i1, j0, j1) for numpy slicing.
        """
        lon_min, lat_min, lon_max, lat_max = map(float, bounds)

        if lat_max <= lat_min:
            raise ValueError(f"Invalid lat bounds (expected lat_min < lat_max): {bounds}")

        # Determine the canonical lon domain from our configured extent.
        if self.extent is not None:
            lon0, lat0, lon1, lat1 = self.extent
        else:
            ox, oy = self.origin
            sx, sy = self.pixel_size
            lon0, lat0 = ox, oy
            lon1 = ox + self.width * sx
            lat1 = oy + self.height * sy

        # Handle dateline wrap.
        if lon_max < lon_min:
            if wrap == "split":
                left = (lon_min, lat_min, lon1, lat_max)
                right = (lon0, lat_min, lon_max, lat_max)
                return [
                    self._bounds_to_window_single(left, clamp=clamp),
                    self._bounds_to_window_single(right, clamp=clamp),
                ]
            if wrap == "error":
                raise ValueError(f"Lon bounds appear to cross the dateline: {bounds}")
            raise ValueError(f"Unknown wrap mode: {wrap!r}")

        return [self._bounds_to_window_single((lon_min, lat_min, lon_max, lat_max), clamp=clamp)]

    def _bounds_to_window_single(
        self,
        bounds: tuple[float, float, float, float],
        *,
        clamp: bool,
    ) -> tuple[int, int, int, int]:
        """Internal: bounds assumed non-wrapping and valid."""
        lon_min, lat_min, lon_max, lat_max = map(float, bounds)

        if lon_max <= lon_min:
            raise ValueError(f"Invalid lon bounds (expected lon_min < lon_max): {bounds}")
        if lat_max <= lat_min:
            raise ValueError(f"Invalid lat bounds (expected lat_min < lat_max): {bounds}")

        h = int(self.height)
        w = int(self.width)
        ox, oy = self.origin
        sx, sy = self.pixel_size

        # Convert lon bounds -> column index bounds.
        if self.center:
            j_lo = (lon_min - ox) / sx - 0.5
            j_hi = (lon_max - ox) / sx - 0.5
            j0 = int(np.ceil(j_lo))
            j1 = int(np.ceil(j_hi))
        else:
            # corner-registered (pixel-as-area): bounds are in the same edge-space as `extent`.
            j_lo = (lon_min - ox) / sx
            j_hi = (lon_max - ox) / sx
            j0 = int(np.floor(j_lo))
            j1 = int(np.ceil(j_hi))

        # Convert lat bounds -> row index bounds.
        # Note: y_up means larger lat => smaller i.
        if self.y_up:
            if self.center:
                i_lo = (h - (lat_max - oy) / sy) - 0.5
                i_hi = (h - (lat_min - oy) / sy) - 0.5
                i0 = int(np.ceil(i_lo))
                i1 = int(np.ceil(i_hi))
            else:
                # corner-registered (pixel-as-area)
                i_lo = h - (lat_max - oy) / sy
                i_hi = h - (lat_min - oy) / sy
                i0 = int(np.floor(i_lo))
                i1 = int(np.ceil(i_hi))
        else:
            if self.center:
                i_lo = ((lat_min - oy) / sy) - 0.5
                i_hi = ((lat_max - oy) / sy) - 0.5
                i0 = int(np.ceil(i_lo))
                i1 = int(np.ceil(i_hi))
            else:
                i_lo = (lat_min - oy) / sy
                i_hi = (lat_max - oy) / sy
                i0 = int(np.floor(i_lo))
                i1 = int(np.ceil(i_hi))

        if clamp:
            i0 = max(0, min(h, i0))
            i1 = max(0, min(h, i1))
            j0 = max(0, min(w, j0))
            j1 = max(0, min(w, j1))

        # Ensure half-open slice is sane (can be empty if bounds miss the raster).
        if i1 < i0:
            i0, i1 = i1, i0
        if j1 < j0:
            j0, j1 = j1, j0

        return i0, i1, j0, j1

    def window_to_extent(
        self,
        window: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        """Convert a pixel window -> lon/lat extent (xmin, ymin, xmax, ymax)."""
        i0, i1, j0, j1 = map(int, window)
        h = int(self.height)
        ox, oy = self.origin
        sx, sy = self.pixel_size

        # X is always left->right.
        lon_min = ox + j0 * sx
        lon_max = ox + j1 * sx

        # Y depends on y_up.
        if self.y_up:
            lat_top = oy + (h - i0) * sy
            lat_bot = oy + (h - i1) * sy
        else:
            lat_bot = oy + i0 * sy
            lat_top = oy + i1 * sy

        lat_min = min(lat_bot, lat_top)
        lat_max = max(lat_bot, lat_top)

        return float(lon_min), float(lat_min), float(lon_max), float(lat_max)

    def crop(
        self,
        img: np.ndarray,
        bounds: tuple[float, float, float, float],
        *,
        clamp: bool = True,
        wrap: str = "split",
    ) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
        """Crop an image by lon/lat bounds (lon_min, lat_min, lon_max, lat_max).

        Returns a list of (img_crop, extent) tuples.
        If `wrap='split'` and bounds cross the dateline, you'll get two crops.
        """
        windows = self.bounds_to_windows(bounds, clamp=clamp, wrap=wrap)
        out: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        for win in windows:
            i0, i1, j0, j1 = win
            img_crop = img[i0:i1, j0:j1].copy()
            extent = self.window_to_extent(win)
            out.append((img_crop, extent))
        return out