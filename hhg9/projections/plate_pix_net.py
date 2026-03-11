# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Projection: NetPixel (n_pix) ↔ OctahedralNet (n_oct).

Pixel indices (j=column, i=row, top-left origin) map to n_oct world coords:
    x = wi * (j + 0.5) / pix_w
    y = he * (pix_h - i - 0.5) / pix_h
where (pix_w, pix_h) = n_pix.image_dims(pixels).
Points outside the net faces are filtered on forward.
"""
import numpy as np
from hhg9 import Points
from hhg9.base.projection import Projection


class PlatePixelNet(Projection):
    """Pixel raster → OctahedralNet world coordinates."""

    def __init__(self, registrar, n_dom):
        super().__init__(registrar, 'pix_net', 'n_pix', n_dom)
        self.n_oct = registrar.domain(n_dom)
        self.wi = self.n_oct.wi
        self.he = self.n_oct.he

    def forward(self, pts: Points) -> Points:
        """n_pix pixel coords → n_oct world coords, filtered to valid faces."""
        pix_w = pts.domain.width
        pix_h = pts.domain.height
        px = pts.coords[:, 0]
        py = pts.coords[:, 1]
        x = self.wi * (px + 0.5) / pix_w
        y = self.he * (pix_h - py - 0.5) / pix_h
        xy = np.stack([x, y], axis=1)
        valid = self.n_oct.valid(xy)
        samples = pts.samples[valid] if pts.samples is not None else None
        return Points(xy[valid], domain=self.fwd_cs, samples=samples)

    def backward(self, pts: Points) -> Points:
        """n_oct world coords → n_pix pixel coords."""
        pix_w = self.rev_cs.width
        pix_h = self.rev_cs.height
        x, y = pts.coords[:, 0], pts.coords[:, 1]
        px = x / self.wi * pix_w - 0.5
        py = pix_h - y / self.he * pix_h - 0.5
        coords = np.stack([np.rint(px), np.rint(py)], axis=1)
        samples = pts.samples if pts.samples is not None else None
        return Points(coords, domain=self.rev_cs, samples=samples)
