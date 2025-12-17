# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base.projection import Projection


class PlatePixelNet(Projection):
    """
        Convert Plate Carrée pixel to Barycentric Net
    """

    def __init__(self, registrar, n_dom):
        super().__init__(registrar, 'pix_net', 'n_pix', n_dom)
        self.n_oct = registrar.domain(n_dom)
        self.l_height = self.n_oct.layout['height']
        self.l_width = self.n_oct.layout['width']
        self.wi = self.n_oct.wi
        self.he = self.n_oct.he
        self.w_a, self.h_a = self.n_oct.img_adj()

    def forward(self, pts: Points) -> NDArray:
        """
        INPUT:  Plate Carrée coordinates (origin bottom-left)
        OUTPUT: Barycentric Net
        """
        mx, my = pts.domain.width, pts.domain.height
        # mx, my = pts.domain.width+self.w_a, pts.domain.height+self.h_a
        # dx, dy = self.wi / mx, self.he / my
        # px, py = pts.coords[:, 0].astype(np.uint32), pts.coords[:, 1].astype(np.uint32)
        # nx = px * dx + 1.0 * dx
        # ny = py * dy + 1.0 * dy
        rs = np.stack([mx, my], dtype=np.float64, axis=-1)
        dv = self.n_oct.valid(rs)
        return Points(rs[dv], domain=self.fwd_cs, samples=pts.samples[dv])

    def backward(self, pts: Points) -> NDArray:
        """
        Convert net to pixel (x, y) coordinates.
        """
        raise NotImplementedError('net to image not yet supported')
