# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'n_oct' flattened octahedron net xy
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import BaryNet
from hhg9.domains.nets import net_layouts
from hhg9.h9 import H9K, H9P
from hhg9.algorithms.geometry import inside_triangle_cw


class OctantNet(ComponentDomain):
    """
    This a 2D side of an Octant that belongs to a Net.
    Validity should be easy enough since we have the 3 points that define it.
    """
    def __init__(self, registrar, dom, name: str, sign: tuple, mode: int):
        super().__init__(registrar, name, dom, mode, sign, 2)

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        raise NotImplementedError


class OctahedralNet(CompositeDomain):
    """
    This is a 2d net correlate of the Octahedron.
    Triangles have edge length √2 in a unit octahedron
    """
    from hhg9 import Points

    R3 = H9K.R3
    GW = H9K.lattice.U * 3  # grid unit width U = H9GC.U H9GC.W/6
    GH = H9K.lattice.V * 3  # grid unit height
    RT = np.pi / 3.      # grid rotation in 60º

    def __init__(self, registrar, *, layout='mortar'):
        c_oct = registrar.domain('c_oct')
        b_oct = registrar.domain('b_oct')
        if layout not in net_layouts:
            layout = 'mortar'
        super().__init__(registrar, f'n_oct:{layout}', 2)
        tp = H9P.sv  # mode vertices
        self.sides = {}
        self.projs = {}
        self.face_tris = {}
        self.sign_to_side = {}
        self.c_oct = c_oct
        self.b_oct = b_oct
        self.layout = net_layouts[layout]
        grid_xy = np.array(list(self.layout['grid'].values()))[:, :2]
        # gx_min, gx_max = grid_xy[:, 0].min(), grid_xy[:, 0].max()
        # gy_min, gy_max = grid_xy[:, 1].min(), grid_xy[:, 1].max()

        # Each placed face contributes [x_off, x_off + 2*GW] and [y_off, y_off + 3*GH]
        # So total width/height is span of gx,gy plus the single-face width/height
        # self.wi = (gx_max - gx_min + 2) * self.GW
        # self.he = (gy_max - gy_min + 3) * self.GH
        self.wi = self.layout['width'] * H9K.derived.W
        self.he = self.layout['height'] * H9K.derived.H

        self.oid_mo = np.zeros((8,), dtype=np.uint8)
        for sign, val in self.layout['grid'].items():
            side = self.c_oct.signs[sign]
            oid = self.c_oct.face_id[side]
            bary = b_oct.sides[side]
            gx, gy, th = val
            x_off = gx * self.GW
            y_off = gy * self.GH
            n_theta = (th % 6) * self.RT
            flipped = th % 2
            mode = {0: 1, 1: 0}[bary.mode] if flipped else bary.mode
            n_sig = f'{self.name}:{side}'
            b_sig = f'{b_oct.name}:{side}'
            self.oid_mo[oid] = mode
            self.sides[sign] = OctantNet(registrar, self, n_sig, sign, mode)
            face = BaryNet(registrar, side, b_sig, n_sig, n_theta, (x_off, y_off))
            self.projs[side] = face
            tri = H9P.sv[bary.mode]  # triangle from H9P. Use bary.mo b/c will transform!
            tri_rt = tri @ face.matrix + face.offset  # bary->net
            # Map sign→triangle and sign→side for fast lookup
            self.face_tris[sign] = tri_rt
            self.sign_to_side[sign] = side
            if 'c2' in self.layout:
                c2f = self.layout['c2'][sign]
                c2x = [(x * self.GW, y * self.GH, (t % 6) * self.RT) for (x, y, t) in c2f]
                face.c2trans = c2x
            self.components[sign] = self.sides[sign]


    def ratio(self):
        """Return width/height ratio"""
        return self.wi/self.he

    def img_adj(self):
        """
        :return: w,h adjustment to pixels (subtracted when outputting to image)
        """
        return self.layout['width'] + 0.51, self.layout['height'] + 0.51

    def image_dims(self, pixels: int):
        """Given the side of a triangle in pixels, return the image dimensions."""
        tri_w = pixels
        tri_h = pixels * self.R3 * 0.5
        l_width = self.layout['width']
        l_height = self.layout['height']
        w_a, h_a = self.img_adj()
        img_w = l_width * tri_w - w_a
        img_h = l_height * tri_h - h_a
        pix_w = int(img_w)
        pix_h = int(img_h)
        return pix_w, pix_h

    def dim_from_image(self, pix_w: int, pix_h: int):
        """given the image dimensions, return the side of a triangle in pixels"""
        img_w = float(pix_w)
        img_h = float(pix_h)
        tri_h = self.R3 * 0.5
        tri_w = 1.0
        l_width = self.layout['width']
        l_height = self.layout['height']
        w_a, h_a = self.img_adj()
        img_w += w_a
        img_h += h_a
        img_w /= l_width * tri_w
        img_h /= l_height * tri_h
        return np.rint((img_w + img_h) / 2.0)

    def valid(self, pts: NDArray) -> NDArray:
        """
        Test that points are valid
        """
        if pts.shape[-1] < 2:
            raise ValueError('Points must have 2 dimensions')
        signs = self.pt_face(pts)
        return np.any(signs != 0, axis=1)

    def pt_face(self, pts: NDArray) -> NDArray:
        """Vectorised: identify octant sign for each point in net coordinates.
        Returns (N,3) int8 array of signs (±1), or (0,0,0) for invalid.
        """
        num_points = pts.shape[0]
        out = np.zeros((num_points, 3), dtype=np.int8)
        for sign, tri in self.face_tris.items():
            mask = inside_triangle_cw(pts, tri)
            if not np.any(mask):
                continue
            out[mask] = np.array(sign, dtype=np.int8)
        return out

    def px_pt(self, x, y, pix):
        """
        Given a pixel coordinate and the side of a
        triangle in pixels return the pt in this domain
        tri_w = pixels
        tri_h = pixels * self.R3 * 0.5
        """
        l_width = self.layout['width']
        l_height = self.layout['height']
        tri_w = pix
        tri_h = pix * H9K.derived.RH
        img_w = l_width * tri_w - (l_width + 0.51)
        img_h = l_height * tri_h - (l_height + 0.51)
        ux = self.wi * x/img_w
        uy = self.he * y/img_h
        oc = self.pt_face(np.array([ux, uy]))
        return oc, ux, uy

    def binning(self, pts: Points, sig: tuple = None):
        """Identify the components of the points"""
        cmp = self.pt_face(pts.coords)
        pts.components = np.array(cmp)

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        for side in self.sides:
            self.sides[side].register_format(af)
