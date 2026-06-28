# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'n_oct' flattened octahedron net xy
"""

from collections import namedtuple
import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import BaryNet
from hhg9.domains.nets import net_layouts
from hhg9.h9 import H9K, H9P, H9O
from hhg9.algorithms.geometry import inside_convex_polygon_cw


# A dynamic ("local") net unfolding for a chosen subset of octants.
#   layout    : {oid: (matrix (2,2), offset (2,))} placing each octant in the plane.
#   residual  : worst seam-closure error across hinges (≈0 ⇒ exact; large ⇒ a
#               reused per-octant matrix is mis-oriented for its hinge, i.e. the
#               flavour cut this pair apart — see project_local_net approach (b)).
#   unreached : octants requested but not connected to the fundamental via oid_nb.
LocalLayout = namedtuple('LocalLayout', 'layout residual unreached')


class OctantNet(ComponentDomain):
    """
    This a 2D side of an Octant that belongs to a Net.
    Validity should be easy enough since we have the 3 points that define it.
    """
    def __init__(self, registrar, dom, name: str, oid: int, mode: int):
        super().__init__(registrar, name, dom, oid, 2)
        self.mode = mode  # need to over-ride here!

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
    RT = np.pi / 3.        # grid rotation in 60º
    # GT = np.pi / 3.        # base for net theta 30º

    def __init__(self, registrar, *, layout='mortar', theta=None):
        c_oct = registrar.domain('c_oct')
        b_oct = registrar.domain('b_oct')
        self.base_theta = 0.0
        if layout not in net_layouts:
            layout = 'mortar'
        if theta in [None, 0, '0000']:
            net_name = f'n_oct:{layout}'
        else:
            self.base_theta = float(theta) / 1000
            net_name = f'n_oct:{layout}:{int(self.base_theta * 1000):04d}'
        super().__init__(registrar, net_name, 2)
        tp = H9P.sv  # net_mode vertices
        self.face_polys = {}
        self._registrar = registrar
        self.c_oct = c_oct
        self.b_oct = b_oct
        self.global_theta = self.base_theta * self.RT
        self.tri_w = H9K.derived.W / H9K.lattice.U
        self.tri_h = H9K.derived.H / H9K.lattice.V
        self.layout = net_layouts[layout]
        vals = list(self.layout['grid'].values())

        self.wi = self.layout['width'] * H9K.lattice.U
        self.he = self.layout['height'] * H9K.lattice.V

        self.oid_mo = np.zeros((8,), dtype=np.uint8)
        for sign, val in self.layout['grid'].items():
            c2 = None
            oid = H9O.cmp_oid[sign]
            side = H9O.oid_str[oid]
            bary = b_oct.sides[side]
            if isinstance(val, list) and isinstance(val[0], tuple):
                c2 = val[1:]
                val = val[0]
            gx, gy, th = val
            x_off = gx * self.GW
            y_off = gy * self.GH
            n_theta = (th % 6) * self.RT
            mode = int(bary.mode + th) % 2
            n_sig = f'{self.name}:{side}'
            b_sig = f'{b_oct.name}:{side}'
            self.oid_mo[oid] = mode
            self.sides[side] = OctantNet(registrar, self, n_sig, oid, mode)
            face = BaryNet(registrar, side, b_sig, n_sig, n_theta, (x_off, y_off))
            self.projs[side] = face
            if c2 is None:
                tri = H9P.sv[bary.mode]  # triangle from H9P. Use bary.mo b/c will transform!
                tri_rt = tri @ face.matrix + face.offset  # bary->net
                # Map sign→triangle and sign→side for fast lookup
                self.face_polys[sign] = [tri_rt]
            else:  # if c2 is not None:
                c2x = []
                polys = []
                for (x, y, t) in c2:  # calculate adjusted net_mode, theta units, x offset, y offset, theta-radians.
                    m = int(mode + t) % 2
                    t = int(t % 6)
                    ox = x * self.GW
                    oy = y * self.GH
                    r = t * self.RT
                    c2x.append((m, t, ox, oy, r))
                face.set_c2trans(c2x)
                for c2 in [0, 1, 2]:  # calculate adjusted net_mode, theta units, x offset, y offset, theta-radians.
                    matr, off = face.c2_affine(c2)
                    hh = H9P.hh[bary.mode, c2]
                    polys.append((hh @ matr) + off + face.offset)
                self.face_polys[sign] = polys

        # Apply global 2D rotation around net centroid if theta != 0.
        # The rotation is baked into the per-face BaryNet affines and face_polys so that
        # all existing projection machinery (BaryNet.forward/backward, pt_face) remains
        # unchanged.  b_oct addressing and the H9 grid are completely unaffected.
        if self.global_theta != 0.0:
            theta = self.global_theta
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            c = np.array([self.wi / 2.0, self.he / 2.0])
            for face in self.projs.values():
                face.offset = np.array(face.offset)
                face.offset = (face.offset - c) @ R + c
                if face.c2trans is None:
                    # Non-c2: update combined rotation matrix.
                    # forward: xn = xy @ M + off  →  xy @ (M @ R) + (off - c) @ R + c
                    face.matrix = face.matrix @ R
                else:
                    # C2: c2_affine returns (M @ C2m, c2_off).  Leave M alone; rotate each
                    # C2m → C2m @ R and c2_off → c2_off @ R so that the combined becomes
                    # M @ C2m @ R, and the full output lands in the rotated frame.
                    face.c2trans = [
                        [mo, rt, offs @ R, matr @ R]
                        for mo, rt, offs, matr in face.c2trans
                    ]
            # Rotate face_polys so pt_face hit-testing works in the rotated frame.
            self.face_polys = {
                sign: [(poly - c) @ R + c for poly in polys]
                for sign, polys in self.face_polys.items()
            }
            # Recompute tight bounding box from rotated face vertices and shift
            # everything to origin so wi/he reflect the actual rotated net extent.
            all_verts = np.vstack([v for polys in self.face_polys.values() for v in polys])
            xy_min = all_verts.min(axis=0)
            xy_max = all_verts.max(axis=0)
            shift = -xy_min
            for face in self.projs.values():
                face.offset = face.offset + shift
            self.face_polys = {
                sign: [poly + shift for poly in polys]
                for sign, polys in self.face_polys.items()
            }
            self.wi, self.he = xy_max - xy_min

    def ratio(self) -> float:
        """Return width/height ratio. Use NetPixel.ratio() for new code."""
        return self.wi / self.he

    def img_adj(self) -> tuple:
        """Pixel trim adjustment. Use NetPixel.img_adj() for new code."""
        # wi = layout['width'] * U  →  wi / W = layout['width'] / tri_w  (same as before)
        l_width = self.wi / H9K.derived.W
        l_height = self.he / H9K.derived.H
        return l_width + 0.51, l_height + 0.51

    def image_dims(self, pixels: int) -> tuple[int, int]:
        """Triangle side in pixels → image (W, H). Use NetPixel.image_dims() for new code."""
        tri_h = pixels * self.R3 * 0.5
        l_width = self.wi / H9K.derived.W
        l_height = self.he / H9K.derived.H
        w_a, h_a = self.img_adj()
        return int(l_width * pixels - w_a), int(l_height * tri_h - h_a)

    # def dim_from_image(self, pix_w: int, pix_h: int) -> float:
    #     """Image (W, H) → triangle side in pixels. Use NetPixel.dim_from_image() for new code."""
    #     tri_h = self.R3 * 0.5
    #     l_width = self.layout['width'] / self.tri_w
    #     l_height = self.layout['height'] / self.tri_h
    #     w_a, h_a = self.img_adj()
    #     img_w = (float(pix_w) + w_a) / l_width
    #     img_h = (float(pix_h) + h_a) / (l_height * tri_h)
    #     return np.rint((img_w + img_h) / 2.0)

    def filter(self, pts):
        """
        Test that points are valid
        """
        from hhg9 import Points
        from hhg9.base.points import OID_INVALID
        if not isinstance(pts, Points):
            raise TypeError('pts must be Points')
        if pts.domain != self:
            raise ValueError('pts must be in this domain')
        oids = self.pt_face(pts.coords)
        good = oids != OID_INVALID
        result = self.Points(pts.coords[good], domain=self, oid=oids[good])
        if pts.samples is not None:
            result.samples = pts.samples[good]
        return result

    def valid(self, pts: NDArray, return_signs=False) -> NDArray:
        """
        Test that points are valid
        """
        if pts.shape[-1] < 2:
            raise ValueError('Points must have 2 dimensions')
        from hhg9.base.points import OID_INVALID
        return self.pt_face(pts) != OID_INVALID

    def pt_face(self, pts: NDArray) -> np.ndarray:
        """Vectorised: identify octant for each point in net coordinates.
        Returns (N,) uint8 oid array; OID_INVALID (255) for points outside all faces.
        """
        from hhg9.base.points import OID_INVALID
        num_points = pts.shape[0]
        out = np.full(num_points, OID_INVALID, dtype=np.uint8)
        for sign, polys in self.face_polys.items():
            oid = H9O.cmp_oid[sign]
            for poly in polys:
                mask = inside_convex_polygon_cw(pts, poly)
                if not np.any(mask):
                    continue
                out[mask] = oid
        return out

    def binning(self, pts: Points, sig: tuple = None):
        """Identify the octant id of each point."""
        pts.oid = self.pt_face(pts.coords)

    def local_layout(self, octants, fundamental=None, tol=1e-7) -> LocalLayout:
        """Build a *dynamic* net unfolding for the given octants.

        Where the fixed ``net_layouts`` choose one global unfolding, this lays a
        chosen subset of octants flat by hinging them out from a fundamental
        octant — so a bounded region spanning a seam (which has no single b_oct
        plot) becomes a tear-free plane.

        Each octant keeps its own ``proj.matrix`` (so hexes stay regular); only
        the offset is recomputed relative to the fundamental.  Because every
        octant is a √2-side equilateral preserved in n_oct, the hinge across a
        shared edge is a pure translation:

            offset_B = offset_A + (matrix_A·xA_shared − matrix_B·xB_shared)

        constant along the seam.  Adjacency is ``H9O.oid_nb[A, c2] = B``; the two
        shared corners are found by g_gcd coincidence of the octant corner
        triangles (``H9P.sv``).  The worst seam-closure error is returned as
        ``residual`` — ≈0 confirms translation-only suffices for these octants.

        Args:
            octants:     iterable of octant ids (oid) to place.
            fundamental: octant to fix at the origin; defaults to the first.
            tol:         g_gcd coincidence tolerance (degrees) for shared corners.

        Returns:
            :class:`LocalLayout` (layout, residual, unreached).
        """
        from hhg9 import Points
        octants = list(dict.fromkeys(int(o) for o in octants))
        if not octants:
            return LocalLayout({}, 0.0, [])
        if fundamental is None:
            fundamental = octants[0]
        g_gcd = self._registrar.domain('g_gcd')

        mat, cb, cg = {}, {}, {}
        for oid in octants:
            side = H9O.oid_str[oid]
            proj = self.projs[side]
            if proj.c2trans is not None:
                raise NotImplementedError(
                    'local_layout: c2-split nets (e.g. rhombus) not yet supported; '
                    'use a non-c2 flavour such as butterfly.')
            mat[oid] = np.asarray(proj.matrix, dtype=float)
            corners = np.asarray(H9P.sv[self.b_oct.sides[side].mode], dtype=float)
            cb[oid] = corners                                   # (3, 2) b_oct
            cg[oid] = self._registrar.project(
                Points(corners, self.b_oct, oid), [self.b_oct, g_gcd]).coords

        off = {fundamental: np.zeros(2)}
        queue = [fundamental]
        residual = 0.0
        while queue:
            a = queue.pop(0)
            for c2 in range(3):
                b = int(H9O.oid_nb[a, c2])
                if b not in mat or b in off:
                    continue
                ia, ib = [], []                                # shared corners
                for i in range(3):
                    d = np.linalg.norm(cg[b] - cg[a][i], axis=1)
                    j = int(np.argmin(d))
                    if d[j] < tol:
                        ia.append(i)
                        ib.append(j)
                if len(ia) < 2:                                # not edge-adjacent
                    continue
                world_a = cb[a][ia] @ mat[a] + off[a]
                cand = world_a - cb[b][ib] @ mat[b]            # offset candidates
                off[b] = cand.mean(0)
                residual = max(residual, float(
                    np.linalg.norm(cand - off[b], axis=1).max()))
                queue.append(b)

        layout = {o: (mat[o], off[o]) for o in off}
        unreached = [o for o in octants if o not in off]
        return LocalLayout(layout, residual, unreached)

    def place(self, coords: NDArray, oids: NDArray, layout: dict) -> NDArray:
        """Apply a local-net *layout* to b_oct *coords*: ``coords·matrix + offset``
        per octant.  Octants absent from the layout map to NaN.

        Args:
            coords: (N, 2) b_oct coordinates.
            oids:   (N,) octant id per coordinate.
            layout: the ``layout`` dict from :meth:`local_layout`.
        """
        coords = np.asarray(coords, dtype=float)
        oids = np.asarray(oids)
        out = np.full_like(coords, np.nan)
        for oid, (mtx, off) in layout.items():
            m = oids == oid
            if np.any(m):
                out[m] = coords[m] @ mtx + off
        return out

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        for side in self.sides:
            self.sides[side].register_format(af)
