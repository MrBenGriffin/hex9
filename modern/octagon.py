import numpy as np
from geographiclib.geodesic import Geodesic
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from modern.octahedron_h9 import H9Octahedron
from util import Util
from ak import AK
from osprojection import OSProjection
from grid_h9 import GridH9
from photo import Photo


# This rotates an octahedral face onto the plane.
# It checks that a point is on the face.
# And it determines the hexagonal address for it.
# It calculates he h9 address of stonehenge,
# and plots it onto matplotlib at any depth one wishes. (to about 10^-30)
# 3D rotation matrices
# It now plots to a global map, including the full list of containing half-hexes.



class Grid:  # Octahedral Side
    def __init__(self, _ti, _c2: tuple, _nm: tuple, _c1: [tuple | None] = None):
        self.h9 = GridH9()
        self.ti = _ti  # 'VΛ'
        self.c1 = _c1  # c1 override (normally 0,1,2)
        self.nm = _nm  # hex names, following c2.
        self.c2 = _c2  # (c2 values in 0,1,2 order)
        self.c2s = ''.join([f'{_i}' for _i in _c2])
        self.hx = None
        self._set_hx()

    def _set_hx(self):
        k = tuple([3 * self.c2[i] + i for i in range(3)])
        if self.c1:
            v = self.c1
            self.c1 = (1, 1, 1)
            # NWP: 0->1, 4->4, 8->7 (from SEP)
            # SEA: 5->4, 0->1, 7->7 (from SWP
        else:
            v = k
        self.hx = {k: w for k, w in zip(k, v)}
        done = True


class Octant:
    R3 = 3 ** 0.5
    OCT_DIHEDRAL = np.arccos(-1. / 3.)
    OCT_EDGE = 2 ** 0.5
    TRI_HEIGHT = OCT_EDGE * R3 / 2
    OCT_DEVIATION = 1e-200  # some edge needed here.
    GRID_W = OCT_EDGE / 2.
    GRID_H = TRI_HEIGHT / 3.
    GRID_Z = R3 / 3.

    # Each of eight parts into which a space or solid body is divided by three planes which intersect
    # (especially at right angles) at a single point. Here they respect that the radius = 1 (ie unit sphere)
    # This defines the edge-length of the Octahedron to be √2
    def __init__(self, _vertices, ud: str, _h9_rot: float, _map_rot: float, _offset: tuple):
        self.u = Util()
        self.vertices = tuple(_vertices)  # a 3-tuple in 3D Euclidean coordinates that defines the plane intersections.
        self.ud = ud  # V/
        self.e = np.mean(self.vertices, axis=0) * self.OCT_DEVIATION
        self.off_val = _offset
        self.offs = [f * g for (f, g) in zip(list(_offset), [self.GRID_W, self.GRID_H, self.GRID_Z])]
        self.proj = None
        self.grid = None
        self.examples = None
        self.normal = None
        self.set_normal()
        self.matrix = self.set_matrix()
        self.h9_rot = self.u.rz(- (_h9_rot - 0.5) * np.pi / 6.)  # negative to go clockwise.
        self.map_rot = self.u.rz(- _map_rot * np.pi / 6.)  # negative to go clockwise.
        self.map_r2d = self.u.r2d(- _map_rot * np.pi / 6.)  # negative to go clockwise.
        self.map_a2d = self.u.r2d(4*np.pi/3. + (_map_rot * np.pi) / 6.)  # negative to go clockwise.
        self.map_matrix = self.set_map_matrix(_map_rot)

    def set_normal(self):
        u, v, w = np.array(self.vertices)
        base = np.cross(v - u, w - u)
        self.normal = base / np.linalg.norm(base)

    def set_matrix(self):
        # Sets the octahedron rotation.
        target = np.array([0, 0, 1])
        normal = self.normal
        axis = np.cross(normal, target)  # Compute rotation axis and angle
        angle = np.arccos(np.dot(normal, target))
        axis = axis / np.linalg.norm(axis)  # Normalise
        sk_s = np.array([  # Compute skew-symmetric cross-product matrix
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rtm = np.eye(3) + np.sin(angle) * sk_s + (1. - np.cos(angle)) * np.dot(sk_s, sk_s)
        return rtm

    def set_map_matrix(self, _rot):
        if _rot == 0:
            return self.matrix
        return self.map_rot @ self.matrix

    def set_proj(self, proj: OSProjection):
        self.proj = proj

    def set_examples(self, _examples):
        self.examples = _examples

    def set_grid(self, grid: Grid):
        self.grid = grid

    def adr_rotate(self):  # for h9 addresses.
        if self.ud == 'V':
            return (self.u.my() @ (self.h9_rot @ self.matrix)).T
        else:
            return (self.h9_rot @ self.matrix).T

    def map_rotate(self):  # for grid drawing.
        return self.map_matrix.T

    def map_rotation(self, flat=False):  # once something has already been adr. rotated
        return self.map_r2d.T if flat else self.map_rot.T

    def offset(self, pts, inverse=False):
        if pts.shape[-1] == 3:
            if not inverse:
                pts[..., 0] += self.offs[0]
                pts[..., 1] += self.offs[1]
                pts[..., 2] += self.offs[2]
            else:
                pts[..., 0] -= self.offs[0]
                pts[..., 1] -= self.offs[1]
                pts[..., 2] -= self.offs[2]
        else:
            if not inverse:
                pts[..., 0] += self.offs[0]
                pts[..., 1] += self.offs[1]
            else:
                pts[..., 0] -= self.offs[0]
                pts[..., 1] -= self.offs[1]
        return pts

    def face3d(self):
        return self.vertices

    @classmethod
    def frac(cls, a, b, f):  # linear fraction from a->b
        na, nb = np.array(a), np.array(b)
        return na + (nb - na) * f  # 0.3333 fraction of a->b

    def sub_t_pts(self, tri):
        # Given 3 points {i,j,k} of a planar triangle
        # derive the 10 points
        # that define its subdivision into 9 triangles
        # ordered by barycentre, then i=>j=>k
        (i, j, k) = tri
        t = np.full_like(i, 1. / 3)
        return tuple([
            np.mean([i, j, k], axis=0),  # centre.
            i,  # pt i
            self.frac(i, j, t),  # i->j.
            self.frac(j, i, t),  # j->i.
            j,  # p_j.
            self.frac(j, k, t),  # j->k.
            self.frac(k, j, t),  # k->j.
            k,  # p_k.
            self.frac(k, i, t),  # k->i.
            self.frac(i, k, t)  # i->k.
        ])

    def sub_tris(self, tri):
        # given 3 points of a planar triangle
        # subdivide and return nine composite affine triangles.
        c = self.sub_t_pts(tri)
        py = [
            (1, 2, 9), (0, 9, 2), (2, 3, 0),
            (3, 4, 5), (5, 0, 3), (0, 5, 6),
            (8, 6, 7), (6, 8, 0), (9, 0, 8)
        ]
        # 0, 3, 6 are corners
        # 1, 4, 7 are adjacent to corners.
        # 2, 5, 8 are on the edges.
        return [[c[i] for i in p] for p in py]

    def sub_thh_pts(self, tri, chiral=True):
        # Given 3 points {i,j,k} of a planar equilateral triangle
        # derive the 7 points that define its
        # subdivision into 3 half-hexagons according to chirality
        # 'True' starts with long edge on i->k
        # 'False' starts with long edge on i->j
        # ordered by barycentre, then i=>j=>k
        (i, j, k) = tri
        t = np.full_like(i, 1. / 3)
        return tuple([
            np.mean([i, j, k], axis=0),  # centre.
            i,  # pt i
            self.frac(i, j, t) if chiral else self.frac(j, i, t),
            j,  # p_j.
            self.frac(j, k, t) if chiral else self.frac(k, j, t),
            k,  # p_k.
            self.frac(k, i, t) if chiral else self.frac(i, k, t)  # i->k.
        ])

    def sub_thh(self, tri, chiral=True):
        c = self.sub_thh_pts(tri, chiral)
        # this gives us 7 pts that divide a triangle into 3 hh.
        py = [(1, 2, 0, 6), (3, 4, 0, 2), (5, 6, 0, 4)]
        return [[c[i] for i in p] for p in py]

    def sub_hh(self, hh, chiral=True):
        # Given 4 points {a,b,c,d} of a half-hexagon.
        # ⬣: a↗b b→c c↘d
        # This can also be an array of hh!
        if len(hh) == 3:
            return [
                self.sub_hh(hh[0], chiral),
                self.sub_hh(hh[1], chiral),
                self.sub_hh(hh[2], chiral)
            ]
        else:
            (a, b, c, d) = hh
            e = np.mean([a, d], axis=0)
            return [
                self.sub_thh([b, e, a], chiral),
                self.sub_thh([e, b, c], not chiral),
                self.sub_thh([c, d, e], chiral)
            ]

    def hh(self, depth, pts, chiral=True):
        if depth == 0:
            return pts
        repo = self.sub_hh(pts, chiral)
        return self.hh(depth - 1, repo, chiral)

    def tri3d(self, depth, project=False, pts=None):
        if pts is None:
            pts = [self.vertices]
        if depth == 0:
            if project:
                return self.proj.os(np.array(pts))
            else:
                return pts
        repo = []
        for pt in pts:  # This is a polygon.
            repo += self.sub_tris(pt)
        return self.tri3d(depth - 1, project, repo)

    def tri2d(self, depth, strip_z=True):
        pts = self.tri3d(depth, False)
        rds = pts @ self.map_rotate()
        tx = self.offset(rds)
        if strip_z:
            tx = np.delete(tx, 2, -1)
        return tx

    def o_adr_pt(self, uvw):
        # given 3D Octahedral, return 2D projected h9 address.
        return np.delete(uvw @ self.adr_rotate(), 2, -1)

    def o_map_pt(self, uvw):
        # given 3D Octahedral, return 2D projected visual map address.
        return np.delete(self.offset(uvw @ self.map_rotate()), 2, -1)



# class Octahedron:
#     def __init__(self, _proj: OSProjection):
#         self.hc = GridH9()
#         self._proj = _proj
#         self.hx = dict()
#         self._v = [
#             [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],  # NS 0 1 (z is vertical)
#             [0.0, +1.0, 0.0], [0.0, -1.0, 0.0],  # EW 2 3 (y is left to right).
#             [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],  # AP 4 5 (x is front to back) Atlantic/Pacific
#         ]
#         self.sides = {
#             # Names are N/S, W/E, AP (north/south, west/east, atlantic/pacific)
#             'NWP': [  # √ North West Pacific: PNW 035: 012/201/120
#                 # Points are in clockwise, starting North
#                 Octant([self._v[pt] for pt in (0, 3, 5)], 'V', 5, 10, (2, 7, -1)),
#                 # 201/714 is the C2 order
#                 Grid('V', (0, 1, 2), ('NP', 'NW', 'WP'), (1, 4, 7))
#             ],
#             'NWA': [  # √ North West Atlantic AWN: 102/021/210 430-4-2 034-2-2
#                 Octant([self._v[pt] for pt in (0, 3, 4)], 'Λ', 2, 2, (2, 5, -1)),
#                 Grid('Λ', (1, 0, 2), ('NA', 'NW', 'WA'))
#             ],
#             'NEA': [  # North East Atlantic EAN
#                 Octant([self._v[pt] for pt in (2, 4, 0)], 'V', 11, 6, (3, 4, -1)),
#                 Grid('V', (1, 2, 0), ('NA', 'NE', 'EA'))  # 1x3+0,2x3+1,0x3+2 = 372
#             ],
#             'NEP': [  # North East Pacific NPE 102/021/210 0=topleft. NEP-052-2-10==NEP-052-6-6
#                 Octant([self._v[pt] for pt in (0, 2, 5)], 'Λ', 0, 6, (4, 5, -1)),
#                 Grid('Λ', (0, 2, 1), ('NE', 'NP', 'EP'))  # 3x1+0,0x3+1,2x3+2 = 318
#             ],
#             'SEA': [  # South East Atlantic AES:421 102/021/210 : SEA-124-3-2
#                 Octant([self._v[pt] for pt in (1, 2, 4)], 'Λ', 3, 2, (3, 2, 1)),
#                 Grid('Λ', (2, 1, 0), ('SA', 'SE', 'EA'), (7, 4, 1))
#             ],
#             'SEP': [  # South East Pacific:PSE  012/201/120
#                 Octant([self._v[pt] for pt in (5, 1, 2)], 'V', 8, 10, (5, 4, 1)),
#                 Grid('V', (2, 0, 1), ('SP', 'SE', 'EP'))  # 0x3+0,1x3+1,2x3+2 = 048
#             ],
#             'SWP': [  # South West Pacific:SPW 021 (102,021,210) SWP-153-7-10
#                 Octant([self._v[pt] for pt in (1, 5, 3)], 'Λ', 7, 10, (6, 5, 1)),
#                 Grid('Λ', (2, 1, 0), ('SP', 'SW', 'WP'))  # 0x3+0, 2x3+1, 1x3+2 = 075
#             ],
#             'SWA': [  # South West Atlantic:WAS 012/201/120
#                 Octant([self._v[pt] for pt in (3, 4, 1)], 'V', 2, 2, (1, 4, 1)),
#                 Grid('V', (0, 1, 2), ('SA', 'SW', 'WA'))  # 2x3+0,0x3+1,1x3+2 = 615
#             ]
#         }
#         self.signs = {
#             (+1, -1, +1): 'NWA',
#             (-1, -1, +1): 'NWP',
#             (+1, +1, +1): 'NEA',
#             (-1, +1, +1): 'NEP',
#             (+1, -1, -1): 'SWA',
#             (-1, -1, -1): 'SWP',
#             (+1, +1, -1): 'SEA',
#             (-1, +1, -1): 'SEP'
#         }
#         self.grid_offs = {s[0].off_val[:-1]: k for (k, s) in self.sides.items()}
#         for ky, (_o, _g) in self.sides.items():
#             _o.set_proj(_proj)
#             _o.set_grid(_g)
#             for n in _g.nm:
#                 self.hx[f'{n}{_g.ti}'] = {v: k for k, v in _g.hx.items()}
#
#     def xyz_side(self, uvw):  # given a 3D pt, return the geo/grid tuple.
#         key = tuple(np.sign(uvw).astype(int).tolist())
#         return self.signs[key]
#
#     def xyz_octant(self, uvw):  # given a 3D pt return the geo
#         return self.sides[self.xyz_side(uvw)][0]
#
#     def s_o(self, uvw):  # given a Spherical return the Octahedral.
#         if uvw.shape == 1:
#             return self._proj.so(uvw)
#         else:
#             return np.apply_along_axis(self._proj.so, -1, uvw)
#
#     def o_s(self, uvw):  # given an Octahedral return the Spherical.
#         return self._proj.os(uvw)
#
#     def side(self, ref):
#         return self.sides[ref]
#
#     def valid(self, pts):
#         # given an array of 3d points, ensure that they are all on the surface of the unit octahedron.
#         return np.all(np.apply_along_axis((lambda a: np.abs(np.sum(np.abs(a)) - 1.) < 1e-15), -1, pts))
#
#     def o_map_pt(self, uvw):
#         # given 3D Octahedral address, 2D projected (visual) map address.
#         _oc = self.sides[self.xyz_side(uvw)]
#         _rtp = _oc.offset(uvw @ _oc.adr_rotate())
#         return _rtp[0], _rtp[1]
#         # return np.delete(_oc.offset(uvw @ _oc.map_rotate()), 2, -1)
#
#     def o_map(self, uvw):
#         return np.apply_along_axis(self.o_map_pt, -1, uvw)  # side_keys
#
#     def xy_side(self, ax, ay):
#         gh = Octant.GRID_H * 3
#         gx = ax // Octant.GRID_W
#         gy = ay // Octant.GRID_H
#         dẋ = self.hc.R3 * ax
#         if ay - gh <= dẋ <= ay + 5 * gh:  # We are in legal space...
#             if dẋ <= ay + gh:  # We are in left-3 triangles
#                 if 5 * gh - dẋ > ay and gy > 2:
#                     if 3 * gh - dẋ < ay:
#                         if gy >= 6:
#                             return 'NWP'
#                         return 'NWA'
#                     return 'SWA'
#                 return None
#             if gy <= 5 and ay >= 3 * gh - dẋ:  # inside remaining 5
#                 if dẋ <= ay + 3 * gh:  # We are in mid-3 triangles
#                     if gy <= 2:
#                         return 'SEA'
#                     if 5 * gh - dẋ > ay:
#                         return 'NEA'
#                     return 'NEP'
#                 if gy >= 3:
#                     if 7 * gh - dẋ > ay:
#                         return 'SEP'  # final 2 triangles
#                     return 'SWP'
#         return None
#
#     def _oad_pt(self, uvw):
#         return self.sides[self.xyz_side(uvw)][0].o_adr(uvw)
#
#     def o_adr(self, uvw):
#         # only use this if all the addresses belong to different sides.
#         # otherwise use o_adr directly.
#         return np.apply_along_axis(self._oad_pt, -1, uvw)  # side_keys
#
#     def o_kma_pt(self, uvw):
#         # given 3D Octahedral address, return Side Key, 2D projected (visual) map address and H9 address.
#         k = self.xyz_side(uvw)
#         _oc = self.sides[k][0]
#         return np.array([
#             k,
#             np.delete(_oc.offset(uvw @ _oc.map_rotate()), 2, -1),
#             np.delete(uvw @ _oc.adr_rotate(), 2, -1)
#         ], dtype=object)
#
#     def o_kma(self, uvw):
#         return np.apply_along_axis(self.o_kma_pt, -1, uvw)  # side_keys
#
#     def oxy_tests(self):
#         for ref, (geo, grid) in self.sides.items():
#             vx = geo.vertices  # vertices are in Euclidean space.
#             ok = [self.xyz_side(v + geo.e) for v in vx]
#             ref_xyz = ll_xyz(np.array([geo.example]))
#             ref_oc = [self.s_o(xyx) for xyx in ref_xyz][0]
#             r_s = self.xyz_side(ref_oc)
#             if r_s != ref:
#                 print(f'{geo.example} appears to be in octant {r_s}, rather than {ref}')
#             xy = self.oct_xy(ref_oc, r_s)  # now have the octal 2D point.
#             h9a = 'unknown'
#             h9o = h9a
#             llr = None
#             try:
#                 h9o = self.xy_h9(xy, r_s)
#                 h9a = self.hc.encode(xy, grid.c2s, 15, True)
#                 cut = h9o[:12]
#                 xyc = self.h9_xy(cut)
#                 hpc = self.xy_oct(xyc, r_s)
#                 llr = xyz_ll(self.o_s(hpc))
#             except:
#                 print(f'{xy} fails with {r_s}')
#             print(f'{ref}:{geo.example}<=>{llr} is at {h9a} // {h9o} (via {h9o[:9]})')
#
#         for ref, (geo, grid) in self.sides.items():
#             vx = geo.vertices  # vertices are in Euclidean space.
#             for v in vx:
#                 if not geo.valid(v):
#                     print(f'{ref}: {v} vertex does not rest on the octahedron')
#                 _s = np.linalg.norm(v) - 1.
#                 if _s > 1E-245:
#                     print(f'{ref}: {v} vertex does not rest on the unit sphere.')
#                 # Now test s_o and o_s
#                 x1, y1, z1 = v
#                 p = self.s_o([v])
#                 x2, y2, z2 = p
#                 if x1 != x2 or y1 != y2 or z1 != z2:
#                     print(f'{ref}: {v} s_o transform affected by octahedron transform {p}')
#                 q = np.around(self.o_s(np.array([v])), 18)
#                 x3, y3, z3 = q[0]
#                 if x1 != x3 or y1 != y3 or z1 != z3:
#                     print(f'{ref}: {v} o_s transform affected by octahedron transform {q}')
#
#     def axy_oct(self, xy, ref):
#         geo = self.sides[ref][0]
#         z = - geo.offs[2]
#         x, y = xy
#         xyz = np.array([x, y, z]) @ geo.adr_rotate().T
#         return xyz
#
#     def xy_h9(self, xy, ref):  # Given side_name
#         geo, grid = self.sides[ref]
#         addr = self.hc.encode(xy, grid.c2s, 32, True)
#         if addr is None:
#             return None
#         _hex = int(addr[0])
#         hn = grid.hx[_hex]
#         hb = grid.nm[hn // 3]  # reference by C2 (hi trit)
#         return f'{hb}{addr[1]}{hn}{addr[2::2]}'
#
#     def h9_gen(self, ref, depth):
#         # Generate an entire list of all addresses at a given depth.
#         geo, grid = self.sides[ref]
#
#         # rt = self.hx[reg]
#         # normative = f'{rt}{_addr}'
#         # hints = self.hc.hint(normative)
#         # return self.hc.decode(normative, hints)
#
#     def h9_xy(self, name):
#         reg, x, _addr = name[:3], name[3], name[4:]
#         ky = self.hx[reg][int(x)]
#         return self.hc.decode(f'{ky}{_addr}')
#
#     def ll_depth(self, _uv, offset=1):
#         # uv is a latitude longitude delta that indicates the maximum resolution of a source.
#         # typically the distance between two pixels at the equator (plate carré).
#         # stonehenge returned 9 [51.17886376133564, -1.826177068348142]
#         # ref_ = [1.0, 22.5]  # stonehenge is probably not the best...
#         ref_ = [51.17886376133564, -1.826177068348142]
#         offs = [-1., 0., 1.]
#         ll_refs = [[ref_[0] + _a * _uv[0], ref_[1] + _b * _uv[1]] for _a in offs for _b in offs]
#         ref_xyz = ll_xyz(np.array(ll_refs))
#         ref_oc = [self.s_o(xyx) for xyx in ref_xyz]
#         ads = []
#
#         for oc in ref_oc:
#             _o = self.xyz_side(oc)
#             _xy = self.oct_xy(oc, _o)
#             ads.append(self.xy_h9(_xy, _o))
#         return self.h9_distinct(ads)  # if offset is zero we will lose fidelity.
#
#     @classmethod
#     def h9_shared(cls, addrs):
#         # find the index of where addresses deviate
#         idx = next((i for i, c in enumerate(zip(*addrs)) if len(set(c)) > 1), -1)
#         return addrs[0][:idx]  # they are all the same, so share the first.
#
#     @classmethod
#     def h9_distinct(cls, addrs):
#         idx = len(cls.h9_shared(addrs))
#         dx = [a[idx:] for a in addrs]  # these are the tails.
#         i = len(dx[0])
#         for i in range(len(dx[0])):
#             ab = [d[:i] for d in dx]
#             if len(set([d[:i] for d in dx])) == len(addrs):
#                 break
#         return idx + i


def set_examples(o):
    examples = {
        'NWP': {  # √ North West Pacific
            'Hollywood sign': (34.1340477, -118.321673),
        },
        'NWA': {  # √ North West Atlantic
            'Stonehenge': (51.178863, -1.826177),
        },
        'NEA': {  # North East Atlantic
            'Great Pyramid': (29.9791625, 31.134263)
        },
        'NEP': {  # √ North East Pacific
            'Fujiyama': (35.360842, 138.72737)
        },
        'SEA': {
            'Good Hope': (-34.34871533030242, 18.474066162932797),
            'i of Antarctica': (-65.23716525290308, 34.34232192365365),
            'Alfred Faure': (-46.432352614, 51.8577388),
            'Zanzibar': (-6.161186137751229, 39.189086035634965),
            'Faux Cap': (-25.567307215246085, 45.5283784696969),
            'Mt Ross': (-49.566964414395166, 69.48302368805365)
        },
        'SEP': {  # South East Pacific
            'Borobudur': (-7.607990892561542, 110.20382993394321),
            'Useless Loop': (-26.137245702750057, 113.40579384577295),
            'Bluff': (-46.59818554425654, 168.328533260205),
            'Bofu PNG': (-8.915946105324913, 148.28442734119457)
        },
        'SWP': {  # South West Pacific
            'Rapa Nui Moai': (-27.1257853, -109.276872)
        },
        'SWA': {  # South West Atlantic
            'Falklands Lighthouse': (-51.681877, -57.7203197)
        }
    }
    for s, v in examples.items():
        o.sides[s][0].set_examples(v)


def rr(_n, v_min, v_max):
    return (v_max - v_min) * np.random.rand(_n) + v_min


def tri_hollow(_o, depth=3):
    repo = []
    for key in _o.sides:
        geo, grd = _o.sides[key]
        bas = geo.tri3d(0, False)[0]  # Get the 3 lines we want to test.
        fnd = []
        for i in range(3):
            a, b = np.array(bas[i]), np.array(bas[(i + 1) % 3])
            fnd.append([a, b - a])
        trx = list(geo.tri3d(depth, False))
        pts = []
        for t in trx:
            for (a, ab) in fnd:
                if np.any(np.cross(ab, t - a) == [0, 0, 0]):
                    pts.append(t)
                    break
        repo.append(pts)
    return repo


def draw_tri_grid(_o, sph=False):
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    repo = []
    for key in _o.sides:
        geo, grd = _o.sides[key]
        repo += list(geo.tri3d(2, sph))
    px = Poly3DCollection(repo, alpha=.95, edgecolor='k', linewidth=0.25)
    ax.add_collection3d(px)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def draw_tri_map(o):
    # This draws the triangle grid onto a 3D graph.
    # It's used to verify the rotations and offsets accordingly.
    w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
    fw, fh = 4 * w, 4 * h
    mpl.rcParams['savefig.pad_inches'] = 0
    mpl.rcParams['savefig.bbox'] = 'tight'
    fig = plt.figure(dpi=100, constrained_layout=True, layout=None, frameon=False)
    fig.set_size_inches(4 * w, 4 * h)
    ax = Axes3D(fig, [0., 0., 1., 1.], proj_type='ortho')
    ax.tick_params(pad=0, color='#0000', labelcolor='#0000')
    ax.view_init(90, -90, 0)  # √ x,y top down.
    repo = []
    label = 'Triangular Grid Map'
    ax.text(0., 0.1, 0., label, 'x', fontsize=15)
    for key in o.sides:
        geo, grd = o.sides[key]
        pts = geo.tri2d(2, False)  # don't strip z.
        clp = [0, 10, 9, 16, 15, 17]
        pts = [p for i, p in enumerate(pts) if i not in clp]
        repo += list(pts)
    px = Poly3DCollection(repo, alpha=.95, edgecolor='k', linewidth=0.25)
    ax.add_collection(px)
    ax.auto_scale_xyz([0, 7. * Octant.GRID_W], [0, 9. * Octant.GRID_H], [-.01, .01])
    ax.set_aspect('equal', adjustable='box')
    fig.add_axes(ax)
    plt.show()


def draw_eff_flat(_o):
    # This draws the triangle grid onto a 2D graph.
    w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
    fw, fh = 4 * w, 4 * h
    ax, fig = init_mpl(2, fw, fh)
    ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    repo = []
    label = 'Triangular Map'
    cols = mpl.colormaps['tab20'](np.linspace(0, 1, 8))
    ax.text(0.1, 0.1, label, fontsize=20)
    for i, key in enumerate(_o.sides):
        geo, grd = _o.sides[key]
        mtx = geo.adr_rotate()
        mrt = geo.map_rotation(True)  # True = flat.
        trx = geo.tri3d(0, False) @ mtx + [0, 0, geo.GRID_Z]  # Base triangle.
        fpt = oct_eff(_o, key) @ mtx + [0, 0, geo.GRID_Z]
        res = geo.offset(np.delete(fpt, 2, -1) @ mrt).tolist()
        tri = geo.offset(np.delete(trx, 2, -1) @ mrt).tolist()
        px = PolyCollection(res + tri, label=key, alpha=.45, edgecolor='k', linewidth=0.25)
        px.set_facecolor(cols[i])
        ax.add_collection(px)
    ax.legend(loc=(0.9, 0.01))
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()


# def make_eff(z=None):
#     # make an F shape from three polygons in octant-flattened space
#     # for the purpose of testing correct orientation and reflection
#     # of each octant under projection and on the map.
#     u = Octant.OCT_EDGE / 8.
#     v = Octant.TRI_HEIGHT / 9.
#     w = v / 3
#     # u, v, w = 8, 8, 1
#     wo = [-w, +w], [+w, +w], [+w, -w], [-w, -w]
#     pt = 'h', (-1, 2), (1, 2)  # top arm points of F
#     pm = 'h', (-1, 0), (1, 0)  # middle arm points of F
#     pl = 'v', (-1, 2), (-1, -2)  # left side points of F
#     ef = []
#     for a in pt, pm, pl:
#         po = []
#         for i in range(4):  # making a quad.
#             f, g = divmod(i, 2)
#             if a[0] == 'h':
#                 px, py = a[1:][f ^ g]  # 0110
#             else:
#                 px, py = a[1:][f]  # 0011
#             ox, oy = wo[i]
#             m, n = u * px + ox, v * py + oy
#             po.append((m, n) if z is None else (m, n, z))
#         ef.append(np.array(po))
#     return np.array(ef)


# def oct_eff(_o, key):
#     # This cannot use add_rotate etc, as
#     # We need to test those against this.
#     # It *does* use the geo.matrix however.
#     rkz = {'NWP': 7.5, 'NWA': 8.5, 'NEA': 1.5, 'NEP': 2.5, 'SEA': 11.5, 'SEP': 4.5, 'SWP': 5.5, 'SWA': 10.5}
#     geo, grd = o.sides[key]
#     rmz = rz(rkz[key] * np.pi / 6.)
#     z_off = -geo.GRID_Z
#     fpt = make_eff(z_off)
#     pts = (fpt @ mx()) @ (rmz @ geo.matrix)
#     if not _o.valid(pts):
#         print(f'oct_eff: z-offset ‘{z_off}’ and rotation to an octagon failed for EFF.')
#     return pts


def draw_effs(_o):
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(22, 9, 0)  #
    tx = tri_hollow(o, 3)
    cols = mpl.colormaps['tab10'](np.linspace(0, 1, 8))
    for i, key in enumerate(o.sides):
        polys = tx[i]
        pts = oct_eff(_o, key)
        polys += pts.tolist()
        px = Poly3DCollection(polys, label=key, alpha=.95, edgecolor='k', linewidth=0.25)
        px.set_facecolor(cols[i])
        ax.add_collection3d(px)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc=(0.20, 0.18))
    plt.show()


def draw_adr_sides(_o, key):
    # Ensure that the rotation is correct for address calcs.
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.text(-1, -1, 0., key, 'x', fontsize=25)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(90, -90, 0)  # √ x,y top down.
    geo, grd = o.sides[key]
    mtx = geo.adr_rotate()
    res = oct_eff(_o, key) @ mtx + [0, 0, geo.GRID_Z]
    px = Poly3DCollection(res, alpha=.95, edgecolor='k', linewidth=0.25)
    ax.add_collection3d(px)
    pts = [_o.hc.poly(i, grd.ti, True) for i in [0, 1, 2]]
    # hhx = geo.hh(1, pts, grd.ti != 'V')
    hxf = np.array(pts).reshape((-1, 4, 3))
    trr = hxf + [0, 0, geo.GRID_Z]
    tx = Poly3DCollection(trr, alpha=.25, edgecolor='k', linewidth=1.5)
    ax.add_collection3d(tx)
    ax.auto_scale_xyz([-.8, .8], [-.8, .8], [-.001, .001])
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def do_grid_scatter(_o, samples=5000):
    # This uses a random set of spherical (xyz) coordinates,
    # identifies their h9 address, and then colours them accordingly.
    # here we are not identifying the side.
    w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
    fw, fh = 4 * w, 4 * h
    ax, fig = init_mpl(2, fw, fh)
    ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    ax.text(0.1, 0.1, 'random pts', fontsize=20)

    for name, (geo, gr) in _o.sides.items():
        cts = []
        grx = gr.c2s
        px = sph_rnd(samples)  # abs:=> fundamental octant.
        pts = np.copysign(px, geo.e)
        ptx = _o.s_o(pts)
        tx = geo.offset(ptx @ geo.map_rotate())
        ptm = np.delete(tx, 2, -1)  # These are now 2D points suitable for map placement.
        rds = ptx @ geo.adr_rotate()
        pta = np.delete(rds, 2, -1)  # These are now 2D points suitable for address identification.
        for (xym, xya) in zip(ptm, pta):
            ps = _o.hc.encode(xya, grx, 5, False)  # Good iff XY is good.
            if ps is None or len(ps) < 3:
                continue
            c = int(ps[1:3])
            cts.append([*xym, c])
        va = np.array(cts)
        xs, ys, cs = va[:, 0], va[:, 1], va[:, 2]
        ax.scatter(xs, ys, marker='o', s=5, c=cs)

    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()


def full_scatter(_o, size=5000):
    # This uses a random set of spherical (xyz) coordinates,
    # identifies their h9 address, and then colours them accordingly.
    w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
    fw, fh = 4 * w, 4 * h
    ax, fig = init_mpl(2, fw, fh)
    ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    ax.text(0.1, 0.1, 'Random spherical', fontsize=20)
    r_pts = sph_rnd(size)
    p_oct = np.apply_along_axis(_o.xyz_side, -1, r_pts)  # side_keys
    o_pts = _o.s_o(r_pts)  # octahedron points.
    cts = []
    col = col_rnd(9 ** 3)  # capturing 3 characters.
    for (ky, pt) in zip(p_oct, o_pts):
        oc = _o.sides[ky][0]
        tx = oc.offset(pt @ oc.map_rotate())
        rds = pt @ oc.adr_rotate()
        ptm = np.delete(tx, 2, -1)
        pta = np.delete(rds, 2, -1)
        h9a = _o.xy_h9(pta, ky)
        c = int(h9a[3:5])
        cts.append([*ptm, c])
    va = np.array(cts)
    xs, ys, cs = va[:, 0], va[:, 1], va[:, 2]
    ax.scatter(xs, ys, cmap=col, marker='o', s=0.25, c=cs)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(f'tri_map_{size}.png')
    plt.show()


def round_trip(_o: Octahedron):
    u = Util()
    # convert octahedron examples into addresses and then back again.
    ex = [e for side in _o.sides for e in _o.sides[side][0].examples.values()]
    en = [k for side in _o.sides for k in _o.sides[side][0].examples.keys()]
    ex_ll = np.array(ex)
    ex_xyz = u.ll_xyz(ex_ll)  # examples are held in lat/lon pairs.
    oct_fk = np.apply_along_axis(_o.xyz_side, -1, ex_xyz)  # octagon keys (can use xyz or oct)
    ex_oct = _o.s_o(ex_xyz)  # project onto octahedron points.
    oct_bk = _o.o_s(ex_oct)
    if not np.allclose(ex_xyz, oct_bk):
        print(f'Projection error found for {ex_ll}')
    for (plc, lla, ky, pt3) in zip(en, ex_ll, oct_fk, ex_oct):
        oc = _o.sides[ky][0]  # Octant instance to use for this pt (via key from o.xyz_side(pt))
        art = pt3 @ oc.adr_rotate()
        pta = np.delete(art, 2, -1)  # 2D Grid Address.
        mrt = pt3 @ oc.map_rotate()
        mro = oc.offset(mrt)
        ptm = np.delete(mro, 2, -1)  # 2D Map address.
        h9_raw = _o.hc.encode(pta, oc.grid.c2s, 32, True)
        hf, hh = h9_raw[0::2], h9_raw[1::2]
        try:
            h9a = _o.xy_h9(pta, ky)
            ipa = np.array(_o.h9_xy(h9a))
            if not np.allclose(pta, ipa):
                print(f'Address Conversion/Inversion address Error found with  {pta}, {ipa} (given {h9a})')

            h9b = _o.xy_h9(ipa, ky)
            ipb = _o.h9_xy(h9b)
            ptb = _o.axy_oct(ipb, ky)  # [-0.21507068 -0.45977145  0.32515786]
            xyz = _o.o_s(ptb)
            llb = u.xyz_ll(xyz)

            if np.allclose(lla, llb) and np.allclose(pt3, ptb) and h9a == h9b:
                continue
            print(f'\nTest {ky} {plc} , {lla}; Oct:{pt3}; Grid:{pta}; Map:{ptm}; Raw:{h9_raw}')
            print(f'LL:{lla}=>{llb}')
            print(f'OC:{pt3}=>{ptb}')
            print(f'H9:{h9a}=>{h9b}')
        except:
            print(f'\nException {ky} {plc} , {lla}; Oct:{pt3}; Grid:{pta}; Map:{ptm}; Raw:{h9_raw}')


def rnd_round_trip(_o: Octahedron, size=5000):
    u = Util()
    r_pts = u.sph_rnd(size)
    p_oct = np.apply_along_axis(_o.xyz_side, -1, r_pts)  # side_keys
    o_pts = _o.s_o(r_pts)  # octahedron points.
    oct_bk = _o.o_s(o_pts)
    if not np.allclose(r_pts, oct_bk):
        print(f'rnd_round_trip: Projection errors found.')
        for (a, b) in zip(oct_bk, oct_bk):
            if not np.allclose(a, b):
                print(f'{a}, {b}')

    for (ky, pt) in zip(p_oct, o_pts):
        oc = _o.sides[ky][0]  # Octant instance to use for this pt (via key from o.xyz_side(pt))
        art = pt @ oc.adr_rotate()
        pta = np.delete(art, 2, -1)  # 2D Grid Address.
        try:
            h9a = _o.xy_h9(pta, ky)
            ipa = np.array(_o.h9_xy(h9a))
            if not np.allclose(pta, ipa):
                print(f'Address Conversion/Inversion address Error found with  {pta}, {ipa} (given {h9a})')
            h9b = _o.xy_h9(ipa, ky)
            ipb = _o.h9_xy(h9b)
            ptb = _o.axy_oct(ipb, ky)  # [-0.21507068 -0.45977145  0.32515786]
            if np.allclose(pt, ptb) and h9a == h9b:
                continue
            print(f'\nTest {ky}; Oct:{pt}; Grid:{pta};')
            print(f'OC:{pt}=>{ptb}')
            print(f'H9:{h9a}=>{h9b}')
        except:
            print(f'\nException {ky}; Oct:{pt}; Grid:{pta};')


def draw_hex_map(_o):
    print('This function is not yet implemented...')
    # This intends to use a random set of spherical (xyz) coordinates,
    # identify their h9 address, and then colours them accordingly.
    size = 10
    w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
    fw, fh = 4 * w, 4 * h
    ax, fig = init_mpl(2, fw, fh)
    ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    ax.text(0.1, 0.1, 'Random Hexagons', fontsize=20)
    col = col_rnd(5 * size)  # five scales across 500 pts.
    cts = []
    for scale in range(1, 6):
        r_pts = sph_rnd(size)  # Random Euclidean points on surface of a sphere
        o_pts = _o.s_o(r_pts)  # 3D Octahedron Projection.
        kma_pts = _o.o_kma(o_pts)  # return side id, map coordinate, h9 coordinate
        for k, mp, ap in kma_pts[:]:
            h9 = _o.xy_h9(ap, k)  # obtain the h9 encoding
            col_idx = int(h9[11:12])  # add a col for it.
            # cts.append([*mp, int(h9[11:12])])  # add a col for it.
            # hh = _o.xy_hh(ap, k, scale)  # given h9 and a scale, return hh for it.
    # va = np.array(cts)
    # xs, ys, cs = va[:, 0], va[:, 1], va[:, 2]
    ax.scatter(xs, ys, cmap=col, marker='o', s=0.25, c=cs)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(f'hex_map.png')
    plt.show()


def sample_ll_map(_o):
    # given a map and arrays of latitudes and longitudes,
    # find the colour, and display it via an enmeshed half-hexagon.
    # Because latitude/longitude is non equal area.. there are loads
    # of clashes.
    map_w, map_h = 3.5 * Octant.OCT_EDGE, 3 * Octant.TRI_HEIGHT
    scale = 10
    fig = plt.figure(figsize=(map_w * scale, map_h * scale), dpi=150, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, map_w), ylim=(0, map_h), xticks=[], yticks=[])
    p = Photo()
    p.load('../preparatory/world.topo.bathy.200406.3x5400x2700.png')
    p.set_latlon([-90., 90.], [-180., 180.])
    llm = np.array([[lat, lon] for lat in p.lat for lon in p.lon])
    # sz, stp = 3, 1
    # sz, stp = 4, 2
    # sz, stp = 5, 8
    sz, stp = 6, 8
    # llm = np.array([[lat, lon] for lat in np.linspace(-89.99999, 89.99999, stp * 90) for lon in
    #                 np.linspace(-179.99999, 179.99999, stp * 2 * 90)])
    mp_xyz = ll_xyz(llm)
    fns = np.apply_along_axis(_o.xyz_side, -1, mp_xyz)
    mp_oct = _o.s_o(mp_xyz)
    all_polys = []
    all_cols = []
    bad_poly, clash = 0, 0
    hash_bank = set()
    for (ky, pt, ll) in zip(fns, mp_oct, llm):
        c = p.col(*ll)
        oc, grd = _o.sides[ky]
        mtx = oc.adr_rotate()
        ap = [pt] @ mtx
        fpt = np.delete(ap[0], 2, -1)
        poly = _o.hc.enmesh(fpt, grd.c2s, sz, True)  # The size is affected by the samples.
        if len(poly) == 0:
            bad_poly += 1
            continue
        mrt = oc.map_rotation(True)  # True = flat.
        mapped = oc.offset(poly @ mrt)
        # mapped = np.delete(pof).tolist()
        # b1 = poly @ r2d(np.pi)  # Why is this the case?
        # mapped = oc.offset(b1)
        hsh = tuple([pk for pj in mapped.round(6).tolist() for pk in pj])
        if hsh in hash_bank:
            clash += 1
            continue
        hash_bank.add(hsh)
        all_cols.append(f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}')
        all_polys.append(mapped)
    # px = PolyCollection(all_polys, edgecolor='k', linewidth=0.001)
    px = PolyCollection(all_polys, linewidth=0)
    px.set_facecolors(all_cols)
    ax.add_collection(px)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(f'mesh_ll_{sz}.png')
    print(f'{len(all_polys)} fitted; {bad_poly} failures; {clash} clashes')


def test_xy_sides(_o):
    map_w, map_h = 3.5 * Octant.OCT_EDGE, 3 * Octant.TRI_HEIGHT
    scale = 300.
    idx = {
        'SWP': (23, 23, 129),    # far right ocean
        'SEP': (183, 75, 42),    # australia
        'NEP': (226, 190, 39),   # far east
        'NEA': (79, 172, 224),   # Europe and mid east.
        'SEA': (5, 168, 28),     # Africa
        'NWP': (108, 135, 168),  # California and W Pacific
        'NWA': (30, 153, 186),   # N Atl.
        'SWA': (42, 112, 9)      # Brazil and S Atl.
    }
    dw, dh = int(scale*map_w), int(scale*map_h)
    pd = Photo()  # destination image.
    pd.new(dw, dh)  # photo-pixels.
    for wx in range(dw):  # This gives us the pt.
        for iy in range(dh):
            wy = dh - iy - 1  # Photo uses an inverted Y.
            ax = wx / scale
            ay = wy / scale
            sk = _o.xy_side(ax, ay)
            if sk is not None:
                pd.img[iy, wx] = idx[sk]
    pd.convert()
    pd.show('direct', pause=True)


def sample_map(_o):
    # here we find the x,y of the resultant map and
    # attempt to find out what it's latitude and longitude is
    # from which we take a sample.
    map_w, map_h = 3.5 * Octant.OCT_EDGE, 3 * Octant.TRI_HEIGHT
    scale = 250.
    ps = Photo()   # source image
    ps.load('../preparatory/world.topo.bathy.200406.3x5400x2700.png')
    ps.set_latlon([-90., 90.], [-180., 180.])
    dw, dh = int(scale*map_w), int(scale*map_h)
    pd = Photo()  # destination image.
    pd.new(dw, dh)  # photo-pixels.

    for wx in range(dw):  # This gives us the pt.
        for iy in range(dh):
            wy = dh - iy - 1
            ax = wx / scale
            ay = wy / scale
            sk = _o.xy_side(ax, ay)  # This looks to be good.
            if sk is not None:
                geo = _o.side(sk)[0]
                px, py, pz = geo.offs
                mpt = np.array([ax-px, ay-py])
                pt = mpt @ geo.map_r2d.T
                p3 = np.append(pt, pz)
                # p3 = (np.array([ax, ay, 0]) - geo.offs) * [1., 1., -1.]
                po = p3 @ geo.matrix.T
                # oso = _o.xyz_side(po)
                # df = np.sum(np.abs(po))
                sph = _o.o_s(po)
                sso = _o.xyz_side(sph)

                ll = xyz_ll(np.array([sph]))
                la, lo = ll[0]
                c = ps.col(la, lo, False)
                pd.img[iy, wx] = c
    pd.convert()
    pd.save(f'direct_map')


# def draw_mesh_map(_o):
#     w, h = 7 * Octant.GRID_W, 9 * Octant.GRID_H
#     fig = plt.figure(figsize=(w, h), dpi=200, frameon=False)
#     fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
#     ax = fig.add_subplot(111)
#     ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
#     all_polys = []
#     for key in o.sides.keys():
#         geo, grd = _o.sides[key]
#         b0 = np.array([_o.hc.poly(i, grd.ti) for i in grd.c2])
#         b1 = b0 @ r2d(np.pi)  # On the map: V and L are inverted.
#         b2 = geo.offset(b1)
#         all_polys += b2.tolist()
#     px = PolyCollection(all_polys, alpha=.60, edgecolor='k', linewidth=1)
#     px.set_facecolors(['#f00a0a', '#c00000', '#800000'])  #
#     ax.add_collection(px)
#     ax.set_aspect('equal', adjustable='box')
#     plt.axis('off')
#     plt.savefig(f'mesh_map_c2.svg')
#     plt.show()


# def draw_adr_mesh(_o, key):
#     w, h = 20 * Octant.OCT_EDGE, 20 * Octant.TRI_HEIGHT
#     fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
#     fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
#     ax = fig.add_subplot(111)
#     geo, grd = _o.sides[key]
#     xg, yg = geo.offs[0], geo.offs[1]
#     ax.set(xlim=(-.81 + xg, .81 + xg), ylim=(-.82 + yg, 0.82 + yg))
#     locs = [(k, np.array(v)) for k, v in geo.examples.items()]
#     mp_xyz = ll_xyz(np.array([ll[1] for ll in locs]))
#     fns = np.apply_along_axis(_o.xyz_side, -1, mp_xyz)
#     if not np.all(fns == key):
#         print('address found outside of area')
#     mp_oct = _o.s_o(mp_xyz)
#     for i, (pt, llk) in enumerate(zip(mp_oct, locs)):
#         adr_p = geo.o_adr_pt(pt)
#         polys = _o.hc.enmesh(adr_p, grd.c2s, 10, False)
#         if len(polys) == 0:
#             print(f'address {adr_p} not found for meshing')
#             return
#         mapped = geo.offset(polys @ geo.map_a2d)
#         pc = PolyCollection(mapped, alpha=.30, edgecolor='k', linewidth=0.1)
#         # if i == 0:
#         #     pc.set_facecolors(['#008e19'])  #
#         # else:
#         #     pc.set_facecolors(['#1f77b4'])  #
#         ax.add_collection(pc)
#     ax.set_aspect('equal', adjustable='box')
#     plt.axis('off')
#     plt.savefig(f'{key}_map.svg')
#     plt.show()


if __name__ == '__main__':
    # Fix octahedron <=> plane
    # The mapping of a point onto the octahedron should be ok. (AK)
    #
    # Need to do a map generation.
    # Fix round-trips.
    # To look at adding a/b/c as an address for half-hex identity of 6/7/8.
    # √Draw a half-hexagon...
    # Draw a hexagon...
    # Draw a hexagon by its address (or with a 'level' parameter).
    # To do a binning exercise looking at splitting hexes when they reach a threshold.
    np.random.seed(42)
    geod = Geodesic.WGS84
    jk = AK()
    o = H9Octahedron()
    # draw_tri_grid(o)          # Draw 3D Octahedron as a triangle grid
    # draw_tri_grid(o, True)    # Same, projected onto sphere.
    # for key in o.sides.keys():
    #     draw_adr_sides(o, key)   # check that address rotations are good.
    # draw_adr_sides(o, 'NEP')     # Or just one.
    # draw_effs(o)                # 3D Octahedron showing F in 'correct' rotation.
    draw_eff_flat(o)             # 3D Octahedron Fs flattened onto 2D flat map grid.
    # draw_tri_map(o)           # 3D Octahedron as a triangle grid, flattened onto 3D grid.

    set_examples(o)
    # test_xy_sides(o)
    # sample_map(o)
    # sample_ll_map(o)          # very slow!

    # draw_mesh_map(o)          # Ensures that c2 colouring is correct.
    # draw_adr_mesh(o, 'SEP')   # current test/debug..

    # do_grid_scatter(o)        # random spherical coordinates, coloured according to their h9 address.
    # full_scatter(o, 5000)    # this time without hints and using h9.
    # draw_hex_map(o)
    # round_trip(o)  # lat/lon => h9 and inverse
    # rnd_round_trip(o)
    # print('done')
