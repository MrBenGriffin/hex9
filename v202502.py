import sympy as sp
import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.optimize import root
import matplotlib as mpl
import matplotlib.pyplot as plt
import simplekml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# This rotates an octahedral face onto the plane.
# It checks that a point is on the face.
# And it determines the hexagonal address for it.
# For added bonus, it calculates any latitude and longitude
# and plots it onto matplotlib at any depth one wishes. (to about 10^-30)


OCT_DIHEDRAL = np.arccos(-1. / 3.)

# 3D rotation matrices

def rx(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[1., 0, 0], [0, ct, -st], [0, st, ct]])


def ry(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])


def rz(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])


# Given any face of an octahedron,
# rotate it such that it is upright and centred.
def flatten(s):
    rots = {
        'NWP': rz(np.pi) @ rx(OCT_DIHEDRAL * 0.5) @ rz(3 * np.pi / 4.),
        'NWA': rz(np.pi) @ rx(OCT_DIHEDRAL * 0.5) @ rz(-3 * np.pi / 4.),
        'NEP': rz(np.pi) @ rx(OCT_DIHEDRAL * 0.5) @ rz(1 * np.pi / 4.),
        'NEA': rz(np.pi) @ rx(OCT_DIHEDRAL * 0.5) @ rz(-1 * np.pi / 4.),
        'SWP': rz(np.pi) @ rx(-OCT_DIHEDRAL * 0.5) @ rz(3 * np.pi / 4.),
        'SAW': rz(np.pi) @ rx(-OCT_DIHEDRAL * 0.5) @ rz(-3 * np.pi / 4.),
        'SPE': rz(np.pi) @ rx(-OCT_DIHEDRAL * 0.5) @ rz(1 * np.pi / 4.),
        'SAE': rz(np.pi) @ rx(-OCT_DIHEDRAL * 0.5) @ rz(-1 * np.pi / 4.)
    }
    return rots[s]


# generate a polygon for a kmz file.
def add_poly_kmz(kz, tpt, name=None, col=None, alpha=0.5, width=4):
    mim = max(tpt[..., 1]) - min(tpt[..., 1])
    if mim > 179.9999:
        path = [(o if o >= 0 else o + 360.0000, a) for (a, o) in tpt]
    else:
        path = [(o, a) for (a, o) in tpt]
    pol = kz.newpolygon(name=name)
    pol.outerboundaryis = path
    pol.style.polystyle.fill = 0
    pol.style.polystyle.outline = 1
    pol.style.linestyle.width = width
    if col:
        is_i = all([(i.is_integer() or i > 1) for i in col])
        (r, g, b) = [int(i) if is_i else int((i * 255)) for i in col]
        a = int(255 * alpha)
        pol.style.polystyle.color = simplekml.Color.rgb(r, g, b, a)


# add polygons for MPL plot.
# this is a 3D plot - but we are only looking at 2D
# after the face flattening operation.
def add_poly_mpl(ax, side, poly_base):
    np.random.seed(42)
    # octahedron dihedral = np.degrees(np.arccos(-1/3)) 109.47122063449069
    z_off = np.tan(np.pi / 6.) if side[0] == 'S' else -np.tan(np.pi / 6.)  # tan 30 is the z offset.
    rt2 = (2. ** 0.5)
    x_theta = np.arccos(-1. / 3.) * 0.5  # dihedral/2 = 54.73561º
    # z_theta = np.pi/6.
    poly_count = len(poly_base)
    rth = flatten(side)
    # rth = rz(np.pi) @ rx(x_theta) @ rz(np.pi/4.)  #0 Z first then X then move back..
    polys = [(rth @ poly_base[i].T) for i in range(poly_count)]  # Transpose for poly3DCollection.
    grad = np.linspace(0, 1, 256)
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, poly_count))
    np.random.shuffle(colors)
    px = Poly3DCollection([p.T + [0, 0, z_off] for p in polys], alpha=.35, facecolors=colors)
    ax.add_collection3d(px)


class Tx:
    # [0, 5, 2], [0, 3, 5], [0, 2, 4], [0, 4, 3],  # N+ AE,WA,EP,PW ΛVVΛ
    # [1, 2, 5], [1, 5, 3], [1, 4, 2], [1, 3, 4]  # S+ EA,AW,PE,WP VΛΛV

    # The point indices MUST be absolute:
    # They form the basis for points 1,4,9
    T_INDEX = {
        'Λ': {
            -507: (1, 2, 9), 480: (0, 9, 2), 831: (9, 0, 8),
            -831: (3, 4, 5), 156: (5, 0, 3), 264: (2, 3, 0),
            -264: (8, 6, 7), 723: (6, 8, 0), 507: (0, 5, 6)
        },
        'V': {
            -480: (1, 2, 9), 507: (0, 9, 2), 723: (2, 3, 0),
            -156: (3, 4, 5), 831: (5, 0, 3), 480: (0, 5, 6),
            -723: (8, 6, 7), 264: (6, 8, 0), 156: (9, 0, 8)
        }
    }

    def __init__(self, v, o='V', _gen=0, _idx=None):
        self.gen = _gen
        self.idx = _idx
        self.tri = np.array(v)
        self.chn = None
        self.pts = None
        self.lv = o

    @staticmethod
    def frac(a, b, f):  # linear fraction from a->b
        return a + (b - a) * f  # 0.3333 fraction of a->b

    def set_pts(self):
        i, j, k = self.tri
        t = np.full_like(i, 1. / 3)
        self.pts = tuple([
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

    def chd(self, idx):
        if not self.chn or idx not in self.chn:
            self.procreate(idx)
        return self.chn[idx]

    def procreate(self, chd=None):
        _CLV = {
            507: 'Λ', 723: 'V', 831: 'Λ',
            480: 'V', 264: 'Λ', 156: 'V'
        }
        if self.pts is None:
            self.set_pts()
        self.chn = {} if self.chn is None else self.chn
        ptt = self.T_INDEX[self.lv]
        kids = [chd] if chd is not None else ptt.keys()
        for ky in kids:
            c_pt_idx = ptt[ky]
            c_pts = [self.pts[v] for v in c_pt_idx]
            c_o = _CLV[abs(ky)]
            self.chn[ky] = Tx(c_pts, c_o, self.gen + 1, ky)

    def full_gen(self, gen):
        if gen > 0:
            if self.chn is None:
                self.procreate()
            for ch in self.chn:
                self.chn[ch].full_gen(gen - 1)

    def _hh(self, n: int, full=True):
        perms = {
            'V': {
                0: [1, 2, 0, 8, 9],
                1: [4, 5, 0, 2, 3],
                2: [7, 8, 0, 5, 6]
            },
            'Λ': {
                0: [3, 0, 9, 1, 2],
                1: [6, 0, 3, 4, 5],
                2: [9, 0, 6, 7, 8]
            }
        }
        return perms[self.lv][n] if full else perms[self.lv][n][:-1]

    def points(self, _repo):
        # this constructs triangles.
        if self.chn is None:
            _repo.append(self.tri)
        else:
            for child in self.chn:
                child.points(_repo)

    def hh(self, _repo, _depth=0):
        # 'full' means include all five points, instead of the defining four.
        if self.pts is None:
            self.set_pts()
        if _depth == 0:
            for idx in range(3):
                pts = np.array([self.pts[i] for i in self._hh(idx, False)])
                _repo.append(pts)
        else:
            if self.chn:
                for ch in self.chn:
                    self.chn[ch].hh_addr(_repo, _depth - 1)


class AK:
    # This is concerned with generating the actual point values
    # for a sphere.
    _ALPHA = 3.227806237143884260376580641604959964752197265625  # 𝛂 - vis. Kaseorg.
    _jac_fn = None
    _e = 1e-20

    @classmethod
    def xyz_ll(cls, spt):
        x, y, z = spt[0, ...], spt[1, ...], spt[2, ...]
        lat = np.degrees(np.arctan2(z, np.sqrt(x ** 2. + y ** 2.)))
        lon = np.degrees(np.arctan2(y, x))
        return np.stack([lat, lon], axis=-1)

    @classmethod
    def ll_xyz(cls, ll):  # convert to radians.
        phi, theta = np.radians(ll[:, 0]), np.radians(ll[:, 1])
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)  # z is 'up'
        return np.stack([x, y, z], axis=1)

    @classmethod
    def aka(cls, uvw):
        # rx = [ak(p) for p in uvw]
        # return np.stack(rx, axis=0)
        # # Convert a np.array of points of an octahedron onto the sphere.
        # # Anders Kaseorg: https://math.stackexchange.com/questions/5016695/
        # # input:  oct_pt is a Euclidean point on the surface of a unit octahedron.
        # # output: UVW on a unit sphere.
        t_uvw = np.tan((np.pi * uvw + cls._e) * 0.5)
        xu, xv, xw = t_uvw[..., 0], t_uvw[..., 1], t_uvw[..., 2]
        u2, v2, w2 = xu ** 2., xv ** 2., xw ** 2.
        y0p = xu * (v2 + w2 + cls._ALPHA * w2 * v2) ** 0.25
        y1p = xv * (u2 + w2 + cls._ALPHA * u2 * w2) ** 0.25
        y2p = xw * (u2 + v2 + cls._ALPHA * u2 * v2) ** 0.25
        pv = np.stack([y0p, y1p, y2p], axis=-1)
        np.seterr(invalid='ignore')
        _rx = pv / np.linalg.norm(pv, axis=-1, keepdims=True)
        return _rx

    @classmethod
    def alt(cls, uvw):
        # Appl. Sci. 2020, 10, 655
        # A New Coordinate System for Constructing Spherical Grid Systems
        # Kin Lei, Dongxu Qi, Xiaolin Tian
        nl = np.abs(uvw)
        l0 = 0.25 * np.pi * np.arctan2(nl[:, 0], 1. + nl[:, 1] + nl[:, 2])
        l1 = 0.25 * np.pi * np.arctan2(nl[:, 1], 1. + nl[:, 0] + nl[:, 2])
        l2 = 0.25 * np.pi * np.arctan2(nl[:, 2], 1. + nl[:, 0] + nl[:, 1])
        pv = np.stack([l0, l1, l2], axis=1)
        rx = np.sign(uvw) * pv / np.linalg.norm(pv, axis=1, keepdims=True)
        return rx

    @classmethod
    def fn_root(cls, op, tx):  # octa_point, target_sphere_point
        norm = np.linalg.norm(op, ord=1)
        val = cls.aka(np.array([op / norm])) - np.array(tx)
        return val[0]

    @classmethod
    def constraint(cls, op):  # Constraint: |u|+|v|+|w|=1 (surface of the unit octahedron)
        return np.sum(np.abs(op)) - 1

    @classmethod
    def set_jac(cls):
        if cls._jac_fn:
            return
        u, v, w = sp.symbols('u v w')  # Define symbolic variables for inputs
        tan_u = sp.tan(sp.pi * u / 2)
        tan_v = sp.tan(sp.pi * v / 2)
        tan_w = sp.tan(sp.pi * w / 2)

        u2 = tan_u ** 2
        v2 = tan_v ** 2
        w2 = tan_w ** 2

        y0p = tan_u * (v2 + w2 + cls._ALPHA * w2 * v2) ** 0.25
        y1p = tan_v * (u2 + w2 + cls._ALPHA * u2 * w2) ** 0.25
        y2p = tan_w * (u2 + v2 + cls._ALPHA * u2 * v2) ** 0.25

        # Combine outputs into a vector
        y = sp.Matrix([y0p, y1p, y2p])

        # Normalize the vector (divide by its magnitude)
        norm = sp.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        y_normalized = y / norm

        variables = [u, v, w]
        jacobian = y_normalized.jacobian(variables)
        cls._jac_fn = sp.lambdify(sp.Matrix(variables), jacobian, modules=['numpy'])

    @classmethod
    def ak_inv(cls, tsp):  # Inverse function using numerical optimization
        if not cls._jac_fn:
            cls.set_jac()

        def wrapped_jac(x, _):
            return cls._jac_fn(*x)

        result = root(
            cls.fn_root,
            np.sign(tsp) * 1. / 3.,  # initial_guess,
            args=(tsp,),
            jac=wrapped_jac,
            method='hybr', tol=1e-12
        )
        result.x /= np.linalg.norm(result.x, ord=1)
        return result.x


def hx_addr(_par, tra: tuple):
    _or = {  # use (abs)
        507: 'Λ', 264: 'Λ', 831: 'Λ',
        480: 'V', 723: 'V', 156: 'V'
    }
    _ha = {
        'Λ': {
            264: 0, 480: 0, -507: 0,
            507: 1, 156: 1, -831: 1,
            831: 2, 723: 2, -264: 2
        },
        'V': {
            156: 0, 507: 0, -480: 0,
            723: 1, 831: 1, -156: 1,
            480: 2, 264: 2, -723: 2
        }
    }
    _hex = []
    for _chd in tra:
        hx = _ha[_or[abs(_par)]][_chd]
        _hex.append(f'{abs(_par)}'[hx])  # here we are using the name of the parent.
        _par = _chd
    orx = _or[abs(_par)]       # final value 'Λ' or 'V'
    hx = _ha[orx][480 if orx == 'Λ' else 507]
    _hex.append(f'{abs(_par)}'[hx])  # here we are using the name of the parent.

    # _hex.append(_or[abs(_par)])  # the final child value is just 'Λ' or 'V'
    return ''.join(_hex)+orx


def exp(par, chd, hh):
    # VΛ convention: V is default.
    # Hex address in base 3 is [C2C1]
    # C2 can be seen as distance from Centre (0,1,2)
    # C1 can be seen as orientation (–,/,\ for 0,1,2 respectively).
    # Given a parent/child address (with child half-hex VΛ) we calculate the
    # parental half-hex identity.
    # Eg 00V => 0V0V
    # lut = [hh, 'Λ', 'V', 'Λ' if hh == 'V' else 'V']
    lut = [hh, 'V', 'Λ', 'Λ' if hh == 'V' else 'V']
    c2, c1 = divmod(chd, 3)  # unravel base 3 values.
    p_c1 = par % 3  # we only use the c1 value of parent.
    if c2 != 1:
        idx = (p_c1 - c1) % 3
    else:
        idx = (c1 - p_c1) % 3
    return lut[idx] if c2 != 2 else lut[3] if idx == 0 else lut[0]


def hint(addr):
    # 520826162014320318416260730241Λ ->
    # VVVVVVΛΛVVVVΛVVVVVΛΛΛVVVVΛΛVΛΛ ?!
    result = []
    if addr[-1] in {'Λ', 'V'}:
        chain, h = reversed(addr[:-1]), addr[-1]
    else:
        chain, h = reversed(addr), 'V'  # VΛ convention: V is default.
    chd = None
    for par in chain:
        if chd:
            h = exp(int(par), chd, h)
        chd = int(par)
        result.append(h)
        # result.append(f'{chd}{h}')
    return ''.join(reversed(result))


def print_lut():
    fn = (lambda a, b: (a - b) % 3)
    for n in range(9):
        print(f'?{n}')
        g, i = divmod(n, 3)  # g = 0/1/2 for 0..2/3..5/6..8
        for p in range(9):
            v = [f'{p}X{n}X', f'{p}Λ{n}', f'{p}V{n}', f'{p}X{n}Y']
            idx = fn((p % 3), i) if g != 1 else fn(i, (p % 3))
            rx = v[idx] if g != 2 else v[3] if idx == 0 else v[0]
            vl = exp(p, n, 'Λ')
            vv = exp(p, n, 'V')
            print(f'{p}{n}={rx}; {p}{vl}{n}Λ; {p}{vv}{n}V')


def hx_inv(val):
    reg, addr = val[:3], val[3:]
    _lut = {
        'Λ0': {
            (5, 'Λ'): -507,
            (0, 'Λ'): -507,
            (7, 'Λ'): -507,
            (4, 'V'): 480,
            (8, 'V'): 480,
            (0, 'V'): 480,
            (2, 'Λ'): 264,
            (6, 'Λ'): 264,
            (4, 'Λ'): 264
        },
        'Λ1': {
            (0, 'Λ'): 507,
            (5, 'Λ'): 507,
            (7, 'Λ'): 507,
            (1, 'V'): 156,
            (5, 'V'): 156,
            (6, 'V'): 156,
            (8, 'Λ'): -831,
            (3, 'Λ'): -831,
            (1, 'Λ'): -831,
        },
        'Λ2': {
            (8, 'Λ'): 831,
            (3, 'Λ'): 831,
            (1, 'Λ'): 831,
            (7, 'V'): 723,
            (2, 'V'): 723,
            (3, 'V'): 723,
            (2, 'Λ'): -264,
            (6, 'Λ'): -264,
            (4, 'Λ'): -264,
        },
        'V0': {
            (4, 'V'): -480,
            (8, 'V'): -480,
            (0, 'V'): -480,
            (5, 'Λ'): 507,
            (0, 'Λ'): 507,
            (7, 'Λ'): 507,
            (1, 'V'): 156,
            (5, 'V'): 156,
            (6, 'V'): 156
        },
        'V1': {
            (7, 'V'): 723,
            (2, 'V'): 723,
            (3, 'V'): 723,
            (8, 'Λ'): 831,
            (3, 'Λ'): 831,
            (1, 'Λ'): 831,
            (1, 'V'): -156,
            (5, 'V'): -156,
            (6, 'V'): -156
        },
        'V2': {
            (7, 'V'): -723,
            (2, 'V'): -723,
            (3, 'V'): -723,
            (2, 'Λ'): 264,
            (6, 'Λ'): 264,
            (4, 'Λ'): 264,
            (4, 'V'): 480,
            (8, 'V'): 480,
            (0, 'V'): 480
        }
    }
    _tl = {
        507: {5: 'Λ0', 0: 'Λ1', 7: 'Λ2'},
        264: {2: 'Λ0', 6: 'Λ1', 4: 'Λ2'},
        831: {8: 'Λ0', 3: 'Λ1', 1: 'Λ2'},
        156: {1: 'V0', 5: 'V1', 6: 'V2'},
        723: {7: 'V0', 2: 'V1', 3: 'V2'},
        480: {4: 'V0', 8: 'V1', 0: 'V2'}
    }
    hints = hint(addr)
    _ctxt = _tl[507] if reg[0] == 'N' else _tl[480]
    # This will use the first digit of the address to find out the next number.
    # eg NWP => north-east half-hex. (?initial lambda is not needed)
    # ctxt(NWP) = {5: 'Λ0', 0: 'Λ1', 7: 'Λ2'}; [5] => gives [Λ0]
    # NWP507(264, -507, -831, 264, 264, 156, 156, 264, -507, -831, -264, 723, 264, -507, -831, -831, 831, 480, 156, 156, 264, 264, 507, 507, 723, 480, 264, 480, 156, 507)
    # NWP520826162014320318416260730241Λ
    #   'ΛΛΛΛVVΛΛΛVVVVΛVVΛVΛΛVVVΛΛΛΛVΛΛ'
    result = []
    prev = None
    for _i in range(len(hints)):
        hx, h0 = int(addr[_i]), hints[_i]
        if prev:   # prev = 'Λ0', h0, hx = Λ, 2
            trx = _lut[prev][(hx, h0)]  # gives 264 (correct).
            result.append(trx)
            _ctxt = _tl[abs(trx)]  # {2: 'Λ0', 4: 'Λ2', 6: 'Λ1'}
            pos = _ctxt[hx]
        else:
            pos = _ctxt[hx]  # 'Λ0'
        prev = pos  # 'Λ0'
    # NWP 507 (264, -507, -831, 264, 264, 156, 156, 264, -507, -831, -264, 723, 264, -507, -831, -831, 831, 480, 156, 156, 264, 264, 507, 507, 723, 480, 264, 480, 156, 507)
    return result


class H9Conversion:

    R3 = 3**0.5
    W = 2**0.5
    H = 6**0.5 / 2
    Ẇ = W * 3 ** 0.5 / 3.
    ΛC, ΛF = 2*H/3., -H/3.
    VC, VF = H/3., -2*H/3.
    U, V = W/6., H/9.
    OFS = {
        (0, 'Λ', '021'): (0, V * 2.),
        (0, 'Λ', '201'): (-U, V),
        (0, 'Λ', '102'): (-U * 2., V * 2.),
        (1, 'Λ', '102'): (U, -V),
        (1, 'Λ', '120'): (U, V),
        (1, 'Λ', '210'): (U * 2., V * 2.),
        (2, 'Λ', '210'): (-U, -V),
        (2, 'Λ', '012'): (0, -V * 2.),
        (2, 'Λ', '021'): (0, -V * 4.),
        (0, 'V', '012'): (0, -V * 2.),
        (0, 'V', '210'): (-U, -V),
        (0, 'V', '120'): (-U * 2., -V * 2.),
        (1, 'V', '201'): (-U, V),
        (1, 'V', '021'): (0, V * 2.),
        (1, 'V', '012'): (0, V * 4.),
        (2, 'V', '120'): (U, V),
        (2, 'V', '102'): (U, -V),
        (2, 'V', '201'): (U * 2., -V * 2.)
    }

    @classmethod
    def in_scope(cls, ẋ, y, ud='Λ') -> bool:
        # `ẋ` is a synonym for `√3(x)`
        if ud == 'Λ':
            return cls.ΛF < y <= cls.ΛC - abs(ẋ)
        else:
            return cls.VF + abs(ẋ) < y <= cls.VC

    @classmethod
    def get_c1(cls, ẋ, y, ud='Λ'):
        if ud == 'Λ':
            if y < 0 and y <= ẋ:
                return 0
            elif -ẋ >= y > ẋ:
                return 1
            else:  # 0 < y > -ẋ:
                return 2
        else:
            if y >= 0 and y > -ẋ:
                return 0
            elif -ẋ >= y > ẋ:
                return 2
            else:  # y <= ẋ and y < 0:
                return 1

    @classmethod
    def get_c2(cls, ẋ, y, c1, ud='Λ'):
        if ud == 'Λ':
            if c1 == 0:
                if y <= -ẋ:
                    return '021'  # y <= -ẋ identifies 021
                if y <= ẋ - cls.Ẇ:
                    return '102'  # y <= ẋ-ẇ identifies 102
                return '201'     # y < -ẋ and y > ẋ - ẇ identifies 201
            if c1 == 1:
                if y >= 0:
                    return '102'  # y >= 0 identifies 102
                if y <= -ẋ - cls.Ẇ:
                    return '210'  # y ≤ -ẋ-ẇ identifies 210
                return '120'      # y < 0 and `y > -ẋ-ẇ` identifies 120
            # c1 == 2
            if y <= ẋ:
                return '210'     # `y <= ẋ identifies 210
            if y >= cls.VC:
                return '021'     # y >= h/3 identifies 021
            return '012'         # y > ẋ and y < h/3 identifies 012
        else:  # Now for 'V'
            # For points in 0V (flat),
            # y >= ẋ identifies 012
            # y >= ẇ-ẋ identifies 120
            # y < ẋ and y < ẇ-ẋ identifies `210`
            if c1 == 0:
                if y >= ẋ:
                    return '012'  # y <= -ẋ identifies 012
                if y >= cls.Ẇ - ẋ:
                    return '120'  # y >= ẇ-ẋ identifies 120
                return '210'      # y < ẋ and y < ẇ-ẋ identifies 210
            # For points in `1V` (forward),
            # y >= -ẋ identifies 201
            # y <= -h/3 identifies 012
            # y > -h/3 and y < -ẋ identifies 021
            if c1 == 1:
                if y >= -ẋ:
                    return '201'  # y >= -ẋ identifies 201
                if y <= -cls.VC:
                    return '012'  # y <= -h/3 identifies 012
                return '021'      # y > -h/3 and y < -ẋ identifies 021
            # For points in `2V` (back),
            # y <= 0 identifies 120
            # y >= ẇ+ẋ identifies 201
            # y > 0 and y < ẇ+ẋ identifies 102
            if y <= 0:
                return '120'     # y <= 0 identifies 120
            if y >= cls.Ẇ + ẋ:
                return '201'     # y >= ẇ+ẋ identifies 201
            return '102'         # y > 0 and y < ẇ+ẋ identifies 102

    @classmethod
    def xy_to_h9(cls, pt, c2='021'):
        ud = 'Λ' if c2 in {'021', '102', '210'} else 'V'
        x, y = pt
        ẋ = cls.R3 * x
        if not cls.in_scope(ẋ, y, ud):
            return None
        c1 = cls.get_c1(ẋ, y, ud)  # 0,1,2
        hx = int(c2[c1])*3 + c1
        cc2 = cls.get_c2(ẋ, y, c1, ud)
        xo, yo = cls.OFS[c1, ud, cc2]
        x2, y2 = x+xo, y+yo
        return hx, ud, (3.*x2, 3.*y2), cc2

    @classmethod
    def address(cls, pt, loc='021', _depth=32, full=False):
        result = []
        for d in range(_depth):
            vals = cls.xy_to_h9(pt, loc)
            if not vals:
                break
            hx, ud, pt, loc = vals
            result.append(f'{hx}' if not full else f'{hx}{ud}')
        result.append(f'{ud}') if not full else None
        return ''.join(result)


def tr_add(_x, _y, _loc=507, _depth=32):
    _wdt, _hgt = 2**0.5, 6**0.5 / 2
    _xg, _yg, r3, max_y = _wdt / 6.,  _hgt / 9., 3 ** 0.5, _hgt * (2. / 3.)
    _3hgt, _xg3 = _hgt - max_y, r3 * _wdt / 3.
    _lut = {
        # table of triangle, keyed by bottom centre triangle,
        # 'Λ' ordered clockwise from bottom centred triangle.
        507: [507, 723, 831, 480, 264, 156],
        831: [831, 156, 264, 723, 507, 480],
        264: [264, 480, 507, 156, 831, 723],
        # 'V' ordered counter-clockwise from top centred triangle.
        156: [156, 507, 480, 264, 723, 831],
        723: [723, 264, 156, 831, 480, 507],
        480: [480, 831, 723, 507, 156, 264]
    }
    _slices = {
        # there are three main slices - that carve the central triangle.
        # AKA [-=flat,/=fore,\=back]
        # _slices is a table of booleans which represent
        # which side a region is on for each. Each region represents
        # two triangles, an inner (positive), and an outer (marked negative).
        # clockwise from bottom.  # ex. flat,  fore,   back
        (True, False, False): 0,  # 507 below  right  left  (and -480)
        (True, True, False): 1,   # 723 below  left   left  (and -264)
        (False, True, False): 2,  # 831 above  left   left  (and -156)
        (False, True, True): 3,   # 480 above  left   right (and -507)
        (False, False, True): 4,  # 264 above  right  right (and -723)
        (True, False, True): 5,   # 156 below  right  right (and -831)
    }
    _offs = [
        # _offs are offsets to translate the sub-triangle to the centre
        # after each identity they are indexed in relation to _slices
        # eg, (0, 2. * _yg)
        (0, 2. * _yg), (_xg, _yg), (_xg, -_yg),
        (0, -2 * _yg), (-_xg, -_yg), (-_xg, _yg),
        (2 * _xg, 2 * _yg), (0, -4 * _yg), (-2 * _xg, 2 * _yg)
    ]
    if _y > max_y - np.abs(_x * r3):  # Is the point in the triangle at all?
        return tuple()
    _result = []
    for _level in range(_depth):
        x, y, h0 = (_x, -_y, -1) if _loc in {480, 156, 723} else (_x, _y, 1)
        xr3 = x * r3
        tri = _slices[(y <= 0, y >= xr3, -y <= xr3)]  # This identifies the segment.
        t_offs, t_id = tri, _lut[_loc][tri]  # the id is found from _lut.
        if tri % 2 == 1:  # test for corner regions.
            _tm = tri // 2
            # y < -(_xg3+xr3)
            # y < -((3**0.5 * 2**0.5 / 3.)+(x * 3**0.5))
            if [y < -(_xg3+xr3), y > _3hgt, y < (xr3-_xg3)][_tm]:
                tri = [4, 0, 2][_tm]
                t_offs = _tm + 6
                t_id = - _lut[_loc][tri]
        (o_x, o_y) = _offs[t_offs]  # and the offset based on it.
        _x = (o_x + _x) * 3.
        _y = (h0 * o_y + _y) * 3.
        _loc = _lut[507 if h0 > 0 else 480][tri]
        _result.append(-_loc if t_id < 0 else _loc)
    return tuple(_result)


def init_mpl():
    # mpl.rcParams['figure.frameon'] = False
    # mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.pad_inches'] = 0
    # mpl.rcParams['figure.figsize'] = (30, 30)
    # _fig, _ax0 = plt.subplots(1, 1)
    # _ax0.remove()
    _fig = plt.figure(figsize=(30, 30), dpi=100, frameon=False)
    _ax = _fig.add_subplot(111, projection='3d')
    _fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    _ax.set_xlabel('X', fontsize=30)
    _ax.set_ylabel('Y', fontsize=30)
    _ax.set_zlabel('Z', fontsize=30)
    # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    _ax.view_init(90, -90, 0)  # x,y top down.
    _ax.set_proj_type('ortho')  # FOV = 0 deg
    return _ax, _fig


def init_oct():
    # define octahedron vertices.
    vertices = [
        [0.0, 0.0, +1.0], [0.0, 0.0, -1.0],  # NS 0 1 (z is vertical)
        [+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],  # PA 2 3 (x is front to back)
        [0.0, +1.0, 0.0], [0.0, -1.0, 0.0]   # EW 4 5 (y is left to right).
    ]
    # define octahedron sides.
    side_v = {
        'NWP': [0, 2, 5], 'NWA': [0, 3, 5], 'NEP': [0, 2, 4], 'NEA': [0, 3, 4],  # N+ PW,WA,EP,EA ΛVVΛ
        'SWP': [1, 5, 2], 'SAW': [1, 5, 3], 'SPE': [1, 4, 2], 'SAE': [1, 4, 3]   # S+ WP,AW,PE,AE VΛΛV
    }
    _sides = {}
    for sk, sv in side_v.items():
        vs = tuple([vertices[vx] for vx in sv])
        o, sid = ('Λ', 507) if sk[0] == 'N' else ('V', 480)
        _sides[sk] = Tx(vs, o, 0, sid)
    return _sides


def side(uvw):  # given a 3D pt, identify the side it is on.
    _lut = {
        (+1, -1, +1): 'NWP',
        (-1, -1, +1): 'NWA',
        (+1, +1, +1): 'NEP',
        (-1, +1, +1): 'NEA',
        (+1, -1, -1): 'SPW',
        (-1, -1, -1): 'SAW',
        (+1, +1, -1): 'SPE',
        (-1, +1, -1): 'SAE'
    }
    kx = np.sign(uvw)[0].astype(int).tolist()
    key = tuple(kx)
    return _lut[key]


def rr(_n, v_min, v_max):
    return (v_max - v_min) * np.random.rand(_n) + v_min


def draw_repo(_ax, _fig, _reg, _origin, _zoom, name):
    ht = np.sqrt(6) / 2.
    wx = 0.5 * 3 ** -zoom
    min_x, max_x = cx - wx / 2., cx + wx / 2.
    min_y, max_y = cy - wx / 2., cy + wx / 2.
    zx = 1e-50
    ax.set_title(f'{name}: {_zoom}', fontsize=24, x=0.5, y=0.14)
    ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y), zlim=(-zx, zx))
    ax.set_aspect('equal', adjustable='box')
    # rt3 = 3. ** .5
    # tx = np.array([[2 / 3., 0], [-1 / 3., rt3 / 3.]])
    ri = {i: [] for i in [507, 723, 831, 480, 264, 156]}
    ra = []
    xs, ys = [_origin[0]], [_origin[1]]
    for n in range(len(xs)):
        x, y = xs[n], ys[n]
        hx = tr_add(x, y, 507, 14)
        if hx:
            ra.append(hx)
            hl = hx[-1]
            ri[abs(hl)].append([x, y, 0.25, 'white'])
    for n, v in ri.items():
        va = np.array(v)
        if len(v) > 0:
            ax.scatter(va[:, 0].astype('float64'),
                       va[:, 1].astype('float64'),
                       va[:, 2].astype('float64'), marker='o', s=50, color=va[:, 3])
    plt.show()


if __name__ == '__main__':
    # hints = hint('520826162014320318416260730241Λ')
    # print_lut()
    # generate an octahedron and flatten it.
    # While I can generate 2D values
    # I might want to use this to discover the h9 address from gcd.
    geod = Geodesic.WGS84
    jk = AK()
    jk.set_jac()
    unit_octahedron = init_oct()
    heng = jk.ll_xyz(np.array([[51.17886376133564, -1.826177068348142]]))
    # NWP520826162014320318416260730241Λ  Home!
    # home = jk.ll_xyz(np.array([[51.58739155518026, -0.09633464340741414]]))
    h_region = side(heng)
    # g_region = side(heng)
    rot_m = flatten(h_region)
    hp = jk.ak_inv(heng)
    # hg = jk.ak_inv(heng)
    bc = rot_m @ hp  # Z will be 0.57735 or so; tan(π/6.) (30º)
    # bg = rot_m @ hg  # Z will be 0.57735 or so; tan(π/6.)
    region = h_region
    cx, cy = bc[0], bc[1]
    # cx, cy = -0.35460115, -0.40824698
    oct_region = unit_octahedron['NWP']
    depth = 30
    path = tr_add(cx, cy, 507, depth)
    hexa = region + hx_addr(507, path)
    hex2 = H9Conversion.address((cx, cy), '021', depth, full=False)
    # a_hint = hint(hx_addr(507, path))
    # org = hx_inv(hexa)
    # hex2 = region + hx_addr(507, org)
    # b_hint = hint(hx_addr(507, org))
    # print(f'{region}{507}{path}\n{hexa}\n{hex2}\n{region}{507}{org}\n{a_hint}\n{b_hint}')
    repo = []
    regs = [oct_region]
    for i in range(len(path)):
        regs.append(regs[-1].chd(path[i]))
    for level in [2]:
        regs[level].procreate()
        start = regs[max(0, level - 3)]
        zoom = level - 2
        for i in range(6):
            start.hh(repo, i)

        ax, fig = init_mpl()
        add_poly_mpl(ax, region, repo)
        draw_repo(ax, fig, region, (cx, cy), zoom, hex2)
