
import numpy as np
from modern.octahedron import Octahedron
from modern.hhg_tetrahedral import HHGTetrahedral
from modern.util import Util

# This currently conflates the H9 addressing / Hierarchic hexagons
# and the 2D projected octahedron 'net' map.
# This is 'old'


class H9Side:
    def __init__(self, ctx, key, ud, theta, offs, hx, c2, c1=(0, 1, 2)):
        d120 = 2 * np.pi / 3.
        self.ctx = ctx
        self.util = ctx.util
        self.oct = ctx.oct
        self.ud = ud
        self.grid_theta = theta
        self.rot_theta = ctx.oct_th[ud]
        self.theta = self.rot_theta if key != 'NWP' else 0.5 * np.pi / 1.5
        if self.ud != 'V':
            self.theta += np.pi - d120
        else:
            self.theta += d120
        self.a_theta = self.theta + np.pi  # hmm. octagon is inverted.
        self.offs = offs
        self.hx = hx
        self.c2 = c2
        self.c2s = f'{c2[0]}{c2[1]}{c2[2]}'
        self.c1 = c1
        self._rev = {self.hx[self.c2[i]]: 3*self.c2[i]+i for i in range(3)}
        self._fwd = {v: k for k, v in self._rev.items()}
        if self.c1[2] == 1:
            self._fwd[3*self.c2[0]+1] = self._fwd[3*self.c2[0]]
            self._fwd[3*self.c2[2]+1] = self._fwd[3*self.c2[2]+2]
        self.matrix = self.oct.matrices[key]

    def addr(self):
        return [f'{h}{self.ud}' for h in self.hx]

    def addr_pt(self, uvw):
        return self.util.d3_2(uvw @ (self.matrix.T @ self.util.rz(self.a_theta)))

    def enc(self, pt):                      # Given X,Y address...
        addr = self.ctx.grid.encode(pt, self.c2s, 32)
        # (0, 1, 2)
        if addr is None:
            print(f'pt {pt} failed for some reason')
            return None
        hb = self._fwd[int(addr[0])]   # eg 5
        return f'{hb}{self.ud}{addr[1:]}'   # {addr[1]}{hn}{addr[2::2]}'

    def encode(self, uvw):  # convert an octahedral address to H9.
        return self.enc(self.addr_pt(uvw))

    def dec(self, sig, addr):  # decode to a full 3D octahedral.
        xy = self.decode_xy(sig, addr)
        return self.util.d2_3(np.array([xy]), self.ctx.oct.i3) @ (self.matrix.T @ self.util.rz(self.a_theta)).T

    def decode_xy(self, sig, addr=''):  # decode to x,y for the side.
        v = self._rev[sig]
        return self.ctx.grid.decode(f'{v}{addr}')


class H9Octahedron:
    oct_th = {
        # having rotated from octahedron
        # this adjusts so the N/S point is apex.
        # use either in Z or r2d.
        'V': np.pi / -3,
        'Λ': np.pi / 1.5
    }

    def __init__(self, o):
        super().__init__()
        self.util = Util()
        self.oct = o
        self.grid = HHGTetrahedral()
        rt = np.pi / 1.5  # grid rotation in 120º
        self.r3 = o.r3
        gw = o.r2 / 2.  # grid unit width
        gh = o.r6 / 6.  # grid unit height
        glx, gly = (0, o.r2 * 3.5), (0, gh * 9)
        self.gw = gw
        self.glx = glx
        self.gly = gly
        self.gh = gh
        self.sides = {
            'NEA': H9Side(self, 'NEA', 'V', 0, (gw * 3., gh * 4.),
                          #  'V' 120= 1*3+0;2*3+1;0*3+2; = [3,7,2]
                          ('NE', 'EA', 'NA'), (1, 2, 0)),  # [3,7,2]
            'NEP': H9Side(self, 'NEP', 'Λ', +rt, (gw * 4., gh * 5.),
                          #  'Λ' 102= 1*3+0;0*3+1;2*3+2 = [3,1,8]
                          ('NE', 'NP', 'EP'), (1, 0, 2)),  # [3,1,8]
            'NWA': H9Side(self, 'NWA', 'Λ', -rt, (gw * 2., gh * 5.),
                          #  'Λ' 210= 2*3+0;1*3+1;0*3+2 = [6,4,2]
                          ('NA', 'NW', 'WA'), (2, 1, 0)),  # [6,4,2]
            'NWP': H9Side(self, 'NWP', 'V', +rt, (gw * 2., gh * 7.),
                          #  'V' 012= 0*3+0;1*3+1;2*3+2;   = [0,4,8]
                          #  'V' 012= 0*3+1;1*3+1;2*3+1    = [1,4,7]
                          ('NW', 'WP', 'NP'), (1, 2, 0), (1, 1, 1)),  # [1,4,7]
            'SEA': H9Side(self, 'SEA', 'Λ', 0, (gw * 3., gh * 2.),
                          #  'Λ' 021= 0*3+1;+2*3+1;+1*3+1; = [1,7,4]
                          ('SA', 'EA', 'SE'), (0, 2, 1), (1, 1, 1)),
            'SEP': H9Side(self, 'SEP', 'V', +rt, (gw * 5., gh * 4.),
                          #  'V' 012= 0*3+0;1*3+1;2*3+2;   = [0,4,8]
                          ('SP', 'SE', 'EP'), (0, 1, 2)),
            'SWA': H9Side(self, 'SWA', 'V', -rt, (gw * 1., gh * 4.),
                          #  'V' 201= 2*3+0;0*3+1;1*3+2; = [6,1,5]
                          ('WA', 'SA', 'SW'), (2, 0, 1)),
            'SWP': H9Side(self, 'SWP', 'Λ', 0, (gw * 6., gh * 5.),
                          #  'Λ' 021= 0*3+0;2*3+1;1*3+2; = [0,7,5]
                          ('SP', 'SW', 'WP'), (0, 2, 1))
        }
        self._adr_side = {a: k for k, v in self.sides.items() for a in v.addr()}
        self.side_ud = {
            'NEA': 'V',
            'NEP': 'Λ',
            'NWA': 'Λ',
            'NWP': 'V',
            'SEA': 'Λ',
            'SEP': 'V',
            'SWA': 'V',
            'SWP': 'Λ'
        }
        self.grid_xy = {  # from barycentre origin to map.
            'NWP': (gw * 2., gh * 7.),
            'NWA': (gw * 2., gh * 5.),
            'NEA': (gw * 3., gh * 4.),
            'NEP': (gw * 4., gh * 5.),
            'SEA': (gw * 3., gh * 2.),
            'SEP': (gw * 5., gh * 4.),
            'SWP': (gw * 6., gh * 5.),
            'SWA': (gw * 1., gh * 4.)
        }
        # self.offs = {
        #     'NWP': (gw * 2., gh * 7.),
        #     'NWA': (gw * 2., gh * 5.),
        #     'NEA': (gw * 3., gh * 4.),
        #     'NEP': (gw * 4., gh * 5.),
        #     'SEA': (gw * 3., gh * 2.),
        #     'SEP': (gw * 5., gh * 4.),
        #     'SWP': (gw * 6., gh * 5.),
        #     'SWA': (gw * 1., gh * 4.)
        # }
        self.grid_th = {  # from up/down to map where r60=ccw 60º
            'NEA': 0,
            'NEP': +rt,
            'NWA': -rt,
            'NWP': +rt,
            'SEA': 0,
            'SEP': +rt,
            'SWA': -rt,
            'SWP': 0
        }
        # pts = p2s @ Util.r2d(np.pi / -3)  # This does Λ
        # pts = p2s @ Util.r2d(np.pi / 1.5)  # This does V

    def dec(self, addr):
        side, key, body = self.h9side(addr)
        return self.sides[side].dec(key, body)

    def h9side(self, addr):
        if addr is None:
            return None, None, None
        key = addr[:3]
        return self._adr_side[addr[:3]], key[:2], addr[3:]

    def pt_face(self, uvw):
        """
         Which face does this point belong to?
         Calculated by testing the polarity/sign of each dimension.
         This does *not* test that the points are on the surface of the octahedron.
         That is done with pt_valid / pts_valid
        """
        if np.all(uvw):  # not = 0..
            key = np.sign(uvw)
        else:
            dx = np.mean(uvw, keepdims=True) * 1E-100
            key = np.sign(uvw - dx)
        return self.oct.pt_signs[tuple(key.astype(int))]

    def enc(self, uvw, key=None):
        if key is None:
            key = self.pt_face(uvw)
        return self.sides[key].encode(uvw)

    def xy_side(self, ax, ay):
        gh = self.gh * 3
        gx = ax // self.gw
        gy = ay // self.gh
        dẋ = self.r3 * ax
        if ay - gh <= dẋ <= ay + 5 * gh:  # We are in legal space...
            if dẋ <= ay + gh:  # We are in left-3 triangles
                if 5 * gh - dẋ > ay and gy > 2:
                    if 3 * gh - dẋ < ay:
                        if gy >= 6:
                            return 'NWP'
                        return 'NWA'
                    return 'SWA'
                return None
            if gy <= 5 and ay >= 3 * gh - dẋ:  # inside remaining 5
                if dẋ <= ay + 3 * gh:  # We are in mid-3 triangles
                    if gy <= 2:
                        return 'SEA'
                    if 5 * gh - dẋ > ay:
                        return 'NEA'
                    return 'NEP'
                if gy >= 3:
                    if 7 * gh - dẋ > ay:
                        return 'SEP'  # final 2 triangles
                    return 'SWP'
        return None
