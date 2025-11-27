"""
Part of the H9 project
"""
import numpy as np
from hhg9.base.point_format import PointFormat
from hhg9.base.h9_engine import H9Engine, Style
from .. import Points


class OctahedralH9(PointFormat):
    """
    Addressing of a half-hexagonal grid over
    OctahedralBarycentric.
    This handles the eight face names NEA, etc.
    """
    def __init__(self):
        super().__init__('h9')
        self.engine = H9Engine()
        self.width = 34
        self.subs = {
            'x': Style.HEX,
            'f': Style.FULL,
            'c': Style.CFULL,
            'e': Style.EXTENDED,
            'h': Style.HALFHEX,
            'i': Style.NUMERIC,
        }

    def is_valid(self, address: str) -> bool:
        """
        :return: true if address is valid, false otherwise
        """
        pass

    def revert(self, address: str):
        """
        :return: bary(?) address(es)
        """
        if len(address) < 3 or address[0] == 'I':
            raise ValueError("Invalid Address")
        geo, val = address[:3], address[3:]
        if geo not in self.composite.h9map:
            raise ValueError("Invalid Octahedral Side Region (should be e.g. 'NAV'")
        hp, oc2, sign, name = self.composite.h9map[geo]
        mode = 0 if geo[2] == 'V' else 1
        c1 = oc2.index(hp)
        res = self.engine.unformat_addresses(f'{c1}{val}', mode)
        xy = res[:, :2]
        return Points(xy, self.composite, np.array([sign]))

    def format(self, arr: Points, _, sub: str):
        """
        return h9 address(es)
        :return:
        """
        width = self.width
        style = Style.HEX
        if sub != '':
            st = sub[0]
            if st in self.subs:
                style = self.subs[st]
                sub = sub[1:]
            if len(sub) > 0:
                width = int(sub)
        prf, pts, treg, thx = self.engine.addr(arr, width)
        if width <= 21:  # can use uint64. 21 b/c we have 'lost a digit' in the root/terminator.
            num_digits = pts.shape[1]
            p10 = 10 ** np.arange(num_digits - 1, -1, -1)
            numbers = (pts * p10).sum(axis=1).astype(np.uint64)
            if style == style.NUMERIC:
                return numbers[0]
            big = numbers.astype(f'<U{width}')
            body = np.char.zfill(big, width)
        else:
            body = np.array([''.join(row) for row in pts.astype('<U1')])
        if style in (style.NUMERIC, style.U64):
            return int(body[0])
        pb = np.strings.add(prf, body)
        pbs = np.strings.add(pb, treg)
        ok = np.strings.add(pbs, thx.astype('<U1'))
        return str(ok[0])
        # HEX = 0
        # FULL = 1
        # EXTENDED = 2
        # HALFHEX = 3
        # NUMERIC = 4
        # CFULL = 5
        # U64 = 6

    def format_arr(self, pts: Points, sub: str = '', prefix=True):
        """
        return h9 address(es)
        :return:
        """
        width = self.width
        style = Style.HEX
        if sub != '':
            st = sub[0]
            if st in self.subs:
                style = self.subs[st]
                sub = sub[1:]
            if len(sub) > 0:
                width = int(sub)
        arr = pts.coords
        reg = pts.components
        dom = pts.domain
        res = []

        prf, pts, treg, thx = self.engine.addr(pts, width, prefix)
        if width <= 21:  # can use uint64. 21 b/c we have 'lost a digit' in the root/terminator.
            num_digits = pts.shape[1]
            p10 = 10 ** np.arange(num_digits - 1, -1, -1)
            numbers = (pts * p10).sum(axis=1).astype(np.uint64)
            if style == style.NUMERIC:
                return numbers
            big = numbers.astype(f'<U{width}')
            body = np.char.zfill(big, width)
        else:
            body = np.array([''.join(row) for row in pts.astype('<U1')])
        if prefix:
            pb = np.strings.add(prf, body)
        else:
            pb = body
        pbs = np.strings.add(pb, treg)
        ok = np.strings.add(pbs, thx.astype('<U1'))
        return ok