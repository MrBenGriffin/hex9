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
        self.width = 32
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
        if len(address) < 3:
            raise ValueError("Invalid address")
        geo, val = address[:3], address[3:]
        if geo not in self.composite.h9map:
            raise ValueError("Invalid Octahedral Side Region (should be e.g. 'NAV'")
        hp, oc2, sign, name = self.composite.h9map[geo]
        res = self.engine.oct_decode(f'{hp}{val}', oc2)
        return Points(np.array([res]), self.composite, np.array([sign]))

    def format(self, arr: Points, dom, sub: str):
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
            if len(sub) > 1:
                width = int(sub)
        ad = self.engine.oct_encode(arr, dom.tr, width, style)
        if ad is None:
            return f'XXX:{arr[0]},{arr[1]}'
        if hasattr(dom, 'geo'):
            ofs = 2 if style == Style.FULL else 1
            geo9 = f'{dom.geo[ad[0]]}{dom.mode}{ad[ofs:]}'
            return geo9
        return ad

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
        for pt, c in zip(arr, reg):
            context = dom.components[tuple(c)]
            ad = self.engine.oct_encode(pt, context.tr, width, style)
            if ad is not None:
                if prefix:
                    res.append(f'{context.geo[ad[0]]}{context.mode}{ad[1:]}')
                else:
                    tri = context.tr.index(ad[0])
                    if context.mode == 'V':
                        cx = [8, 5, 0][tri]
                    else:
                        cx = [8, 0, 5][tri]
                    res.append(f'{cx}{ad[1:]}')
            else:
                res.append('0' * width)
        if style == style.NUMERIC:
            return np.array(res, dtype=np.uint64)
        return res
