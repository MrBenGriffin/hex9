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
        self.width = 20
        self.subs = {
            'x': Style.HEX,
            'f': Style.FULL,
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
        if geo not in self.composite.sides:
            raise ValueError("Invalid Octahedral Side Region (should be e.g. 'NWP'")
        region = self.composite.sides[geo]
        res = self.engine.decode(val)
        return Points(np.array([res]), self.composite, np.array([region.sign]))

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
        ad = self.engine.encode(arr, dom.c1, width, style)
        if ad is None:
            return f'XXX:{arr[0]},{arr[1]}'
        if hasattr(dom, 'geo'):
            return dom.geo + dom.m1[ad[0]]+ad[1:]
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
            m1 = context.m1
            ge = context.geo
            ad = self.engine.encode(pt, context.c1, width, style)
            if ad is not None:
                base = m1[ad[0]]+ad[1:]
                if prefix:
                    base = ge+base
                res.append(base)
            else:
                res.append('0' * width)
        if style == style.NUMERIC:
            return np.array(res, dtype=np.uint64)
        return res
