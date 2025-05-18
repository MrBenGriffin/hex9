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
        return np.array(res).view(Points).set_domain(region)

    def format(self, arr: Points, sub: str):
        """
        return h9 address(es)
        :return:

        HEX = 0  (default)
        FULL = 1
        EXTENDED = 2
        HALF.HEX = 3
        h9/x = default.
        h9/h.12
        """
        subs = {
            'x': Style.HEX,
            'f': Style.FULL,
            'e': Style.EXTENDED,
            'h': Style.HALFHEX,
        }
        width = self.width
        style = Style.HEX
        if sub != '':
            st = sub[0]
            if st in subs:
                style = subs[st]
                sub = sub[1:]
            if len(sub) > 1:
                width = int(sub)
        if arr.dom.name not in self.component:
            raise ValueError(f"Unknown Domain '{arr.dom.name}' for OctahedralH9 under {self.composite.name}")
        composite = self.component[arr.dom.name]
        res = self.engine.encode(arr, '210', width, style)
        if res is not None:
            return f'{composite.geo}{res}'
        return f'XXX:{arr[0]},{arr[1]}'
