# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from hhg9 import Registrar
from hhg9.base.point_format import PointFormat
from hhg9.base.points import Points
import hhg9.h9.addressing as adr
from hhg9.algorithms.packing import u64_pack


class OctahedralH9(PointFormat):
    """
    Addressing of a half-hexagonal grid over OctahedralBarycentric.
    Handles the eight face names (NEA, …). Implements the PointFormat contract:
      - __init__(name)
      - format(points, composite, sub)
      - revert(address: str|list[str]) → Points
    """
    def __init__(self, registrar: Registrar) -> None:
        super().__init__(registrar, 'h9')
        self.scheme = adr.H9_RA
        self.width = 34  # default printed width
        self.subs = {
            'k': adr.TailStyle.key,
            'x': adr.Style.HEX,
            'i': adr.Style.NUMERIC,
            'u': adr.Style.UH64,
            'r': adr.Style.UR64,
        }

    def is_valid(self, address: str) -> bool:
        """Required by PointFormat; not implemented yet."""
        return True  # placeholder until a validator is added

    def revert(self, address, style=adr.Style.HEX):
        """
        :return: bary addresses(es) as Points.
        """
        result = None

        if isinstance(address, str):
            if '\n' in address:
                arr = address.splitlines()
            else:
                arr = [address]
            if style != adr.Style.HEX:
                dgt = [([int(a, 16) for a in adx]) for adx in arr]
                arr = u64_pack(np.array(dgt))
            else:
                arr = np.array(arr)
        match style:
            case adr.Style.HEX:
                result = adr.hex_str_decode(arr, self.registrar, self.scheme)
            case adr.Style.UH64:
                result = adr.hex_unpack(arr, self.registrar, self.scheme)
            case adr.Style.NUMERIC:
                result = adr.hex_unpack(arr, self.registrar, self.scheme)
            case adr.Style.UR64:
                result = adr.reg_unpack(arr, self.registrar, self.scheme)
        return result

    def _select_style(self, sub: str = None):
        """
        Given a string, determine format style
        eg h9.f33, h9.u18, h9.24
        """
        width = self.width
        style = adr.Style.HEX
        # Parse sub: optional style letter(s) + optional integer width
        if sub:
            # long token 'u' takes precedence; otherwise first char
            if sub.startswith('u'):
                width = 13
                style = self.subs['u']
                sub = sub[1:]
            elif sub.startswith('r'):
                width = 14
                style = self.subs['r']
                sub = sub[1:]
            elif sub[0] in self.subs:
                style = self.subs[sub[0]]
                sub = sub[1:]
            # numeric tail → width
            if sub:
                try:
                    width = int(sub)
                except Exception:
                    raise ValueError(f"h9f format: invalid width '{sub}' — expected integer (e.g. :h9f.33)")
        return style, width

    def format(self, arr: Points, _, sub: str):
        """
        Return H9 label(s) for the given Points.
        The `sub` string (from Points.__format__) may contain an optional style prefix
        and/or a decimal width, e.g. 'x33', 'i21', '33'. Default: hex body, width=self.width.
        """
        # if self.engine is None:
        #     self.engine = arr.domain.engine
        if not isinstance(arr, Points):
            pts = arr
        else:
            pts = arr
        tail_style = adr.TailStyle.reversible
        count = len(pts)
        style, width = self._select_style(sub)
        if style == adr.TailStyle.key:
            tail_style = adr.TailStyle.key

        if style == adr.Style.UH64:
            u64 = adr.hex_pack(pts, width, self.scheme)  # 3 metres - not great!
            strs = [''.join([f'{u:0x}' for u in v])[:width+4] for v in u64]
            if count < 2:
                return strs[0]
            return '\n'.join(strs)

        if style == adr.Style.UR64:
            u64 = adr.reg_pack(pts, width, self.registrar, self.scheme)  # 0.5 metres - ok!
            strs = [''.join([f'{u:016x}' for u in v])[:width+2] for v in u64]
            if count < 2:
                return strs[0]
            return '\n'.join(strs)

        if style == adr.Style.NUMERIC:
            u64 = adr.hex_pack(pts, width, self.scheme)
            strs = [''.join(f"{n:0x}" for n in row) for row in u64]
            if count < 2:
                return str(strs[0])
            return '\n'.join(n for n in strs)

        h9h = adr.hex_str_encode(pts, width, tail_style, self.scheme)
        if count < 2:
            return str(h9h[0])
        return '\n'.join(n for n in h9h)
