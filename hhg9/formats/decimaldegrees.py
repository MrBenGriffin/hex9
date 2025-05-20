"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base.point_format import PointFormat


class DecimalDegrees(PointFormat):
    def __init__(self):
        super().__init__('decimal')

    def is_valid(self, address: str) -> bool:
        """Check if the address is valid."""
        return True

    def revert(self, arr: NDArray):
        """
        take a string (or representation) and return the address according to it's Domain
        :return:
        """
        return True

    def format(self, arr: NDArray, _, sub):
        """
        return decimal formatted address(es)
        :return:
        """
        vals = np.array([arr]) if len(arr.shape) == 1 else arr
        if len(vals.shape) == 1:
            return ''.join([f'{p[0]:.7f}, {p[1]:.7f}' for p in vals[..., :]])
        return '\n'.join([f'{p[0]:.7f}, {p[1]:.7f}' for p in vals[..., :]])
