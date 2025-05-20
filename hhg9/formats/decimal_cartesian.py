"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base.point_format import PointFormat


class DecimalCartesian(PointFormat):
    def __init__(self):
        super().__init__('dec')

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
        # vals = np.array([arr]) if len(arr.shape) == 1 else arr
        if arr.ndim == 1 or arr.shape[0] == 1:
            return ''.join([f'{p[-3]:.7f}, {p[-2]:.7f}, {p[-1]:.7f}' for p in arr[..., :]])
        return '\n'.join([f'{p[-3]:.7f}, {p[-2]:.7f}, {p[-1]:.7f}' for p in arr[..., :]])
