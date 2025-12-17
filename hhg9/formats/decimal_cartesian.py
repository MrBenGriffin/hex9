# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base.point_format import PointFormat


class DecimalCartesian(PointFormat):
    def __init__(self, registrar):
        super().__init__(registrar, 'dec')

    def is_valid(self, address: str) -> bool:
        """Check if the addresses is valid."""
        return True

    def revert(self, arr: NDArray):
        """
        take a string (or representation) and return the addresses according to it's Domain
        :return:
        """
        return True

    def format(self, arr: NDArray, _, sub):
        """
        return decimal formatted addresses(es)
        :return:
        """
        # vals = np.array([arr]) if len(arr.shape) == 1 else arr
        if arr.ndim == 1 or arr.shape[0] == 1:
            return ''.join([f'{p[-3]:.7f}, {p[-2]:.7f}, {p[-1]:.7f}' for p in arr[..., :]])
        return '\n'.join([f'{p[-3]:.7f}, {p[-2]:.7f}, {p[-1]:.7f}' for p in arr[..., :]])
