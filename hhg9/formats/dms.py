"""
Part of the H9 project
"""
import re
import numpy as np
from numpy.typing import NDArray
from hhg9.base.point_format import PointFormat


class DMS(PointFormat):
    """Degrees/Minutes/Seconds"""
    def __init__(self):
        super().__init__('dms')

    def is_valid(self, address: str) -> bool:
        """Check if the format is valid."""
        dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"([NEWS])"
        return len(re.findall(dms_pattern, address)) == 2

    def revert(self, address: str):
        """
        take a string (or representation) and return the address according to it's CoordinateSet
        :return:
        """
        dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"([NEWS])"
        parts = re.findall(dms_pattern, address)
        if len(parts) != 2:
            raise ValueError("Invalid DMS format")

        def dms_to_dd(d, m, s, hemi):
            """component function"""
            dd = float(d) + float(m) / 60 + float(s) / 3600
            return -dd if hemi in "SW" else dd

        lat = dms_to_dd(*parts[0])
        lon = dms_to_dd(*parts[1])
        return self._set([[lat, lon]])

    def format(self, arr: NDArray, sub):
        """
        return decimal formatted address(es)
        :return:
        """

        def dms(dd, hemi_pos, hemi_neg):
            hemi = hemi_pos if dd >= 0 else hemi_neg
            dd = abs(dd)
            d = int(dd)
            m = int((dd - d) * 60)
            s = (dd - d - m / 60) * 3600
            return f"{d}°{m}'{s:.6f}\"{hemi}"

        result = []
        vals = np.array([arr]) if len(arr.shape) == 1 else arr
        for coords in vals:
            lat_dms = dms(coords[0], "N", "S")
            lon_dms = dms(coords[1], "E", "W")
            result.append(f"{lat_dms}, {lon_dms}")
        return '\n'.join(result)
