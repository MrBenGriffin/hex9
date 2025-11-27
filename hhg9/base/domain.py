# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
A Domain is a numerical coordinate space used inside Hex9.

It defines:
  • The coordinate axes (their order, units, and interpretation),
  • The region of that space that is considered valid (via `valid`),
  • The default point / address formats for representing those coordinates.

A Domain is **not** a CRS: it has no datum, no ellipsoid choice, and no external
projection baked in. Instead, Domains are stitched together by separate
Projection objects (e.g. g_gcd ↔ c_oct ↔ b_oct), which map between Domains
and thus between Hex9 spaces and real-world geodetic coordinates.

If we have some latitude/longitude coordinates in a numpy array, we use the 'g_gcd' Domain from Registrar,
and then use that to instantiate the coordinates into a Projection-ready Points classs.

reg=Registrar()
gcd_pts=Points(my_coords, reg.domain('g_gcd'))    # <-- domain tells Points what my_coords represent.
ell_pts=reg.project(gcd_pts, ['c_ell'])           # can now project lat-lon to ECEF xyz.

ellipsoidal surface of the WGS84 ellipsoid using a GCD projection.
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from .point_format import PointFormat


class Domain(ABC):
    """
    Base class for a Domain.
    This is not a CRS, because it does not include projections.
    """

    def __init__(self, registrar, name: str, axes):
        self.name = name
        self.axes = axes
        if registrar is not None:
            registrar.register_domain(self)
        self.default_format = None
        self.address_formats = {}  # class static

    def sig(self) -> str:
        """
        :return: my name.
        """
        return self.name

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat"""
        self.address_formats[af.name] = af
        af._set = self

    @abstractmethod
    def valid(self, _) -> NDArray:
        """Return a set of bools for those points which are valid in this set."""
        ...

    def is_valid(self, pts) -> bool:
        """
        :param pts: set of points for in this set.
        :return: that the points are all valid.
        """
        return np.all(self.valid(pts))

    def where_valid(self, pts):
        """
        :param pts: set of points for in this set.
        :return: those which are legal.
        """

        vx = self.valid(pts)
        return np.array(pts)[vx]

    def adopt(self, pts: NDArray):
        """
        Take an array and adopt as this domain.
        """
        from .points import Points
        good = self.where_valid(pts)
        return Points(good, self)

    def __repr__(self):
        return f'{self.name}'

