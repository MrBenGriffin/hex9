# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Composite Domains are a case where multiple domains are managed as a single group.
In this case we can have 8 octant-domains for the unit octahedron.

Component Domains are domains that belong to a composite domain.
"""

from functools import cache
import numpy as np
from abc import abstractmethod, ABC
from numpy.typing import NDArray
from .domain import Domain
from .point_format import PointFormat
from .points import Points
from hhg9.h9 import H9O


class CompositeDomain(Domain, ABC):
    """
        A case where multiple domains are managed as a single group.
        For example, 8 octants for a sphere.
    """

    def __init__(self, registrar, name: str, axes):
        super().__init__(registrar, name, axes)
        self.eps = 1e-22
        self.projs = {}  # component projections
        self.sides = {}  # component domains

    def binning(self, pts: Points, sig: tuple = None):
        """Identify the octant id of each point."""
        if self.axes == 3:
            from hhg9.base.points import OID_INVALID
            c = pts.coords.copy()
            finite_mask = np.isfinite(c).all(axis=1)
            oids = np.full(c.shape[0], OID_INVALID, dtype=np.uint8)

            if np.any(finite_mask):
                cf = c[finite_mask]
                oids[finite_mask] = self.seam_oid(cf < 0, cf == 0)
            pts.oid = oids
        else:
            raise NotImplementedError(f'{self.name} needs to bin {self.axes} axes')

    @staticmethod
    def seam_oid(neg: NDArray, free: NDArray) -> NDArray:
        """Derive octant ids under the on-seam mode-0 convention.

        The single place of derivation for the oid convention: octant bits
        come from coordinate signs (ECEF x, y, z -> bits 0, 1, 2, set when
        strictly negative). On a seam a coordinate is exactly 0 and its
        axis-bit is "free"; the design convention is that on-seam points
        belong to a mode-0 octant (popcount(oid) even). So instead of forcing
        zeros to the positive side, the free bit(s) are chosen to make the
        popcount even (free bits default to 0 and the highest is flipped up
        only when needed).

        Domains that bin in other spaces (e.g. OctahedralNet) resolve their
        own sign/seam criteria and hand over here — do not reimplement this
        rule elsewhere.

        Args:
            neg:  (N, 3) bool — coordinate strictly negative, per axis.
            free: (N, 3) bool — coordinate on the seam (axis-bit free), per
                  axis. ``neg`` must be False wherever ``free`` is True.
        Returns:
            (N,) uint8 octant ids.
        """
        oid = (
            (neg[:, 2].astype(np.uint8) << 2) |
            (neg[:, 1].astype(np.uint8) << 1) |
            neg[:, 0].astype(np.uint8)
        )
        odd = (np.bitwise_count(oid) & np.uint8(1)).astype(bool)
        fix = odd & free.any(axis=1)
        if np.any(fix):
            axes = np.array([0, 1, 2], dtype=np.int8)
            hi_free = np.where(free, axes, np.int8(-1)).max(axis=1)  # highest free axis
            rows = np.flatnonzero(fix)
            oid[rows] |= (np.uint8(1) << hi_free[rows].astype(np.uint8))
        return oid


    @cache
    def handlers(self):
        """
        Return the composite handlers indexed by oid (0-7).
        """
        _handlers = np.empty(8, dtype=object)
        for handler_instance in self.projs.values():
            _handlers[handler_instance.oid] = handler_instance
        return _handlers

    @cache
    def oc_c2(self):
        """
        Return array of c2 values indexed by oid (0-7).
        """
        data = np.empty((8, 3), dtype='<U3')
        for handler_instance in self.projs.values():
            data[handler_instance.oid] = handler_instance.oc
        return data

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        af.composite = self

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
        good = self.where_valid(pts)
        # Points(good, self) auto-calls binning via __init__
        return Points(good, self)


class ComponentDomain(Domain):

    def __init__(self, registrar, name: str, dom, oid, axes):
        super().__init__(registrar, name, axes)
        mode = H9O.oid_mo[oid]
        sign = H9O.oid_cmp[oid]
        self.oid = oid
        self.dom = dom
        self.mode = mode
        self.mode_str = 'V' if mode == 0 else 'Λ'
        self._sign = sign

    def sig(self) -> tuple:
        return self._sign

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        af.component[self.name] = self

    def adopt(self, pta: NDArray, only_valid=True):
        """
        Take an array and adopt as this domain.
        This is almost always not the right method to use.
        Far better to bin and then instantiate Points correctly.
        """
        coords = self.where_valid(pta) if only_valid else pta
        return Points(coords, domain=self.dom, oid=np.uint8(self.oid))
