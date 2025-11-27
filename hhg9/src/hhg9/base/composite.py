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


class CompositeDomain(Domain, ABC):
    """
        A case where multiple domains are managed as a single group.
        For example, 8 octants for a sphere.
    """

    def __init__(self, registrar, name: str, axes):
        super().__init__(registrar, name, axes)
        self.eps = 1e-22
        self.components = {}  # class static

    @classmethod
    def binning(cls, pts: Points, sig: tuple = None):
        """Identify the components of the points"""
        return pts.binning(sig)

    @cache
    def handlers(self):
        """
        Return the composite handlers in 'cm' order (see Points)
        While this might be a bit round-about it guarantees integrity with the cm values.
        """
        _handlers = np.empty(8, dtype=object)
        for sign_tuple, handler_instance in self.components.items():
            components_arr = np.array([sign_tuple])
            octant_id = Points.calc_octant_ids(components_arr)[0]
            _handlers[octant_id] = handler_instance
        return _handlers

    @cache
    def oc_c2(self):
        """
        Return array of c2 values in 'c2' order (see Points)
        """
        data = np.empty((8, 3), dtype='<U3')
        for sign_tuple, handler_instance in self.components.items():
            components_arr = np.array([sign_tuple])
            octant_id = Points.calc_octant_ids(components_arr)[0]
            data[octant_id] = handler_instance.oc
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
        pts = Points(good, self)
        pts.components = np.zeros((good.shape[0], 3), dtype='b')  # signed byte.  was using <U9 but seems crazy.
        return self.binning(pts)


class ComponentDomain(Domain):

    def __init__(self, registrar, name: str, dom, mode, sign: tuple, axes):
        super().__init__(registrar, name, axes)
        if mode is None:
            _, mode_a = Points.class_mode([sign])
            mode = mode_a[0]
        elif isinstance(mode, str):
            mode = 0 if mode == 'V' else 1
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
        if only_valid:
            good = self.where_valid(pta)
            pts = Points(good, domain=self.dom)
            pts.components = np.zeros((good.shape[0], 3), dtype='b')  # signed byte.  was using <U9 but seems crazy.
        else:
            pts = Points(pta, domain=self.dom)
            pts.components = np.zeros((len(pts), 3), dtype='b')
        pts.components += self.sig()
        return pts
