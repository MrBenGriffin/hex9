"""
Part of the H9 project
"""
from functools import lru_cache

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
        self.no_zero = 1e-200
        self.components = {}  # class static

    def binning(self, pts: Points, sig: tuple = None):
        """Return points with domain set by composite set_domain. This can be overridden"""
        pts.coords = np.atleast_2d(pts.coords)
        pts.components = np.atleast_2d(np.sign(pts.coords + self.no_zero).astype(np.int8))
        return pts

    @lru_cache(maxsize=None)
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
        # keys = list(self.components.keys())
        # pts = Points(coords=np.zeros([8, self.axes]), domain=self, components=np.array(keys))
        # order, _ = pts.cm()
        # h = [r for r in range(len(order))]
        # for i, o in enumerate(order):
        #     h[o] = self.components[keys[i]]
        # _handlers = np.array(h)
        # return _handlers

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
    @abstractmethod
    def sig(self, ) -> tuple:
        """Return the composite signature that this component handles."""
        ...

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        af.component[self.name] = self

    def adopt(self, pta: NDArray, only_valid=True):
        """
        Take an array and adopt as this domain.
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
