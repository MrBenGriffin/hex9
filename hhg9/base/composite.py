"""
Part of the H9 project
"""
import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from .domain import Domain
from .point_format import PointFormat
from .points import Points


class CompositeDomain(Domain):
    """
        A case where multiple domains are managed as a single group.
        For example, 8 octants for a sphere.
    """

    def binning(self, pts: Points, sig: tuple = None) -> dict:
        """Return points with domain set by composite set_domain. This is overridden"""
        return pts

    @abstractmethod
    def bins(self) -> dict:
        """Return a dict of key->cset that this is composed of."""
        ...

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
        return self.binning(good)


class ComponentDomain(Domain):
    @abstractmethod
    def sig(self, ) -> tuple:
        """Return the composite signature that this component handles."""
        ...

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        super().register_format(af)
        af.component[self.name] = self
