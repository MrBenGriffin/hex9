"""
Part of the H9 project
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

