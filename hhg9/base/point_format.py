"""
Part of the H9 project
"""
from abc import ABC, abstractmethod
import numpy as np
# from .points import Points


class PointFormat(ABC):
    """Base class for Point Formats"""

    def __init__(self, name: str):
        self.name = name
        self.composite = None
        self.component = {}
        self._set = None

    @abstractmethod
    def is_valid(self, _) -> bool:
        """Check if the address is valid."""
        ...

    @abstractmethod
    def revert(self, _: str):
        """
        take a string (or representation) and return the address according to it's CoordinateSet
        :return:
        """
        ...

    @abstractmethod
    def format(self, _: np.ndarray, sub: str) -> str:
        """
        Return the address formatted according to it's Domain
        :return:
        """
        ...
