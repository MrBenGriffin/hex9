# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
PointFormat an abstract base class for add/use a Point formatting classes over their corresponding Domain.
"""
from abc import ABC, abstractmethod
import numpy as np


class PointFormat(ABC):
    """Base class for Point Formats"""

    def __init__(self, registrar, name: str):
        self.name = name
        self.composite = None
        self.component = {}
        if registrar is not None:
            registrar.register_format(self)
        self._set = None
        self.registrar = registrar

    @abstractmethod
    def is_valid(self, _) -> bool:
        """Check if the addresses is valid."""
        ...

    @abstractmethod
    def revert(self, _: str):
        """
        take a string (or representation) and return the addresses according to it's CoordinateSet
        :return:
        """
        ...

    @abstractmethod
    def format(self, _: np.ndarray, dom, sub: str) -> str:
        """
        Return the addresses formatted according to it's Domain
        :return:
        """
        ...
