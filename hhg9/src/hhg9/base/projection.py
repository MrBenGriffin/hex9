# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Projection is abstract base class for managing Points conversions.
For general use, this is not publicly referred to - as projections are normally managed by the Registrar.
"""
from abc import ABC
from numpy.typing import NDArray
from .points import Points


class Projection(ABC):
    """
    Abstract class that manages A<=>B coordinate set conversion.
    This will typically include two methods - forward and backward,
    which are inverses of each other but do not necessarily indicate
    alliance to one side or the other
    """

    def __init__(self, registrar, name: str, fwd, rev):
        self.name = name
        self.fwd_cs = registrar.domain(rev)
        self.rev_cs = registrar.domain(fwd)
        registrar.register_projection(self, [fwd, rev])

    def forward(self, pts: Points) -> NDArray:
        """Move from one coordinate set to another; the inverse of backward."""
        ...

    def backward(self, pts: Points) -> NDArray:
        """Move from one coordinate set to another; the inverse of forward."""
        ...
