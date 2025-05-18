"""
Part of the H9 project
This is the Plate Carrée Projection in GCD
GCD is recognised as [latitude,longitude]
"""
import numpy as np
from numpy.typing import NDArray
from hhg9.base import Domain, Points


class OutOfScope(Domain):

    def __init__(self, registrar):
        super().__init__(registrar, 'x_oos')

    def valid(self, pts: Points) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of points
        """
        return pts[pts.dom == 'x_oos']

